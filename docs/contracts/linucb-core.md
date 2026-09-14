# Contratos: azuma-router core (slice 0001)

| Campo | Valor |
|---|---|
| Ficha | `proposal/tier-a/0001-azuma-linucb-core.md` |
| ADRs vinculantes | `docs/adr/0001` … `0018` |
| Vocabulario | `CONTEXT.md` |
| Rama | `feat/linucb-core` (vía `convoy spin`) |
| Estado | APROBADA (Fase 1, 2026-09-14) — requisitos normativos en `openspec/changes/linucb-core/specs/*/spec.md`; este fichero conserva los contratos exactos (TOML, tipos, payloads, edge cases, AC) |

Todo requisito lleva identificador `FR-nn` (funcional) o `NFR-nn` (no funcional) y se cruza con `tasks.md`. "DEBE"/"NUNCA" son normativos.

---

## 1. Problem Statement

Convoy (orquestador TS) necesita, para cada llamada LLM, elegir en < 1 ms el modelo que maximiza calidad–coste–latencia, aprendiendo online de la calidad observada y sin agotar las cuotas rodantes de los modelos bajo suscripción. azuma-router es un daemon Python/JAX, hijo de Convoy, que expone esa decisión por JSON-RPC 2.0 sobre stdio, persiste su aprendizaje en snapshots atómicos y registra cada decisión en un JSONL que sirve simultáneamente de WAL y de dataset para evaluación off-policy.

## 2. Out of Scope (follow-ups F1–F11 de la ficha)

Plugin TS de Convoy; LinUCB hybrid; Thompson Sampling; embeddings en el featurizer; GPU; multi-tenant; $\mu$ aprendido; reward a nivel de tarea multi-llamada; daemon distribuido; migración de snapshots; hot-reload de brazos.

---

## 3. Functional Requirements

### 3.1 Configuración (ADR-0012, 0017)
- **FR-01** Dado un `azuma.toml` válido, `load_config(path)` DEBE devolver un `Config` inmutable con brazos indexados por orden de aparición.
- **FR-02** Dado un TOML con `arm_id` duplicado, sección obligatoria ausente, tipo incorrecto, valor numérico no finito o con `|v| > 2³¹ − 1` (fuera del rango seguro en float32), `billing_mode = "subscription"` sin ventanas, o `billing_mode = "per_token"` con ventanas o `barrier_mu > 0`, DEBE lanzar `ConfigError` con la ruta del campo (`arms[2].windows[0].capacity`).
- **FR-03** `K = len(arms)` DEBE cumplir `1 ≤ K ≤ K_max`; `W = max(len(arm.windows))` DEBE cumplir `W ≤ W_max`.

### 3.2 Featurizer (ADR-0003, 0004, 0015)
- **FR-04** Dado un `RoutingRequest` válido, `featurize(request) -> f64[32]` DEBE ser determinista y cumplir la tabla de bloques de ADR-0015 exactamente.
- **FR-05** Dado `task_kind` fuera del enum, un conteo negativo, `agent_depth < 0`, o un valor no finito, DEBE lanzar `InvalidContextError`; NUNCA sustituir por defecto.
- **FR-06** Todo componente de salida DEBE estar en $[0,1]$; el featurizer DEBE verificarlo antes de devolver.

### 3.3 Bandido (ADR-0001, 0002, 0008)
- **FR-07** `init_state(config) -> BanditState` DEBE devolver `A[a] = ridge_lambda · I` para todo `a`, `b = 0`, `n_pulls = 0`, `mask[a] = True` para `a < K` y `False` para el padding, `rng_key = PRNGKey(seed)`.
- **FR-08** `select(state, x, quota_view, static) -> (state', arm, propensity, diagnostics)` DEBE calcular por brazo `exploit = x·θ̂`, `bonus = alpha·sqrt(x·A⁻¹x)`, `shadow = max_W −μ_a·ln(1−u)`, `score = exploit + bonus − shadow·est_units`, enmascarar con `−inf`, y muestrear con $\pi = (1-\varepsilon)\,\mathrm{softmax}(\text{score}/\tau) + \varepsilon/K_{act}$.
- **FR-09** `select` DEBE ser una función pura: mismo `(state, x, quota_view)` ⇒ misma salida bitwise. `state'.rng_key` DEBE diferir de `state.rng_key`; todo lo demás igual.
- **FR-10** Si `quota_view.mask_override` (derivado de `allowed_arms`) y la máscara del estado dejan 0 brazos activos, `select` DEBE señalar `no_arm_available = True` en `diagnostics` y el daemon DEBE responder `-32004`.
- **FR-11** `update(state, x, arm, reward, static) -> state'` DEBE producir `A'[arm] = A[arm] + x xᵀ`, `b'[arm] = b[arm] + reward·x`, `n_pulls'[arm] += 1`, y dejar los demás brazos bitwise iguales.
- **FR-12** `A` DEBE permanecer simétrica bitwise y SPD tras cualquier secuencia de `update`; `‖A θ̂ − b‖∞ < 1e-9` en float64.
- **FR-13** Cada función compilada (`select_jit`, `update_jit`) DEBE compilarse **exactamente una vez** por proceso para un `StaticConfig` dado.

### 3.4 Cuota y precio sombra (ADR-0005, 0006, 0011)
- **FR-14** `QuotaLedger.record(arm_id, units, ts)` y `QuotaLedger.utilization(now) -> f32[K_max, W_max]` DEBEN dar `u = consumido_en_ventana / capacity`, con `RollingWindow` (últimos `duration_s`) y `FixedResetWindow` (desde el último `reset_anchor`).
- **FR-15** `u ≥ 1.0` en cualquier ventana ⇒ el brazo DEBE quedar enmascarado en la `QuotaView` de ese `select`.
- **FR-15b** El ledger DEBE retener cada evento al menos `max(duración de la ventana más larga del brazo, pending_ttl_s)` (para brazos `per_token`, sin ventanas: `pending_ttl_s`) y podar los más antiguos en cada `utilization` para todo brazo, para que `correct` en `report` siempre encuentre su `ledger_event_id`; un `report` cuyo evento no existe es una violación de invariante ⇒ `ledger_integrity_error`, salida código 4 **antes** de escribir la línea `report`.
- **FR-16** El consumo estimado (`est_units = units_per_request + units_per_1k_tokens·(input_tokens + expected_output_tokens)/1000`) DEBE anotarse en `select`; `report` DEBE reemplazarlo por el consumo real (`actual_*_tokens`) en el mismo evento del ledger (no sumar un segundo).
- **FR-17** Brazos `per_token` DEBEN tener `shadow = 0` siempre.

### 3.5 Recompensa (ADR-0007)
- **FR-18** `compute_reward(quality, cost_usd, latency_s, static) = clip(w_q·q − w_c·min(cost/cost_budget,1) − w_l·min(lat/latency_budget,1), 0, 1)`.
- **FR-19** Brazos `subscription` DEBEN reportar `cost_usd = 0`; un `report` con `cost_usd > 0` para un brazo `subscription` DEBE responder `-32602`.

### 3.6 WAL y PendingStore (ADR-0011, 0016)
- **FR-20** Antes de responder a `select`, el daemon DEBE haber hecho `append + flush` de una línea `phase="select"` en el JSONL.
- **FR-21** `report` para `decision_id` pendiente DEBE, en este orden: escribir `phase="report"`, corregir el ledger, aplicar `update`, mover a aplicado. Duplicado ⇒ `{"applied": false, "reason": "duplicate"}` sin escritura.
- **FR-22** Pendientes con `now − ts_select > pending_ttl_s` DEBEN expirar en el barrido (inicio de `select` y `snapshot`): línea `phase="expire"`, sin `update`, consumo estimado del ledger se mantiene.
- **FR-23** Al arrancar, el daemon DEBE reconstruir `PendingStore` por replay del JSONL (fichero activo + anterior rotado): **toda** línea `select` retenida sin `report`/`expire` posterior es un pendiente vivo, sea cual sea su `state_version` (los `select` no lo avanzan); el ledger se reconstruye de forma idempotente por `ledger_event_id` (evento ya presente en el snapshot ⇒ no se re-registra). El filtro `state_version > snapshot.state_version` aplica solo a la reaplicación de líneas `report` (FR-24). Un `report` sin `select` retenido ⇒ journal corrupto ⇒ salida código 3.
- **FR-24** `report` de `decision_id` desconocido ⇒ `-32002`; expirado ⇒ `-32003`.

### 3.7 Snapshot (ADR-0009, 0017)
- **FR-25** `snapshot(state, ledger, meta, dir)` DEBE escribir `state.safetensors` + `state.json` en `<dir>/snapshot.tmp/`, `fsync` ambos, `fsync` del directorio, `rename` a `<dir>/snapshot/` (reemplazando el anterior de forma atómica) y `fsync` del directorio padre antes de borrar `snapshot.old/`. Un `OSError` en cualquier paso ⇒ `snapshot_write_error`, salida código 4. `config_sha256` = SHA-256 del JSON canónico del `Config` validado completo.
- **FR-26** `restore(dir, config) -> (state, ledger, meta)` DEBE lanzar `IncompatibleSnapshotError` si difiere cualquiera de: `schema_version`, `d`, `K_max`, `W_max`, `featurizer_version`, `arm_ids`. Nunca migrar.
- **FR-27** `restore(snapshot(s)) == s` bitwise para todos los arrays.
- **FR-28** El daemon DEBE hacer snapshot cuando `updates_since_snapshot ≥ 100` **o** `now − last_snapshot ≥ 30 s` (evaluado tras cada `report`) y en shutdown limpio (EOF en stdin o SIGTERM/SIGINT). Una señal recibida mientras una petición está en curso se **difiere** hasta que esa petición ha respondido (`shutdown_deferred`); nunca se hace snapshot a mitad de un `report`.

### 3.8 Daemon JSON-RPC (ADR-0010, 0013, 0014)
- **FR-29** El daemon lee líneas de `stdin`, una petición JSON-RPC 2.0 por línea, y escribe una respuesta por línea en `stdout`; `stderr` solo logs JSON estructurados.
- **FR-30** Tras arrancar, cargar config, restaurar snapshot, replay del WAL y **warm-up JIT** (una llamada real a `select_jit` y `update_jit`), DEBE emitir la notificación `ready`. Toda petición anterior a `ready` ⇒ `-32005`.
- **FR-31** Métodos: `select`, `report`, `snapshot`, `health`. Método desconocido ⇒ `-32601`.
- **FR-32** Peticiones se procesan **secuencialmente** en orden de llegada; el daemon es single-thread.
- **FR-33** Batch JSON-RPC (array) NO se soporta ⇒ `-32600`.

### 3.9 OPE (ADR-0008)
- **FR-34** `ope.evaluate(log_path, candidate_static) -> {ips, dr, n, ess}` DEBE calcular IPS y DR sobre líneas `report` emparejadas con su `select`, usando la propensity registrada como $\mu$ y la política candidata evaluada sobre el $x_t$ registrado.

---

## 4. Data Models

### 4.1 `azuma.toml`

```toml
schema_version = 1                      # int, obligatorio, DEBE ser 1

[bandit]
seed              = 42                  # int ≥ 0
k_max             = 32                  # int, 1..256; forma estática
w_max             = 4                   # int, 1..8; forma estática
d                 = 32                  # int, DEBE ser 32 en featurizer v1
ridge_lambda      = 1.0                 # float > 0
alpha             = 1.0                 # float ≥ 0
tau               = 0.05                # float ≥ 1e-6 (temperatura; por debajo el softmax degenera en float64)
epsilon_floor     = 0.02                # float en [0, 0.5]
w_quality         = 1.0                 # float ≥ 0
w_cost            = 0.3                 # float ≥ 0
w_latency         = 0.2                 # float ≥ 0
cost_budget_usd   = 0.50                # float > 0; coste que normaliza a 1
latency_budget_s  = 30.0                # float > 0
pending_ttl_s     = 3600                # int > 0
snapshot_dir      = "./.azuma"          # str; se crea si no existe
snapshot_every_updates = 100            # int > 0
snapshot_every_s       = 30             # int > 0
decision_log_path      = "./.azuma/decisions.jsonl"
decision_log_max_bytes = 268435456      # int > 0 (256 MiB)

[[arms]]
model_id          = "gpt-5-codex"       # str, no vacío, sin '@'
provider          = "openai-codex"      # str, no vacío, sin '@'
billing_mode      = "subscription"      # "subscription" | "per_token"
price_in_per_1m   = 0.0                 # float ≥ 0 (USD); DEBE ser 0 si subscription
price_out_per_1m  = 0.0
units_per_request = 1.0                 # float ≥ 0; solo subscription
units_per_1k_tokens = 0.0               # float ≥ 0; solo subscription
barrier_mu        = 0.5                 # float ≥ 0; DEBE ser 0 si per_token

  [[arms.windows]]                      # 1..w_max entradas; obligatorio si subscription, prohibido si per_token
  kind        = "rolling"               # "rolling" | "fixed_reset"
  duration_s  = 18000                   # int > 0; solo rolling (5 h)
  capacity    = 300.0                   # float > 0 y ≤ 2³¹ − 1 (FR-02), en units

  [[arms.windows]]
  kind         = "fixed_reset"
  period       = "weekly"               # "daily" | "weekly"; solo fixed_reset
  reset_anchor = "2026-09-08T00:00:00Z"  # RFC 3339 UTC; el periodo se cuenta desde aquí
  capacity     = 1500.0

[[arms]]
model_id         = "qwen3-coder"
provider         = "openrouter"
billing_mode     = "per_token"
price_in_per_1m  = 0.30
price_out_per_1m = 1.20
```

Derivados: `arm_id = f"{model_id}@{provider}"`, único. Índice `a` = posición en `[[arms]]`.

### 4.2 Tipos Python (contratos; nombres exactos)

```
StaticConfig (frozen dataclass, hashable)     # ADR-0001
  k_max: int, w_max: int, d: int
  ridge_lambda, alpha, tau, epsilon_floor: float
  w_quality, w_cost, w_latency, cost_budget_usd, latency_budget_s: float
  barrier_mu: tuple[float, ...]               # len == k_max, 0.0 en padding y per_token
  is_subscription: tuple[bool, ...]           # len == k_max

BanditState (NamedTuple)                      # ADR-0001/0002/0008
  A: f64[k_max, d, d]
  b: f64[k_max, d]
  n_pulls: i32[k_max]
  mask: bool[k_max]                           # False = padding o brazo deshabilitado
  rng_key: u32[2]

QuotaView (NamedTuple)                        # ADR-0005
  utilization: f32[k_max, w_max]              # 0.0 en slots sin ventana
  est_units: f32[k_max]                       # consumo estimado de ESTA petición por brazo
  allowed: bool[k_max]                        # de allowed_arms; True si no se envió

Diagnostics (NamedTuple)
  exploit, bonus, shadow_price, score, policy: f64[k_max]
  no_arm_available: bool

RoutingRequest (frozen dataclass)             # ADR-0013
  task_kind: TaskKind (enum de 12)
  input_tokens: int ≥ 0
  expected_output_tokens: int | None ≥ 0
  has_code, has_tools, has_images: bool
  agent_depth: int ≥ 0
  diff_lines: int | None ≥ 0
  priority: Priority (low | normal | high)

Pending (frozen dataclass)                    # ADR-0016
  decision_id: str (UUIDv7), arm_index: int, x: f64[32], propensity: float,
  ts_select: float, est_units: float, ledger_event_id: int, state_version: int,
  select_entry: dict          # línea `select` del journal, arrastrada íntegra en la rotación del WAL

SnapshotMeta (frozen dataclass)               # ADR-0009
  schema_version: int (=1), state_version: int, featurizer_version: int (=1),
  arm_ids: list[str], static: StaticConfig, seed: int, created_at: str (RFC 3339),
  ledger: LedgerDump, config_sha256: str
```

Estado de una decisión (máquina de estados, ADR-0011):

```
select ──► PENDING ──report──► APPLIED (terminal)
              └────ttl────► EXPIRED (terminal)
report sobre APPLIED  → {applied:false, reason:"duplicate"}
report sobre EXPIRED  → error -32003
report sobre inexistente → error -32002
```

### 4.3 Snapshot en disco

```
<snapshot_dir>/
  snapshot/                  # publicado atómicamente por rename de directorio
    state.safetensors        # tensores: A, b, n_pulls, mask, rng_key (nombres exactos)
    state.json               # SnapshotMeta
  snapshot.tmp/              # solo durante la escritura; se borra si existe al arrancar
  decisions.jsonl            # WAL activo
  decisions.1.jsonl          # rotado anterior (máx. 1)
```

### 4.4 DecisionLog JSONL (una línea por evento; claves en este orden)

```jsonc
// phase = "select"
{"phase":"select","decision_id":"018f...","ts":1757800000.123,"state_version":41,
 "arm_index":3,"arm_id":"gpt-5-codex@openai-codex","propensity":0.8123,
 "x":[1.0,0.54,...],                       // 32 floats
 "policy":[0.01,0.8123,...],               // 32 floats (0 en enmascarados)
 "score":[...],"exploit":[...],"bonus":[...],"shadow_price":[...],
  "utilization":{"gpt-5-codex@openai-codex":[0.42,0.10]},
 "est_units":2.1,"ledger_event_id":42,"featurizer_version":1}

// phase = "report"
{"phase":"report","decision_id":"018f...","ts":1757800012.9,"state_version":42,
 "quality":0.9,"cost_usd":0.0,"latency_s":11.2,
 "actual_input_tokens":1900,"actual_output_tokens":640,"reward":0.8253,"actual_units":2.54}

// phase = "expire"
{"phase":"expire","decision_id":"018f...","ts":1757803601.0,"state_version":57,"ts_select":1757800000.123}
```

`state_version` en `report` es el valor **tras** aplicar `update` (monótono, +1 por update). Escritura: `json.dumps(..., separators=(",",":"))` + `"\n"` + `flush()`.

---

## 5. API Contracts (JSON-RPC 2.0 sobre stdio)

Framing: UTF-8, un objeto JSON por línea terminada en `\n`, sin `Content-Length`. `id` entero o string; notificaciones sin `id`.

### 5.1 `ready` (notificación daemon → cliente)
```json
{"jsonrpc":"2.0","method":"ready","params":{"state_version":41,"arms":["gpt-5-codex@openai-codex","qwen3-coder@openrouter"],"restored_from_snapshot":true,"pending_replayed":3,"warmup_ms":812.4}}
```

### 5.2 `select`
```jsonc
// request
{"jsonrpc":"2.0","id":1,"method":"select","params":{
  "request":{"task_kind":"code_edit","input_tokens":1834,"expected_output_tokens":600,
             "has_code":true,"has_tools":true,"has_images":false,
             "agent_depth":2,"diff_lines":120,"priority":"normal"},
  "allowed_arms":["gpt-5-codex@openai-codex"],   // opcional
  "verbose":true                                 // opcional, default true
}}
// response
{"jsonrpc":"2.0","id":1,"result":{
  "decision_id":"018f6c2e-...",
  "arm":{"arm_id":"gpt-5-codex@openai-codex","model_id":"gpt-5-codex","provider":"openai-codex","billing_mode":"subscription"},
  "propensity":0.8123,
  "diagnostics":{                                 // ausente si verbose=false; score, exploit, bonus, shadow_price de brazos enmascarados son null
    "score":{"gpt-5-codex@openai-codex":0.71,"qwen3-coder@openrouter":0.44},
    "exploit":{...},"bonus":{...},"shadow_price":{...},"policy":{...},
    "utilization":{"gpt-5-codex@openai-codex":{"rolling:18000":0.42,"fixed_reset:weekly":0.10}},
    "est_units":{"gpt-5-codex@openai-codex":2.1}
  }
}}
```
Errores: `-32602` (params inválidos, con `data.field`), `-32001 InvalidContext` (`data.field`, `data.reason`), `-32004 NoArmAvailable` (`data.masked_reason` por brazo: `"quota_exhausted" | "not_allowed" | "disabled"`), `-32005 NotReady`.

### 5.3 `report`
```jsonc
// request
{"jsonrpc":"2.0","id":2,"method":"report","params":{
  "decision_id":"018f6c2e-...","quality":0.9,"cost_usd":0.0,"latency_s":11.2,
  "actual_input_tokens":1900,"actual_output_tokens":640}}
// response (aplicado)
{"jsonrpc":"2.0","id":2,"result":{"applied":true,"reward":0.8253,"state_version":42}}
// response (duplicado)
{"jsonrpc":"2.0","id":2,"result":{"applied":false,"reason":"duplicate","state_version":42}}
```
Validación: `quality ∈ [0,1]` obligatoria; `cost_usd ≥ 0`, `latency_s ≥ 0`, tokens `≥ 0` enteros; `cost_usd > 0` en brazo subscription ⇒ `-32602`. Errores: `-32002 UnknownDecision`, `-32003 DecisionExpired` (`data.expired_at`), `-32005`.

### 5.4 `snapshot`
```json
{"jsonrpc":"2.0","id":3,"method":"snapshot","params":{}}
{"jsonrpc":"2.0","id":3,"result":{"state_version":42,"path":"./.azuma/snapshot","bytes":270336,"expired_pending":0}}
```

### 5.5 `health`
```json
{"jsonrpc":"2.0","id":4,"method":"health","params":{}}
{"jsonrpc":"2.0","id":4,"result":{"ready":true,"state_version":42,"pending":2,"updates_since_snapshot":7,
 "arms":{"gpt-5-codex@openai-codex":{"mask":true,"n_pulls":31,"utilization":[0.42,0.10]}},
 "jit_compilations":{"select":1,"update":1}}}
```

### 5.6 Errores

| code | name | cuándo |
|---|---|---|
| −32700 | ParseError | línea no es JSON estricto (incluye `NaN`/`Infinity` y anidamiento más profundo que el decodificador); `id` siempre `null`; el bucle continúa |
| −32600 | InvalidRequest | falta `jsonrpc:"2.0"`/`method`, batch, o `id` que no es string/entero/null (`data.reason = "invalid_id"`, se responde con `id: null`) |
| −32601 | MethodNotFound | |
| −32602 | InvalidParams | tipo/rango; `data.field`. Incluye `params` presente y no objeto (`data.field = "params"`, se responde con el `id` de la petición) |
| −32001 | InvalidContext | featurizer rechaza; `data.field`, `data.reason` |
| −32002 | UnknownDecision | |
| −32003 | DecisionExpired | |
| −32004 | NoArmAvailable | |
| −32005 | NotReady | antes de `ready` |
| −32006 | IncompatibleSnapshot | solo en arranque; el daemon escribe `incompatible_snapshot` en stderr y sale con código 3 |

---

## 6. Edge Cases and Failure Modes

| Escenario | Comportamiento |
|---|---|
| `K = 1` | `select` devuelve siempre ese brazo con `propensity = 1.0` (softmax de un elemento + suelo = 1). |
| Todos los brazos con `u ≥ 1` | `-32004` con `masked_reason` por brazo; no se escribe línea `select`. |
| `allowed_arms` con `arm_id` desconocido | `-32602`, `data.field = "allowed_arms[i]"`. |
| `expected_output_tokens` ausente | $x_2 = 0$; `est_units` usa solo `input_tokens`. |
| `input_tokens = 0` | Válido; $x_1 = 0$. |
| `input_tokens ≥ 10^6` | $x_1 = 1$ (clip). |
| Línea stdin vacía | Ignorada silenciosamente. |
| Línea > 1 MiB | `-32600`, `data.reason = "line_too_long"`. |
| `id` ausente en `select`/`report` (notificación) | Se procesa pero no se responde (semántica JSON-RPC); se registra en stderr. Un `"id": null` explícito **no** es notificación: se responde con `id: null`. |
| `report` llega antes de que `select` haya hecho flush | Imposible: `select` responde solo tras flush (FR-20). |
| Crash entre línea `select` y respuesta | Al arrancar, replay ve `select` sin `report`: pendiente; expirará o se reportará. |
| Crash entre línea `report` y `update` aplicado | Replay reaplica `report` (línea existe, `state_version` del snapshot menor): `update` es idempotente por replay porque solo se reaplica si `state_version_line > snapshot.state_version`. Los `report` reaplicados DEBEN ser contiguos (`snapshot.state_version + 1, + 2, …`); un hueco ⇒ `replay_failed`, salida 3. |
| Excepción inesperada tras escribir la línea `report` (`ledger.correct`, `update_jit`, `pending`) | La línea ya es durable: `journal_apply_error`, salida 4 **sin** snapshot; el replay al arrancar deja memoria y journal reconciliados. |
| Snapshot `.tmp` residual al arrancar | Se borra; se restaura `snapshot/` si existe. |
| `snapshot/` ausente | Estado inicial (FR-07); `ready.restored_from_snapshot = false`; un `decisions.1.jsonl` presente sin `snapshot/` es inconsistente (la rotación siempre sigue a un snapshot) ⇒ `wal_inconsistent`, salida código 3; un `decisions.jsonl` sin rotar se reproduce íntegro desde el estado inicial (caso: kill antes del primer snapshot de cadencia). |
| `arm_ids` del snapshot ≠ config | `IncompatibleSnapshotError`, salida código 3, mensaje con el diff de listas. |
| Snapshot o WAL ilegibles (permisos, E/S) en arranque | `snapshot_read_failed` / `replay_failed` en stderr, salida código 3; nunca un traceback. Un `azuma.toml` ilegible ⇒ `config_error` (`file`), salida 3. |
| Snapshot con tensores o ledger que no decodifican (tras pasar versión y formas) | `incompatible_snapshot`, salida código 3. |
| `config_sha256` ≠ pero `arm_ids` y formas iguales (cambió `alpha`, pesos…) | Se acepta: hiperparámetros son valor, no forma; se anota `warning` en stderr. |
| JSONL con línea corrupta al final (crash a mitad de escritura) | Replay descarta **solo la última línea** si no parsea; una línea corrupta no final ⇒ salida código 3. |
| Rotación durante replay | No aplica: la rotación se evalúa tras cada `append` (`select`/`report`/`expire`) en servicio, nunca en replay; va precedida de snapshot y arrastra los `select` vivos al fichero nuevo. Los `expire` escritos al inicio de un `select` cuentan aunque ese `select` acabe rechazado. |
| `NaN` en `A` (imposible por construcción) | `select` DEBE assert `jnp.isfinite(score).all()` fuera del JIT sobre `diagnostics`; fallo ⇒ excepción y salida código 4 (estado corrupto; no se hace snapshot). |
| SIGTERM durante escritura de snapshot | El rename es atómico: o queda el anterior o el nuevo, nunca mixto. |
| SIGTERM/SIGINT con una petición en curso | Se registra `signal_received` + `shutdown_deferred`; la petición responde y después se cierra (snapshot + exit). Toda señal que llegue con un cierre ya iniciado (ordenado o fatal) se ignora: nunca hay un segundo `close` ni un snapshot que el cierre fatal prohíbe. |
| Fallo de E/S persistente (WAL abrir/escribir/rotar, snapshot) | `journal_write_error` ⇒ salida 4 **con** snapshot de cierre; `snapshot_write_error` ⇒ salida 4 sin reintentar el snapshot. Nunca se responde OK a una petición cuya línea no llegó al journal. |
| Respuesta con `NaN`/`Infinity` (serialización) | Imposible por construcción (`allow_nan=False`); si ocurre ⇒ `numerical_integrity_error`, salida 4 sin snapshot. |
| Reloj retrocede (NTP) | El ledger usa `time.time()`; un salto atrás puede rejuvenecer eventos; se acepta (ventanas se recuperan solas). Se registra `warning` si `now < last_ts − 1 s`. |
| `select` en brazo cuyo `barrier_mu = 0` pero con ventana | `shadow = 0` pero `u ≥ 1` sigue enmascarando (FR-15). |

---

## 7. Acceptance Criteria (por FR; se convierten 1:1 en tests en `tasks.md`)

- [ ] AC-01 `load_config(fixtures/azuma.valid.toml)` devuelve 2 brazos con `arm_id` esperados (FR-01).
- [ ] AC-02 Cada fixture inválida de `fixtures/config/invalid/*.toml` lanza `ConfigError` con la ruta de campo esperada (FR-02, FR-03).
- [ ] AC-03 `featurize` de la petición de ejemplo devuelve exactamente el vector de referencia `fixtures/featurizer/v1_reference.json` (FR-04).
- [ ] AC-04 `task_kind="unknown"`, `input_tokens=-1`, `agent_depth=-1` lanzan `InvalidContextError` (FR-05).
- [ ] AC-05 Hypothesis: para toda petición válida, `0 ≤ featurize(r) ≤ 1` y `‖x‖₂ ≤ √32` (FR-06).
- [ ] AC-06 `init_state` cumple FR-07 bitwise.
- [ ] AC-07 Con `A = I, b = 0` y `alpha = 1`, `bonus[a] = ‖x‖` para todo brazo activo (FR-08).
- [ ] AC-08 Con `u = 0` en todas las ventanas, `shadow_price = 0`; con `u = 0.9, μ = 1`, `shadow_price = −ln(0.1)` ± 1e-9 (FR-08).
- [ ] AC-09 Dos llamadas a `select` con el mismo estado devuelven salida bitwise idéntica; `rng_key` cambia (FR-09).
- [ ] AC-10 `allowed = all False` ⇒ `no_arm_available = True` (FR-10).
- [ ] AC-11 `update` modifica solo el brazo elegido; Hypothesis sobre 1 000 updates: `A` simétrica bitwise, `cho_factor` no falla, `‖Aθ̂ − b‖∞ < 1e-9` (FR-11, FR-12).
- [ ] AC-12 Tras 10 000 llamadas alternando `select`/`update`, `jax_compilations(select) == 1` y `(update) == 1` (FR-13).
- [ ] AC-13 Ledger con reloj inyectado: `RollingWindow` olvida eventos tras `duration_s`; `FixedResetWindow` se vacía en el anchor (FR-14).
- [ ] AC-14 `u = 1.0` ⇒ `QuotaView.utilization ≥ 1` y máscara efectiva False (FR-15).
- [ ] AC-15 `record` estimado + `correct(event_id, actual)` deja exactamente un evento con el valor real (FR-16).
- [ ] AC-16 `compute_reward(0.9, 0.0, 11.2)` con los defaults del TOML de ejemplo = 0.8253 ± 1e-9 (FR-18).
- [ ] AC-17 WAL: tras `select`, el fichero contiene la línea `select` **antes** de que `select` retorne (se verifica con un `flush` espiado) (FR-20).
- [ ] AC-18 `report` dos veces ⇒ 1 línea `report`, segunda respuesta `applied:false` (FR-21).
- [ ] AC-19 Con `pending_ttl_s = 1` y reloj inyectado, el siguiente `select` escribe `expire` y `n_pulls` no cambia (FR-22).
- [ ] AC-20 Replay de un JSONL fixture con `select` sin `report` reconstruye 1 pendiente; con `report` posterior, 0 (FR-23).
- [ ] AC-21 `report` desconocido ⇒ `-32002`; expirado ⇒ `-32003` (FR-24).
- [ ] AC-22 Tras `snapshot`, existen `snapshot/state.safetensors` y `snapshot/state.json`, no existe `snapshot.tmp/` (FR-25).
- [ ] AC-23 Snapshot con `arm_ids` distinto, `d` distinto, `featurizer_version` distinto ⇒ `IncompatibleSnapshotError` (FR-26).
- [ ] AC-24 Round-trip bitwise (FR-27).
- [ ] AC-25 Daemon end-to-end por subprocess: recibe `ready`, `health.ready == true`, `select` → `report` → `health.state_version == 1` (FR-29–FR-31).
- [ ] AC-26 Petición antes de `ready` (inyectada en test unitario del loop) ⇒ `-32005` (FR-30).
- [ ] AC-27 Batch `[{...}]` ⇒ `-32600` (FR-33).
- [ ] AC-28 OPE sobre `fixtures/ope/decisions.jsonl` con la misma política que lo generó: `ips ≈ dr ≈ media de reward` ± 0.02 (FR-34).

## 8. Non-Functional Requirements

- **NFR-01 Latencia**: `select_jit` p99 < 1 ms y `update_jit` p99 < 0.5 ms en CPU (`K_max=32, d=32, W_max=4`), medidos con `pytest-benchmark` tras warm-up, ≥ 1 000 iteraciones. Round-trip completo del daemon (`select` incl. featurizer, WAL flush y JSON) p99 < 3 ms en el mismo host.
- **NFR-02 Warm-up**: tiempo desde arranque hasta `ready` < 5 s en CPU arm64/x86_64 (se reporta en `ready.warmup_ms`).
- **NFR-03 Memoria**: estado ≤ 32·32·32·8 B ≈ 262 KiB por copia; el daemon retiene ≤ 2 copias (actual + snapshot en curso).
- **NFR-04 Observabilidad**: stderr en JSON (`ts`, `level`, `event`, campos); eventos mínimos: `ready`, `snapshot_written`, `pending_expired`, `report_duplicate`, `clock_skew`, `signal_received`, `shutdown_deferred`.
- **NFR-05 Tipado**: `mypy --strict` sin errores; `jaxtyping` NO se usa (anotaciones de forma en docstrings).
- **NFR-06 Determinismo**: mismo `seed` + misma secuencia de peticiones y reports ⇒ mismo JSONL bitwise (excepto `ts`).
- **NFR-07 Seguridad**: nunca `pickle`; `safetensors` para arrays; el daemon no abre sockets.
