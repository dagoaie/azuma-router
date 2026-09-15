## Context

Primer change del repositorio. Implementa el slice 0001 de la ficha `proposal/tier-a/0001-azuma-linucb-core.md`: un router LinUCB disjunto compilado en JAX, con cuotas y precios sombra, expuesto como daemon JSON-RPC por stdio. Las 18 decisiones tácticas están cerradas en `docs/adr/0001…0018` (Grill R1 y R2 aprobados el 2026-09-14); este documento las proyecta sobre módulos, la frontera JIT y las secuencias del daemon. Contratos con tipos exactos, payloads y edge cases: `docs/contracts/linucb-core.md`.

Restricciones:
- Python 3.12 (pin `uv`), `jax`/`jaxlib` ≥ 0.11.1, `safetensors`; sin dependencias de red ni NLP.
- `mypy --strict`, `ruff`, TDD por tarea; el andamiaje T00 ya está commitado (`738d62c`, bootstrap fuera del pipeline; ver `proposal/tier-a/0001-plan.md`).
- Hard-cut: sin migraciones, sin dual-path, sin compatibilidad con estados anteriores.

## Goals / Non-Goals

**Goals**
- `select_jit` p99 < 1 ms y `update_jit` p99 < 0.5 ms en CPU con `k_max = 32, d = 32, w_max = 4`; una única compilación por proceso.
- Aprendizaje persistente y recuperable: snapshot atómico + journal WAL con replay idempotente.
- Contrato de protocolo estable para el futuro plugin TS de Convoy.
- Journal utilizable como dataset OPE con propensities no degeneradas.

**Non-Goals** (follow-ups F1–F14, ver abajo)
- Plugin TS, transportes UDS/HTTP/gRPC, LinUCB hybrid, Thompson Sampling, embeddings, GPU, multi-tenant, `μ` aprendido, reward multi-llamada, daemon distribuido, migración de snapshots, hot-reload de brazos.

## Decisions

Cada decisión cita el ADR que la fija; aquí se describe **cómo** se materializa.

### D1 — Estado inmutable `BanditState(NamedTuple)` con formas estáticas (ADR-0001)
`A: f64[k_max,d,d]`, `b: f64[k_max,d]`, `n_pulls: i32[k_max]`, `mask: bool[k_max]`, `rng_key: u32[2]`. `k_max`, `d`, `w_max` viven en `StaticConfig` (frozen dataclass hashable, con `barrier_mu` e `is_subscription` como tuplas) y entran al JIT por `static_argnums`. Los brazos reales `K ≤ k_max` se padean; `mask` los distingue. Alternativa rechazada: arrays dinámicos por `K` ⇒ recompilación por cambio de forma.

### D2 — Solve por Cholesky, no `jnp.linalg.solve` ni Sherman–Morrison (ADR-0002)
`L = cho_factor(A_a)`, `θ̂ = cho_solve(L, b_a)`, `z = solve_triangular(L, x, lower=True)` ⇒ `xᵀA⁻¹x = ‖z‖²`. Explota SPD, evita el drift numérico de mantener `A⁻¹` incremental y un solo factor sirve para exploit y bonus. `A'` es simétrica por construcción (`A + xxᵀ`); no se re-simetriza. Rank-1 update del factor: F12.

### D3 — `vmap` sobre el eje de brazos (ADR-0003)
`per_arm(A_a, b_a, mu_a, u_a, est_a, mask_a, allowed_a)` se vectoriza con `jax.vmap`; sin bucles Python dentro del JIT. Los brazos inactivos reciben `score = −inf` mediante `where`, nunca se filtran (mantendría la forma).

### D4 — Featurizer v1 fuera del JIT, `d = 32` fijo (ADR-0004, ADR-0013, ADR-0015)
`featurize(RoutingRequest) -> f64[32]` en NumPy puro; enum cerrado de 12 `task_kind`; `InvalidContextError` ante cualquier entrada fuera de dominio; validación de rango `[0,1]` antes de devolver. `FEATURIZER_VERSION = 1` viaja en el journal y en el snapshot. La cuota **no** forma parte de `x_t` (se trata como penalización, D6).

### D5 — Ledger Python con deque por brazo; `QuotaView` como entrada al JIT (ADR-0005)
`QuotaLedger(config, clock)` mantiene `deque[LedgerEvent(id, ts, units)]` por brazo (no por ventana: las ventanas son vistas sobre el mismo deque y la poda usa la más larga). `utilization(now) -> f32[k_max, w_max]` se calcula en Python en cada `select`; dentro del JIT solo hay aritmética. `record` en `select` con `est_units`; `correct` en `report` reemplaza el mismo evento. `dump/load` para el snapshot.

### D6 — Precio sombra por barrera logarítmica dentro del JIT (ADR-0006)
`shadow_a = max_W −μ_a·ln(1 − clip(u, 0, 1−1e-6))`; `score = exploit + bonus − shadow·est_units`; `u ≥ 1` en cualquier ventana enmascara. `quota.log_barrier` existe también en NumPy con test de paridad (±1e-12) para OPE y tests de propiedad. `μ` fijo por brazo en TOML; aprendido: F7.

### D7 — Recompensa escalarizada Q-C-L con calidad externa (ADR-0007)
`r = clip(w_q·q − w_c·min(cost/cost_budget, 1) − w_l·min(lat/latency_budget, 1), 0, 1)` en `reward.py`, sin JAX. Brazos `subscription` reportan `cost_usd = 0` (validado en el daemon ⇒ `-32602`).

### D8 — Política softmax con suelo ε y propensity registrada (ADR-0008)
`π = (1−ε)·softmax(score/τ) + ε/K_act` sobre activos, calculado como `exp((score − max)/τ)` (restar el máximo **antes** de dividir: con τ pequeño `score/τ` desborda a `inf − inf = NaN`); `τ ≥ 1e-6` en config; `arm = categorical(sub_key, log π)`; `rng_key` se avanza con `split` y vive en el estado. Cada `select` persiste su `selection_sequence` contigua y la key posterior; `SnapshotMeta` guarda la secuencia incluida, de modo que el replay restaura exactamente la última transición posterior al snapshot. Garantiza `π_a ≥ ε/K_act` para IPS/DR. Con `K_act = 0` se usa `where(no_arm, 0, arm)` para no propagar NaN; el daemon lee `no_arm_available` antes que `arm`.

### D9 — Snapshot safetensors + JSON, publicado por rename de directorio (ADR-0009)
`snapshot.tmp/` → fsync ficheros y directorio → `snapshot/ → snapshot.old/` → `snapshot.tmp/ → snapshot/` → fsync del directorio padre (la entrada `snapshot/` es durable) → borrar `snapshot.old/`. El directorio de estado se crea como `0700` y WAL/tensores/metadata como `0600`. `config_sha256` = SHA-256 del JSON canónico (`sort_keys`) del `Config` validado completo. Al arrancar: `snapshot.old/` sin `snapshot/` ⇒ recuperar `old`; `snapshot.tmp/` residual ⇒ borrar. `restore` valida `SnapshotMeta` y el conjunto, formas, dtypes e invariantes numéricas de los tensores antes de la conversión JAX; nunca migra (F10). `config_sha256` distinto con formas iguales ⇒ warning.

### D10 — Daemon JSON-RPC 2.0 sobre stdio, single-thread, sin batch (ADR-0010, ADR-0014)
Framing por líneas, sin `Content-Length`. `Router` (dicts in/out, lanza `RpcError`) es la unidad testeable sin subprocess; `serve(stdin, stdout, stderr, router)` solo parsea, despacha y serializa. Códigos −32001…−32006 fijos. Señales ⇒ flag comprobada entre peticiones ⇒ snapshot ⇒ salida 0. Transportes alternativos: F1/F14 (nunca dual-path).

### D11 — Journal JSONL como WAL y dataset OPE (ADR-0011)
Una línea por fase (`select`/`report`/`expire`), claves en orden fijo, `separators=(",", ":")`. `append = write + flush`, **sin** `fsync` por línea (costaría ~ms por decisión); inmediatamente antes de publicar cualquier snapshot, el daemon hace `flush + fsync` del WAL, de modo que un snapshot durable nunca puede adelantar a un `report` no durable. Rotación a `.1.jsonl` (máx. 1) evaluada tras **cualquier** `append` (`select`/`report`/`expire`, incluido el `expire` de un `report` tardío y el barrido del RPC `snapshot`): snapshot primero, luego rename + fsync del directorio del journal; el fichero activo nuevo arranca con los `select` vivos. Rotar es responsabilidad de `snapshot()`, así ningún camino la olvida.

### D12 — API pública mínima (ADR-0012)
`__all__ = [select, update, BanditState, StaticConfig, QuotaLedger, snapshot, restore]`. Todo lo demás es interno; el daemon es el consumidor canónico.

### D13 — `RoutingRequest` semi-estructurada; el llamador clasifica (ADR-0013)
Convoy aporta `task_kind`, conteos, flags y `priority`; el router no inspecciona texto. Mantiene el featurizer < 2 ms y sin dependencias.

### D14 — Ciclo de vida `select` → `report` con TTL y `PendingStore` en memoria (ADR-0016)
`PendingStore: dict[str, Pending]` con orden de inserción (UUIDv7 ⇒ orden temporal); `expire(now, ttl)` se detiene en la primera no expirada. TTL 3600 s por defecto, sin imputación de reward. Se reconstruye por replay del journal, nunca se persiste aparte. `applied_ids` como LRU de 10 000 para distinguir duplicado (`applied:false`) de desconocido (`-32002`).

### D15 — Invariante `state_version` para replay idempotente (ADR-0016)
`state_version` == número de `update` aplicados desde el estado inicial; se escribe en la línea `report` **ya incrementado** y antes de aplicar `update`. Replay reaplica solo `report` con `state_version > snapshot.state_version` (usa `x` y `arm_index` del `select` emparejado leído en el mismo replay); las líneas `select` no se filtran por `state_version` (no lo avanzan): todo `select` retenido sin cierre posterior es un pendiente vivo, y el ledger se reconstruye de forma idempotente por `ledger_event_id`. Cada select también lleva una transición PRNG (`selection_sequence`, `rng_key` posterior); se restaura solo si la secuencia es mayor que la guardada en el snapshot y es contigua. Los `report` reaplicados deben ser contiguos desde `snapshot.state_version + 1` (hueco ⇒ `replay_failed`, salida 3). Sin `snapshot/`, un `decisions.1.jsonl` presente ⇒ `wal_inconsistent`, salida 3 (la rotación siempre sigue a un snapshot); un `decisions.jsonl` sin rotar se reproduce íntegro desde el estado inicial.

### D16 — Warm-up antes de `ready` y contador de compilaciones (ADR-0010, ADR-0018)
`Router.warmup()` ejecuta `select_jit` y `update_jit` con `x = 0` y `QuotaView` de ceros; reporta `warmup_ms`. `linucb._compile_count` se incrementa en el cuerpo de la función pura (que solo se ejecuta en Python durante el trazado); AC-12 exige `== 1` y `health.jit_compilations` lo expone.

### D17 — Configuración TOML por `--config`; cambio = reinicio (ADR-0017)
`tomllib` (stdlib); `Config` frozen; `ConfigError(field, reason)` con ruta de campo. Sin hot-reload ni añadir brazos en caliente (F11). Cambio de `arm_ids` ⇒ snapshot incompatible ⇒ decisión explícita del operador.

### D18 — `jax_enable_x64` en `__init__` y dtypes explícitos (ADR-0018)
`__init__.py` activa x64 antes de cualquier `import jax.numpy` en submódulos; `types.py` fija `F64/F32/I32/U32`; `init_state` construye con esos dtypes; test verifica `A.dtype == float64`. Solo `linucb.py`, `state.py` (y `types.py` para dtypes) importan `jax`; el resto es Python + NumPy y se testea sin compilar.

## Module layout

```
src/azuma_router/
  __init__.py   # x64 + __all__ (D12, D18)         wal.py       # DecisionLog + PendingStore (D11, D14)
  errors.py     # AzumaError y derivadas           snapshot.py  # snapshot / restore (D9)
  types.py      # enums, dataclasses, NamedTuples  ope.py       # IPS / DR en NumPy
  config.py     # load_config, to_static_config    daemon.py    # Router + serve + __main__ (D10)
  featurizer.py # featurize, FEATURIZER_VERSION    state.py     # init_state
  linucb.py     # select / update + *_jit          quota.py     # ventanas, ledger, log_barrier, est_units
  reward.py     # compute_reward
```

Dependencias solo hacia abajo: `daemon → {wal, snapshot, quota, reward, featurizer, linucb, state, config}`; `ope → wal`; `snapshot → {types, quota}`; `linucb, state → types`; `config → {types, errors}`; `types → jax.numpy` (dtypes).

## Sequences

**`select`**: barrido de expirados → `rotate_if_needed()` (los `expire` cuentan aunque la petición se rechace) → parse `request` (`-32602`) → `featurize` (`-32001`) → `allowed` (`-32602`) → `est_units` por brazo → `QuotaView(utilization(now), est, allowed)` → `select_jit` → si `no_arm_available` ⇒ `-32004` sin journal ni avance de estado → `ledger.record` → `uuid7()` → `log.append(select)` + flush → `pending[id] = Pending(...)` → `self.state = state'` → respuesta.

**`report`**: validar (`-32602`) → buscar pendiente (`-32002` / duplicado si en `applied_ids`) → expirado (`-32003`, con `expire` journalizado y rotación evaluada) → `compute_reward` → `ledger.has_event` (ausente ⇒ `ledger_integrity_error`, salida 4 sin journal) → `log.append(report, state_version + 1)` → [`ledger.correct` → `state_version += 1` → `self.state = update_jit(...)` → `pending.pop`, `applied_ids.add`; cualquier excepción aquí ⇒ `JournalApplyError` ⇒ salida 4 sin snapshot, porque la línea ya es durable y solo el replay reconcilia] → `maybe_snapshot()` → `rotate_if_needed()` → respuesta. El ledger retiene eventos `max(ventana más larga, pending_ttl_s)` en todo brazo (también `per_token`, que no tiene ventanas) para que `correct` siempre encuentre su evento sin crecer sin límite.

**`boot`**: `load_config` → `restore` o `init_state` (+ limpieza de `snapshot.tmp/`, recuperación de `snapshot.old/`) → `replay` (reconstruir pendientes; reaplicar reports posteriores al snapshot, que deben ser contiguos en `state_version` o ⇒ `replay_failed`) → `warmup` → `ready` → `serve`. Todo fallo de arranque que requiere al operador (`config_error`, `incompatible_snapshot`, `snapshot_read_failed`, `replay_failed`, `wal_inconsistent`) es un evento JSON en stderr + salida 3; nunca un traceback. `wal_inconsistent` es exactamente «existe `decisions.1.jsonl` y no `snapshot/`»: la rotación siempre sigue a un snapshot, mientras que un journal sin rotar se reproduce íntegro desde el estado inicial (kill antes del primer snapshot).

**`shutdown`**: `ShutdownCoordinator` recibe SIGTERM/SIGINT. Sin petición en curso cierra en el propio handler (un flag puro colgaría en el `readline` bloqueante, PEP 475); con una petición en curso marca `shutdown_deferred` y cierra al terminar `_handle_request`, de modo que el snapshot de cierre nunca captura un `report` a medio aplicar. `Router.close` pone `closing = True` antes de nada: el coordinador ignora cualquier señal con `closing` activo, así un SIGTERM durante un cierre fatal (`with_snapshot=False`) no arranca un segundo cierre con snapshot. `serve` recibe el coordinador como parámetro para poder probarlo sin señales reales.

## Risks / Trade-offs

- [Riesgo] Cholesky falla si `A` deja de ser SPD por acumulación numérica → Mitigación: float64, `ridge_lambda > 0`, test de propiedad AC-11 sobre 1 000 updates; `select` aborta con código 4 si `score` no es finito.
- [Riesgo] `fsync` ausente por línea del journal puede perder la última decisión ante corte de energía → Mitigación: aceptado y documentado (F13); antes de cada snapshot se sincroniza el WAL, por lo que un snapshot durable nunca contiene un `report` ausente del journal.
- [Riesgo] Recompilación inadvertida si algún argumento no estático cambia de forma o dtype → Mitigación: contador de compilaciones expuesto en `health` y AC-12 con 10 000 llamadas variadas.
- [Riesgo] Reloj de pared retrocede (NTP) y rejuvenece eventos del ledger → Mitigación: aceptado; las ventanas se autocorrigen; se registra `clock_skew` en stderr.
- [Riesgo] `PendingStore` crece sin cota si el cliente nunca reporta → Mitigación: TTL 3600 s con barrido en cada `select`/`snapshot` y `expire` en el journal.
- [Riesgo] `ε/K_act` con ε = 0.02 y `K = 32` da propensities mínimas de 6e-4 ⇒ IPS de alta varianza → Mitigación: DR como estimador principal y `ess` reportado; ε configurable hasta 0.5.
- [Trade-off] Cambiar `arm_ids` invalida el snapshot (aprendizaje perdido) → Elegido deliberadamente frente a migraciones (hard-cut); F11 reabre si la rotación de modelos es frecuente.
- [Trade-off] Single-thread stdio limita a un cliente por proceso → Suficiente para Convoy como padre; F1/F9 si cambia.

## Follow-ups (fuera de este change)

| # | Diferido | Origen | Reapertura |
|---|---|---|---|
| F1 | Plugin TS y transportes UDS/HTTP | ADR-0010 | consumidor que no pueda lanzar hijo |
| F2 | LinUCB hybrid | ficha §5 | OPE muestra cold-start lento |
| F3 | Thompson Sampling | ADR-0008 | propensities softmax insuficientes |
| F4 | Embeddings / NLP en el router | ADR-0013/0015 | regret offline insuficiente |
| F5 | GPU/TPU | ficha §5 | `K·d²` ×10 |
| F6 | Multi-tenant | ficha §5 | Convoy lo exige |
| F7 | `μ` aprendido | ADR-0006 | utilización sistemáticamente mala |
| F8 | Reward multi-llamada | ADR-0007 | al abordar F1 |
| F9 | Daemon distribuido | ficha §5 | multi-host |
| F10 | Migración de snapshots | ADR-0009 | nunca salvo decisión explícita |
| F11 | Añadir brazos en caliente | ADR-0017 | rotación frecuente de modelos |
| F12 | Cholesky rank-1 update | ADR-0002 | `d ≥ 256` |
| F13 | `fsync` por línea del journal | D11 | durabilidad ante corte de energía |
| F14 | gRPC/HTTP | ADR-0010 | nunca como dual-path; solo sustitución |
| F15 | Contención explícita de peticiones sólo-select y replay en streaming (SF-4) | SEC-3 | tráfico sólo-select masivo |
| F16 | Modularización de daemon.py (loop de protocolo / router / validación) (CC-4) | CC-4 | refactor de mantenibilidad |

## Migration Plan

No aplica: primer change, sin estado previo que migrar. Un snapshot escrito por esta versión con `schema_version = 1` será rechazado por cualquier versión futura que cambie el schema (hard-cut, F10).
