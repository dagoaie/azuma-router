# Design: azuma-router core (slice 0001)

Contratos en `specs/core/spec.md`; decisiones en `docs/adr/`. Este documento describe **cómo** se reparten las responsabilidades entre módulos y dónde está la frontera JIT. No contiene código.

---

## 1. Módulos y dependencias

```
src/azuma_router/
  __init__.py      # jax_enable_x64 + re-export de la API pública (ADR-0012/0018)
  errors.py        # ConfigError, InvalidContextError, IncompatibleSnapshotError, RpcError(code, data)
  types.py         # TaskKind, Priority, RoutingRequest, BanditState, StaticConfig, QuotaView, Diagnostics, Pending, SnapshotMeta
  config.py        # load_config(path) -> Config; Config -> StaticConfig
  featurizer.py    # featurize(RoutingRequest) -> f64[32]; FEATURIZER_VERSION = 1
  state.py         # init_state(config) -> BanditState; shape asserts
  linucb.py        # select / update (puras) + select_jit / update_jit
  quota.py         # RollingWindow, FixedResetWindow, QuotaLedger, log_barrier, est_units
  reward.py        # compute_reward
  wal.py           # DecisionLog (append/flush/rotate/replay) + PendingStore
  snapshot.py      # snapshot / restore (safetensors + json, rename atómico)
  ope.py           # evaluate(log_path, StaticConfig) -> {ips, dr, n, ess}
  daemon.py        # loop stdio JSON-RPC; Router (orquesta los anteriores); __main__
```

Grafo de dependencias (solo hacia abajo; ningún ciclo):

```
daemon ─► wal, snapshot, quota, reward, featurizer, linucb, state, config
ope    ─► wal (lectura), linucb (policy), featurizer (no; usa x registrado)
snapshot ─► types, quota (LedgerDump)
wal    ─► types
linucb ─► types            quota ─► types        featurizer ─► types
state  ─► types, config    reward ─► types       config ─► types, errors
types  ─► (jax.numpy solo para dtypes)           errors ─► —
```

Regla: **solo `linucb.py` y `state.py` importan `jax`** (más `types.py` para dtypes). `quota`, `wal`, `reward`, `featurizer`, `config`, `daemon` son Python + NumPy puro. Así los tests de todo lo que no es matemática JAX corren sin compilar nada.

---

## 2. Frontera JIT

```
                 Python (I/O, reloj, validación)          │        XLA (puro, formas fijas)
                                                           │
 RoutingRequest ─featurize─► x: f64[32] ───────────────────┼─►┐
 QuotaLedger.utilization(now) ─► QuotaView ────────────────┼─►│ select_jit(state, x, quota_view, static)
 BanditState (del Router) ─────────────────────────────────┼─►│   → (state', arm, propensity, diagnostics)
                                                           │  └──────────────────────────────────────────►
 report → compute_reward → r ──────────────────────────────┼─► update_jit(state, x, arm, r, static) → state'
```

### 2.1 `select` (dentro del JIT)

```
per_arm(A_a, b_a, mu_a, u_a[W], est_a, mask_a, allowed_a):        # vmap sobre eje 0
  L      = cho_factor(A_a)                                          # ADR-0002; float64
  theta  = cho_solve(L, b_a)
  z      = solve_triangular(L, x, lower=True)                       # x^T A^{-1} x = ||z||²
  exploit= x @ theta
  bonus  = alpha * sqrt(z @ z)
  u_clip = clip(u_a, 0, 1 - 1e-6)
  shadow = max_W( -mu_a * log1p(-u_clip) )                          # ADR-0006
  active = mask_a & allowed_a & all_W(u_a < 1)
  score  = where(active, exploit + bonus - shadow * est_a, -inf)
  return exploit, bonus, shadow, score, active

scores, active = vmap(per_arm)(...)
k_act    = sum(active)
softmax  = exp(score/tau - logsumexp(score/tau))                    # -inf → 0
policy   = where(active, (1-eps)*softmax + eps/k_act, 0)            # ADR-0008
key, sub = split(state.rng_key)
arm      = categorical(sub, log(policy))                            # sobre k_max; policy=0 nunca sale
no_arm   = k_act == 0
return state._replace(rng_key=key), arm, policy[arm], Diagnostics(...)
```

Notas:
- `k_act == 0` ⇒ `policy` es todo ceros/NaN; el daemon consulta `no_arm_available` **antes** de leer `arm`. Dentro del JIT se usa `where(no_arm, 0, arm)` para no propagar NaN.
- `est_a` viene en `QuotaView.est_units` (calculado en Python por `quota.est_units(config.arms[a], request)`); no hay lógica de tokens dentro del JIT.
- `static: StaticConfig` entra por `static_argnums=(3,)`; `barrier_mu` e `is_subscription` son tuplas ⇒ hashables. Cambiar cualquier hiperparámetro recompila **una vez** (aceptable: solo ocurre en arranque).

### 2.2 `update` (dentro del JIT)

```
A' = A.at[arm].add(outer(x, x))
b' = b.at[arm].add(r * x)
n' = n_pulls.at[arm].add(1)
return state._replace(A=A', b=b', n_pulls=n')
```

`A'` es simétrica por construcción (suma de simétricas); no se re-simetriza.

### 2.3 Compilación y warm-up

- `select_jit = jax.jit(select, static_argnums=(3,))`, `update_jit = jax.jit(update, static_argnums=(4,))` definidos a nivel de módulo en `linucb.py`.
- `daemon.Router.warmup()` ejecuta una llamada real a cada uno con `x = zeros`, `QuotaView` de ceros, y descarta el resultado. Reporta ms en `ready`.
- Test de invariante de forma: envuelve `select`/`update` con un contador vía `jax.jit(..., ).lower().compile()` no; se usa `jax.monitoring`-free approach: `linucb._compile_count` incrementado en un `@functools.wraps` interno que solo se ejecuta durante el trazado (el cuerpo de la función pura corre en Python solo al trazar). AC-12 exige `== 1`.

### 2.4 `jax_enable_x64`

`__init__.py` hace `jax.config.update("jax_enable_x64", True)` **antes** de cualquier `import jax.numpy` en submódulos (import order: `__init__` → `types` → resto). `types.py` fija `F64 = jnp.float64`, `F32 = jnp.float32`, `I32 = jnp.int32`, `U32 = jnp.uint32` y `state.init_state` construye con esos dtypes explícitos; un test verifica `init_state(...).A.dtype == float64`.

---

## 3. Fuera del JIT

### 3.1 `quota.py`
- `RollingWindow(duration_s, capacity)` / `FixedResetWindow(period, reset_anchor, capacity)`: dataclasses; método `window_start(now) -> float`.
- `QuotaLedger(config, clock: Callable[[], float])`: `events: dict[arm_index, deque[LedgerEvent(id, ts, units)]]`, un deque por brazo (no por ventana: las ventanas son vistas sobre el mismo deque; la poda usa la ventana más larga). `record(arm, units, ts) -> event_id`; `correct(event_id, units)`; `utilization(now) -> f32[k_max, w_max]`; `dump()/load()` para snapshot.
- `log_barrier(u, mu)` existe también como función NumPy pura para tests de propiedad y para OPE (misma fórmula que dentro del JIT; test de igualdad entre ambas).
- `est_units(arm_cfg, request) -> float`.

### 3.2 `wal.py`
- `DecisionLog(path, max_bytes)`: `append(dict)` = `write` + `flush` + `os.fsync` **no** (fsync por línea costaría ~ms; se acepta perder la última línea en corte de energía, no en crash de proceso — el rename del snapshot sí hace fsync). `rotate()` si `size > max_bytes` tras un `report`/`expire`. `replay(min_state_version) -> Iterator[dict]` sobre `.1.jsonl` + activo, descartando solo una última línea corrupta.
- `PendingStore`: `dict[str, Pending]` con orden de inserción; `expire(now, ttl) -> list[Pending]` recorre desde el inicio y para en el primer no expirado (UUIDv7 + inserción ⇒ orden temporal).

### 3.3 `snapshot.py`
- `snapshot(dir, state, ledger, meta)`: `safetensors.numpy.save_file` de `{A, b, n_pulls, mask, rng_key}` convertidos a NumPy; `state.json` con `dataclasses.asdict(meta)`; fsync de ficheros y del directorio padre; `os.replace(tmp, final)` — para reemplazar un directorio existente: rename `snapshot/` → `snapshot.old/`, rename `snapshot.tmp/` → `snapshot/`, borrar `snapshot.old/`. Al arrancar, si existe `snapshot.old/` y no `snapshot/`, se restaura `old` (crash entre los dos renames).
- `restore(dir, config)`: valida `SnapshotMeta` contra `config` (lista de campos de FR-26) antes de tocar arrays.

### 3.4 `daemon.py`
- `Router`: posee `state`, `ledger`, `pending`, `log`, `state_version`, `updates_since_snapshot`, `last_snapshot_ts`. Métodos `select(params) -> dict`, `report(params) -> dict`, `snapshot() -> dict`, `health() -> dict`. Sin I/O de stdio: recibe/devuelve dicts, lanza `RpcError`. Es la unidad testeable sin subprocess.
- `serve(stdin, stdout, stderr, router)`: loop `for line in stdin` → parse → dispatch → `stdout.write(json + "\n")`, `flush`. Captura `RpcError` → error JSON-RPC; cualquier otra excepción → log en stderr, snapshot si el estado es sano, salida código 4.
- `__main__`: argparse `--config`; `Router.boot()` = load_config → restore o init → replay WAL → warmup → notificación `ready` → `serve`.
- Señales: `SIGTERM`/`SIGINT` ⇒ flag; el loop lo comprueba entre peticiones ⇒ snapshot ⇒ salida 0.

### 3.5 `ope.py`
Lee pares (`select`, `report`) del JSONL; para cada uno reconstruye `score` a partir de `exploit`, `bonus`, `shadow_price` registrados (re-escalando `bonus` por `alpha_candidate / alpha_logged`) y calcula la `policy` candidata con la misma fórmula softmax (NumPy). IPS y DR con $\hat r$ = `exploit` registrado. Sin JAX.

---

## 4. Secuencia `select` → `report` (daemon)

```
select(params):
  1. sweep: pending.expire(now, ttl) → log.append(expire…) por cada uno
  2. request = parse(params.request)  → -32602 si inválido
  3. x = featurize(request)            → -32001 si InvalidContextError
  4. allowed = mask from params.allowed_arms → -32602 si arm_id desconocido
  5. est = [est_units(arm, request) for arm]
  6. view = QuotaView(ledger.utilization(now), est, allowed)
  7. state', arm, p, diag = select_jit(state, x, view, static)
  8. if diag.no_arm_available → -32004 (no se escribe WAL, no se avanza state)
  9. event_id = ledger.record(arm, est[arm], now)
 10. decision_id = uuid7()
 11. log.append(phase=select…)          # flush antes de responder
 12. pending[decision_id] = Pending(…)
 13. self.state = state'
 14. return result

report(params):
  1. validate → -32602
  2. p = pending.get(id) → -32002 si no está y no está en applied_ids (LRU 10k) → duplicado si applied
  3. if expired (según ts_select) → -32003          # ya se habría barrido; defensa
  4. r = compute_reward(...)
  5. ledger.correct(p.ledger_event_id, actual_units)
  6. state_version += 1
  7. log.append(phase=report…, state_version)
  8. self.state = update_jit(state, p.x, p.arm_index, r, static)
  9. pending.pop; applied_ids.add
 10. maybe_snapshot()
 11. return {applied: true, reward, state_version}
```

Invariante: `state_version` == número de `update` aplicados desde el estado inicial. El replay en arranque reaplica `report` con `state_version > snapshot.state_version` (necesita `x` y `arm_index` de la línea `select` correspondiente, que se lee del mismo replay).

---

## 5. Estrategia de tests (resumen; detalle en `tasks.md`)

| Capa | Herramienta | Qué verifica |
|---|---|---|
| Unitarios puros | pytest | contratos de cada función |
| Propiedades | hypothesis | rangos del featurizer, SPD, monotonía del bonus, idempotencia del ledger |
| Invariantes JIT | contador de trazado | exactamente 1 compilación |
| Benchmark | pytest-benchmark | NFR-01 |
| Integración | subprocess + pipes | daemon end-to-end (AC-25) |
| Fixtures | `fixtures/` | TOML válidos/inválidos, vector de referencia del featurizer, JSONL de replay y OPE |

---

## 6. Decisiones diferidas consolidadas

| # | Diferido | ADR / origen | Condición de reapertura |
|---|---|---|---|
| F1 | Plugin TS y transportes UDS/HTTP | ADR-0010 | consumidor que no pueda lanzar hijo |
| F2 | LinUCB hybrid | ficha §5 | OPE muestra cold-start lento |
| F3 | Thompson Sampling | ADR-0008 | propensities softmax insuficientes |
| F4 | Embeddings / clasificación NLP en router | ADR-0013/0015 | regret offline insuficiente |
| F5 | GPU/TPU | ficha §5 | K·d² ×10 |
| F6 | Multi-tenant | ficha §5 | Convoy lo exige |
| F7 | μ aprendido | ADR-0006 | utilización sistemáticamente mala |
| F8 | Reward multi-llamada | ADR-0007 | al abordar F1 |
| F9 | Daemon distribuido | ficha §5 | multi-host |
| F10 | Migración de snapshots | ADR-0009 | nunca salvo decisión explícita |
| F11 | Añadir brazos en caliente (`azuma migrate-arms`) | ADR-0017 | rotación frecuente de modelos |
| F12 | Cholesky rank-1 update | ADR-0002 | d ≥ 256 |
| F13 | `fsync` por línea del WAL | design §3.2 | requisito de durabilidad ante corte de energía |
| F14 | gRPC/HTTP dual-path | ADR-0010 | **nunca como dual-path**; solo como sustitución |
