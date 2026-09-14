# Tasks: azuma-router core (slice 0001) — plan de ejecución TDD

Rama: `feat/0001-linucb-core`. Orden **bottom-up**; cada tarea es un ciclo red → green → refactor y termina en un commit Conventional Commits. Ninguna tarea empieza sin el test en rojo. Criterios de aceptación referencian `specs/core/spec.md` (AC-nn / FR-nn / NFR-nn).

Convenciones:
- `[ ] src` fichero de implementación · `[ ] test` fichero de test · `[ ] AC` criterio de aceptación · `commit` mensaje.
- Fixtures en `fixtures/`; tests espejo en `tests/`.

---

## Bloque 0 — Andamiaje (sin lógica)

**T00** — Esqueleto del paquete y toolchain
- [ ] src: `pyproject.toml` (`requires-python >=3.12`, deps `jax`, `safetensors`; extra `dev`), `uv.lock`, `.pre-commit-config.yaml` (ruff, mypy --strict), `.github/workflows/ci.yml` (matriz 3.12/3.13 × ubuntu/macos), `LICENSE` (Apache-2.0), `src/azuma_router/__init__.py` (x64 + `__all__`), `src/azuma_router/errors.py`.
- [ ] test: `tests/test_package.py` — importar `azuma_router` activa x64 (`jnp.zeros(1).dtype == float64`); `__all__` == lista de ADR-0012; cada excepción de `errors.py` hereda de `AzumaError`.
- [ ] AC: `uv run pytest` verde; `uv run ruff check` y `uv run mypy --strict src` sin errores; CI verde en las 4 celdas.
- commit: `chore: scaffold package, toolchain and ci`

---

## Bloque 1 — Tipos y parser TOML

**T01** — Enums y `RoutingRequest`
- [ ] src: `src/azuma_router/types.py` (`TaskKind` con los 12 valores en orden ADR-0015, `Priority`, `RoutingRequest` frozen con validación en `__post_init__`).
- [ ] test: `tests/test_types.py` — orden del enum es exactamente el de ADR-0015; `RoutingRequest(input_tokens=-1)` ⇒ `InvalidContextError`; campos opcionales aceptan `None`.
- [ ] AC: AC-04 (parte de validación estructural).
- commit: `feat(types): task kinds, priority and routing request`

**T02** — `StaticConfig`, `BanditState`, `QuotaView`, `Diagnostics`, `Pending`, `SnapshotMeta`
- [ ] src: `src/azuma_router/types.py` (añadir).
- [ ] test: `tests/test_types.py` — `StaticConfig` es hashable y `frozen`; `BanditState` es PyTree (`jax.tree_util.tree_leaves` devuelve 5 hojas); `SnapshotMeta` round-trip `asdict`/`from_dict`.
- [ ] AC: contratos §4.2 del spec.
- commit: `feat(types): bandit state, static config and snapshot metadata`

**T03** — Parser `azuma.toml`
- [ ] src: `src/azuma_router/config.py` (`load_config`, `Config`, `ArmConfig`, `WindowConfig`, `to_static_config`).
- [ ] test: `tests/test_config.py` + fixtures `fixtures/config/azuma.valid.toml`, `fixtures/config/invalid/{dup_arm_id,missing_bandit,sub_without_windows,per_token_with_windows,per_token_with_mu,bad_type,too_many_arms,too_many_windows,bad_period,bad_anchor}.toml`.
- [ ] AC: AC-01, AC-02; `to_static_config` produce `barrier_mu` de longitud `k_max` con ceros en padding y `is_subscription` coherente.
- commit: `feat(config): strict toml loader with field-path errors`

---

## Bloque 2 — Featurizer

**T04** — Vector de referencia y bloques
- [ ] src: `src/azuma_router/featurizer.py` (`FEATURIZER_VERSION = 1`, `featurize`, `D = 32`).
- [ ] test: `tests/test_featurizer.py` + `fixtures/featurizer/v1_reference.json` (petición de ejemplo del spec §5.2 y su vector esperado, calculado a mano en el fixture con comentario por índice).
- [ ] AC: AC-03 (igualdad exacta con tolerancia 1e-12), índices 24–31 son 0, índice 23 == x16·x1.
- commit: `feat(featurizer): v1 block layout with reference vector`

**T05** — Fail-fast y propiedades
- [ ] src: `featurizer.py` (validación de salida: finito, en [0,1], longitud 32).
- [ ] test: `tests/test_featurizer.py` — Hypothesis: estrategia de `RoutingRequest` válidos ⇒ rangos y norma; `task_kind` inválido (string no enum) ⇒ `InvalidContextError` con `field`.
- [ ] AC: AC-04, AC-05, FR-06.
- commit: `feat(featurizer): fail-fast validation and property tests`

---

## Bloque 3 — Matemáticas JAX

**T06** — `init_state`
- [ ] src: `src/azuma_router/state.py`.
- [ ] test: `tests/test_state.py` — dtypes (float64/int32/bool/uint32), formas `(k_max,d,d)` etc., `A[a] == ridge·I`, máscara padding.
- [ ] AC: AC-06.
- commit: `feat(state): initial bandit state with ridge prior`

**T07** — `select` puro: exploit y bonus
- [ ] src: `src/azuma_router/linucb.py` (`_score_arms`: `exploit`, `bonus`, `score` con `shadow = 0` y máscara; el muestreo y la barrera llegan en T08 — no se introduce ningún argmax provisional).
- [ ] test: `tests/test_linucb_select.py` — AC-07; bonus decrece al añadir observaciones alineadas con `x` (monotonía, Hypothesis); brazos con `mask=False` tienen `score = -inf`.
- [ ] AC: AC-07, parte de FR-08.
- commit: `feat(linucb): vectorized ucb scores via cholesky solve`

**T08** — Precio sombra y muestreo softmax
- [ ] src: `linucb.py` (barrera logarítmica dentro del JIT, `allowed`, `active`, softmax + suelo, `categorical`, `rng_key` split, `no_arm_available`).
- [ ] test: `tests/test_linucb_select.py` — AC-08, AC-09, AC-10; `policy` suma 1 sobre activos (±1e-12); `policy[a] ≥ eps/k_act` para activos; con `K=1`, `propensity == 1.0`; frecuencia empírica de `arm` sobre 20 000 muestras ≈ `policy` (±0.02).
- [ ] AC: AC-08, AC-09, AC-10.
- commit: `feat(linucb): shadow price penalty and softmax sampling policy`

**T09** — `update`
- [ ] src: `linucb.py` (`update`).
- [ ] test: `tests/test_linucb_update.py` — AC-11 (Hypothesis, 1 000 updates), otros brazos bitwise iguales, `n_pulls`.
- [ ] AC: AC-11, FR-11, FR-12.
- commit: `feat(linucb): rank-1 ridge update of pulled arm`

**T10** — JIT, invariante de forma y benchmark
- [ ] src: `linucb.py` (`select_jit`, `update_jit`, contador de trazado).
- [ ] test: `tests/test_jit_invariants.py` (AC-12); `tests/bench/test_bench_linucb.py` (NFR-01, marcado `bench`, excluido del `pytest` por defecto vía `-m "not bench"`).
- [ ] AC: AC-12, NFR-01 (p99 impreso en CI; no falla el job).
- commit: `feat(linucb): jit boundaries with single-compilation guarantee`

**T11** — Simulación de regret (test de sistema del bandido)
- [ ] src: — (solo test + fixture `fixtures/sim/arms_synthetic.json` con θ* por brazo).
- [ ] test: `tests/test_regret_simulation.py` — 20 000 rondas con contextos aleatorios; regret acumulado con exponente de ajuste < 0.6; cada brazo ≥ 1 % de pulls.
- [ ] AC: §10 de la ficha (cross-check de regret sublineal).
- commit: `test(linucb): sublinear regret simulation`

---

## Bloque 4 — Cuota y recompensa

**T12** — Ventanas y ledger
- [ ] src: `src/azuma_router/quota.py` (`RollingWindow`, `FixedResetWindow`, `QuotaLedger`, `LedgerEvent`, `dump/load`).
- [ ] test: `tests/test_quota_ledger.py` — reloj inyectado; AC-13; poda de eventos antiguos; `FixedResetWindow` semanal con anchor; `dump/load` round-trip.
- [ ] AC: AC-13, FR-14.
- commit: `feat(quota): rolling and fixed-reset windows with exact ledger`

**T13** — Utilización, máscara y corrección
- [ ] src: `quota.py` (`utilization(now)`, `record`, `correct`, `est_units`).
- [ ] test: `tests/test_quota_ledger.py` — AC-14, AC-15; `est_units` con y sin `expected_output_tokens`; brazo `per_token` ⇒ fila de ceros.
- [ ] AC: AC-14, AC-15, FR-16, FR-17.
- commit: `feat(quota): utilization view, estimated units and correction`

**T14** — Barrera y paridad NumPy/JAX
- [ ] src: `quota.py` (`log_barrier` NumPy).
- [ ] test: `tests/test_quota_barrier.py` — convexidad y monotonía (Hypothesis); igualdad ±1e-12 entre `quota.log_barrier` y el `shadow_price` de `Diagnostics` para una malla de `u`.
- [ ] AC: FR-08 (shadow), ADR-0006.
- commit: `feat(quota): log barrier with numpy/jax parity test`

**T15** — Recompensa
- [ ] src: `src/azuma_router/reward.py`.
- [ ] test: `tests/test_reward.py` — AC-16; clip en 0 y 1; Hypothesis: salida en [0,1].
- [ ] AC: AC-16, FR-18.
- commit: `feat(reward): quality-cost-latency scalarization`

---

## Bloque 5 — WAL y snapshots

**T16** — `DecisionLog` append/flush/rotate
- [ ] src: `src/azuma_router/wal.py` (`DecisionLog`).
- [ ] test: `tests/test_wal.py` — AC-17 (spy en `flush`); rotación al superar `max_bytes` deja `.1.jsonl`; orden de claves del spec §4.4.
- [ ] AC: AC-17, FR-20.
- commit: `feat(wal): append-only decision log with rotation`

**T17** — `PendingStore` y expiración
- [ ] src: `wal.py` (`PendingStore`, `expire`).
- [ ] test: `tests/test_wal.py` — expiración por orden de inserción; parada en el primer no expirado; TTL con reloj inyectado.
- [ ] AC: FR-22 (parte de store).
- commit: `feat(wal): pending store with ttl sweep`

**T18** — Replay
- [ ] src: `wal.py` (`replay`).
- [ ] test: `tests/test_wal_replay.py` + `fixtures/wal/{pending_only,reported,expired,corrupt_tail,corrupt_middle,rotated}.jsonl`.
- [ ] AC: AC-20; `corrupt_tail` descarta 1 línea; `corrupt_middle` lanza; `rotated` lee ambos ficheros en orden.
- commit: `feat(wal): replay pending decisions from log`

**T19** — Snapshot atómico
- [ ] src: `src/azuma_router/snapshot.py` (`snapshot`).
- [ ] test: `tests/test_snapshot.py` — AC-22; simular crash tras primer rename (`snapshot.old/` presente) y verificar recuperación; nombres de tensores exactos.
- [ ] AC: AC-22, FR-25.
- commit: `feat(snapshot): atomic safetensors + json snapshot`

**T20** — Restore y compatibilidad
- [ ] src: `snapshot.py` (`restore`, validación de `SnapshotMeta`).
- [ ] test: `tests/test_snapshot.py` — AC-23 (un caso por campo de FR-26), AC-24, `config_sha256` distinto con formas iguales ⇒ warning y acepta.
- [ ] AC: AC-23, AC-24, FR-26, FR-27.
- commit: `feat(snapshot): fail-fast restore with compatibility checks`

---

## Bloque 6 — Daemon JSON-RPC

**T21** — `Router.select` (sin stdio)
- [ ] src: `src/azuma_router/daemon.py` (`Router`, `select`).
- [ ] test: `tests/test_router.py` — secuencia §4 del design con reloj y log inyectados; `-32602`, `-32001`, `-32004`; línea `select` escrita antes de retornar; `state` avanzado.
- [ ] AC: FR-10, FR-20, FR-29 (parte).
- commit: `feat(daemon): router select orchestration`

**T22** — `Router.report`, expiración y snapshot automático
- [ ] src: `daemon.py` (`report`, `sweep`, `maybe_snapshot`, `snapshot`, `health`).
- [ ] test: `tests/test_router.py` — AC-18, AC-19, AC-21; snapshot tras 100 updates y tras 30 s; `health` refleja `pending` y `state_version`; `cost_usd > 0` en subscription ⇒ `-32602`.
- [ ] AC: AC-18, AC-19, AC-21, FR-21–FR-24, FR-28.
- commit: `feat(daemon): report, ttl sweep and automatic snapshots`

**T23** — Boot: restore + replay + warm-up + `ready`
- [ ] src: `daemon.py` (`Router.boot`, `warmup`).
- [ ] test: `tests/test_router_boot.py` — arranque sin snapshot; con snapshot y WAL pendiente (reconstruye); snapshot incompatible ⇒ `IncompatibleSnapshotError`; WAL con `state_version > 0` sin snapshot ⇒ error; `warmup_ms > 0`.
- [ ] AC: FR-23, FR-30, AC-20 (integración).
- commit: `feat(daemon): boot sequence with replay and jit warm-up`

**T24** — Loop stdio JSON-RPC
- [ ] src: `daemon.py` (`serve`, parse/dispatch, `RpcError` → JSON, `__main__`, señales).
- [ ] test: `tests/test_jsonrpc_loop.py` — con `io.StringIO`: parse error, invalid request, batch ⇒ `-32600` (AC-27), método desconocido, notificación sin `id` no responde, `-32005` antes de ready (AC-26), línea vacía ignorada, línea > 1 MiB.
- [ ] AC: AC-26, AC-27, FR-29, FR-31–FR-33.
- commit: `feat(daemon): stdio json-rpc 2.0 loop`

**T25** — End-to-end por subprocess
- [ ] src: —
- [ ] test: `tests/test_daemon_e2e.py` — lanza `python -m azuma_router.daemon --config fixtures/config/azuma.valid.toml` con `snapshot_dir` temporal; espera `ready`; `health`, `select`, `report`, `health` (`state_version == 1`); SIGTERM ⇒ snapshot en disco y exit 0; relanzar ⇒ `restored_from_snapshot == true`; NFR-02 (`warmup_ms < 5000`).
- [ ] AC: AC-25, NFR-02.
- commit: `test(daemon): end-to-end subprocess lifecycle`

---

## Bloque 7 — OPE y cierre

**T26** — Evaluador IPS / DR
- [ ] src: `src/azuma_router/ope.py`.
- [ ] test: `tests/test_ope.py` + `fixtures/ope/decisions.jsonl` (generado por la simulación de T11 con la política registrada).
- [ ] AC: AC-28; `ess` (effective sample size) reportado; política candidata con `alpha` distinto cambia `ips` y `dr` de forma consistente.
- commit: `feat(ope): ips and doubly-robust off-policy evaluation`

**T27** — Documentación y verificación final
- [ ] src: `README.md` (tagline, quickstart con `azuma.toml` de ejemplo, protocolo), `CLAUDE.md` del proyecto (comandos, layout, convenciones), actualizar `specs/core/spec.md` estado → APROBADA/IMPLEMENTADA.
- [ ] test: — (ejecutar la sección Verification de la ficha §10 y pegar salida en el PR).
- [ ] AC: todos los AC-01…AC-28 verdes; benchmark p99 registrado en el PR; `security-audit` ejecutado (input handling del daemon).
- commit: `docs: readme, project claude.md and verification results`

---

## Orden de dependencias

```
T00 → T01 → T02 → T03 → T04 → T05 → T06 → T07 → T08 → T09 → T10 → T11
                                                        └→ T12 → T13 → T14 → T15
                                                                              └→ T16 → T17 → T18 → T19 → T20
                                                                                                        └→ T21 → T22 → T23 → T24 → T25 → T26 → T27
```

T12–T15 solo dependen de T03; pueden ejecutarse en paralelo con T06–T11 si se trabaja en dos worktrees, pero la integración (T21) requiere ambos.
