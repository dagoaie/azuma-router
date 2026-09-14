## 0. Andamiaje (bootstrap, ya realizado fuera del pipeline — ver `plan.md`)

- [x] 0.1 T00 Esqueleto del paquete y toolchain: `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `LICENSE`, `src/azuma_router/{__init__,errors}.py`, `tests/test_package.py`. Verificación: `uv run pytest` verde, `uv run ruff check` y `uv run mypy` sin errores (commit `738d62c`).

## 1. Tipos y configuración

- [x] 1.1 T01 `types.py`: `TaskKind` (12 valores en el orden de ADR-0015), `Priority`, `RoutingRequest` frozen con validación en `__post_init__`. Test `tests/test_types.py`: orden exacto del enum; `input_tokens=-1` ⇒ `InvalidContextError`; opcionales aceptan `None`. Verificación: AC-04 (parte estructural). Commit `feat(types): task kinds, priority and routing request`.
- [x] 1.2 T02 `types.py`: `StaticConfig`, `BanditState`, `QuotaView`, `Diagnostics`, `Pending`, `SnapshotMeta`, dtypes `F64/F32/I32/U32`. Test: `StaticConfig` hashable y frozen; `BanditState` es PyTree de 5 hojas; `SnapshotMeta` round-trip `asdict`/`from_dict`. Verificación: contratos §4.2 de `docs/contracts/linucb-core.md`. Commit `feat(types): bandit state, static config and snapshot metadata`.
- [x] 1.3 T03 `config.py`: `load_config`, `Config`, `ArmConfig`, `WindowConfig`, `to_static_config`. Fixtures `fixtures/config/azuma.valid.toml` e `invalid/{dup_arm_id,missing_bandit,sub_without_windows,per_token_with_windows,per_token_with_mu,bad_type,too_many_arms,too_many_windows,bad_period,bad_anchor}.toml`. Test `tests/test_config.py`. Verificación: AC-01, AC-02; padding de `barrier_mu`/`is_subscription`. Commit `feat(config): strict toml loader with field-path errors`.

## 2. Featurizer

- [x] 2.1 T04 `featurizer.py`: `FEATURIZER_VERSION = 1`, `D = 32`, `featurize` con el layout v1. Fixture `fixtures/featurizer/v1_reference.json` (vector calculado a mano, comentario por índice). Test `tests/test_featurizer.py`. Verificación: AC-03 (±1e-12), `x[24:32] == 0`, `x[23] == x[16]*x[1]`. Commit `feat(featurizer): v1 block layout with reference vector`.
- [x] 2.2 T05 `featurizer.py`: validación de salida (finito, `[0,1]`, longitud 32) y rechazo de entradas. Test: Hypothesis sobre `RoutingRequest` válidas; `task_kind` inválido ⇒ `InvalidContextError` con `field`. Verificación: AC-04, AC-05, FR-06. Commit `feat(featurizer): fail-fast validation and property tests`.

## 3. Matemáticas JAX

- [x] 3.1 T06 `state.py`: `init_state(config)`. Test `tests/test_state.py`: dtypes, formas, `A[a] == ridge·I`, máscara de padding. Verificación: AC-06. Commit `feat(state): initial bandit state with ridge prior`.
- [x] 3.2 T07 `linucb.py`: `_score_arms` con `cho_factor`/`cho_solve`/`solve_triangular`; `exploit`, `bonus`, `score` con `shadow = 0` y máscara `−inf` (sin argmax provisional). Test `tests/test_linucb_select.py`. Verificación: AC-07; monotonía del bonus (Hypothesis); `mask=False` ⇒ `−inf`. Commit `feat(linucb): vectorized ucb scores via cholesky solve`.
- [x] 3.3 T08 `linucb.py`: barrera logarítmica en JIT, `allowed`, `active`, softmax + suelo, `categorical`, split de `rng_key`, `no_arm_available`. Test: `policy` suma 1 (±1e-12); `policy[a] ≥ ε/K_act`; `K=1` ⇒ `propensity == 1.0`; frecuencia empírica sobre 20 000 muestras ±0.02. Verificación: AC-08, AC-09, AC-10. Commit `feat(linucb): shadow price penalty and softmax sampling policy`.
- [x] 3.4 T09 `linucb.py`: `update`. Test `tests/test_linucb_update.py`: Hypothesis 1 000 updates, simetría bitwise, `cho_factor` ok, `‖Aθ̂−b‖∞ < 1e-9`, otros brazos bitwise iguales. Verificación: AC-11, FR-11, FR-12. Commit `feat(linucb): rank-1 ridge update of pulled arm`.
- [x] 3.5 T10 `linucb.py`: `select_jit`, `update_jit`, contador de compilaciones. Tests `tests/test_jit_invariants.py` y `tests/bench/test_bench_linucb.py` (marcado `bench`). Verificación: AC-12; NFR-01 impreso en CI (informativo). Commit `feat(linucb): jit boundaries with single-compilation guarantee`.
- [x] 3.6 T11 Simulación de regret: fixture `fixtures/sim/arms_synthetic.json`, test `tests/test_regret_simulation.py` (20 000 rondas). Verificación: exponente log-log `< 0.6`; cada brazo ≥ 1 % de pulls. Commit `test(linucb): sublinear regret simulation`.

## 4. Cuota y recompensa

- [x] 4.1 T12 `quota.py`: `RollingWindow`, `FixedResetWindow`, `LedgerEvent`, `QuotaLedger` con reloj inyectado, `dump/load`. Test `tests/test_quota_ledger.py`. Verificación: AC-13; poda; reset semanal con anchor; round-trip `dump/load`. Commit `feat(quota): rolling and fixed-reset windows with exact ledger`.
- [x] 4.2 T13 `quota.py`: `utilization(now)`, `record`, `correct`, `est_units`. Test: AC-14, AC-15; `est_units` con y sin `expected_output_tokens`; `per_token` ⇒ fila de ceros. Verificación: AC-14, AC-15, FR-16, FR-17. Commit `feat(quota): utilization view, estimated units and correction`.
- [x] 4.3 T14 `quota.py`: `log_barrier` NumPy. Test `tests/test_quota_barrier.py`: convexidad y monotonía (Hypothesis); paridad ±1e-12 con `Diagnostics.shadow_price`. Verificación: FR-08 (shadow), ADR-0006. Commit `feat(quota): log barrier with numpy/jax parity test`.
- [x] 4.4 T15 `reward.py`: `compute_reward`. Test `tests/test_reward.py`: AC-16; clip en 0 y 1; Hypothesis rango `[0,1]`. Verificación: AC-16, FR-18. Commit `feat(reward): quality-cost-latency scalarization`.

## 5. Journal y snapshots

- [x] 5.1 T16 `wal.py`: `DecisionLog` append/flush/rotate. Test `tests/test_wal.py`: spy en `flush`; rotación a `.1.jsonl`; orden de claves de §4.4. Verificación: AC-17, FR-20. Commit `feat(wal): append-only decision log with rotation`.
- [x] 5.2 T17 `wal.py`: `PendingStore` con `expire`. Test: orden de inserción; parada en el primer no expirado; TTL con reloj inyectado. Verificación: FR-22 (store). Commit `feat(wal): pending store with ttl sweep`.
- [x] 5.3 T18 `wal.py`: `replay`. Fixtures `fixtures/wal/{pending_only,reported,expired,corrupt_tail,corrupt_middle,rotated}.jsonl`; test `tests/test_wal_replay.py`. Verificación: AC-20; `corrupt_tail` descarta 1 línea; `corrupt_middle` lanza; `rotated` lee ambos en orden. Commit `feat(wal): replay pending decisions from log`.
- [x] 5.4 T19 `snapshot.py`: `snapshot` atómico. Test `tests/test_snapshot.py`: AC-22; crash simulado tras el primer rename (`snapshot.old/`) y recuperación; nombres de tensores exactos. Verificación: AC-22, FR-25. Commit `feat(snapshot): atomic safetensors + json snapshot`.
- [x] 5.5 T20 `snapshot.py`: `restore` con validación de `SnapshotMeta`. Test: un caso por campo de FR-26; round-trip; `config_sha256` distinto ⇒ warning y acepta. Verificación: AC-23, AC-24, FR-26, FR-27. Commit `feat(snapshot): fail-fast restore with compatibility checks`.

## 6. Daemon JSON-RPC

- [x] 6.1 T21 `daemon.py`: `Router.select` (sin stdio). Test `tests/test_router.py` con reloj y journal inyectados: secuencia completa; `-32602`, `-32001`, `-32004`; línea `select` escrita antes de retornar; estado avanzado. Verificación: FR-10, FR-20, FR-29 (parte). Commit `feat(daemon): router select orchestration`.
- [x] 6.2 T22 `daemon.py`: `Router.report`, `sweep`, `maybe_snapshot`, `snapshot`, `health`. Test: AC-18, AC-19, AC-21; snapshot tras 100 updates y tras 30 s; `health` refleja `pending` y `state_version`; `cost_usd > 0` en subscription ⇒ `-32602`. Verificación: AC-18, AC-19, AC-21, FR-21–FR-24, FR-28. Commit `feat(daemon): report, ttl sweep and automatic snapshots`.
- [x] 6.3 T23 `daemon.py`: `Router.boot`, `warmup`. Test `tests/test_router_boot.py`: sin snapshot; con snapshot y journal pendiente; snapshot incompatible ⇒ `IncompatibleSnapshotError`; journal con `state_version > 0` sin snapshot ⇒ error; `warmup_ms > 0`. Verificación: FR-23, FR-30, AC-20 (integración). Commit `feat(daemon): boot sequence with replay and jit warm-up`.
- [x] 6.4 T24 `daemon.py`: `serve`, parse/dispatch, `RpcError` → JSON, `__main__`, señales. Test `tests/test_jsonrpc_loop.py` con `io.StringIO`: parse error; invalid request; batch ⇒ `-32600`; método desconocido; notificación sin `id`; `-32005` antes de `ready`; línea vacía; línea > 1 MiB. Verificación: AC-26, AC-27, FR-29, FR-31–FR-33. Commit `feat(daemon): stdio json-rpc 2.0 loop`.
- [x] 6.5 T25 End-to-end por subprocess: `tests/test_daemon_e2e.py` lanza `python -m azuma_router.daemon --config fixtures/config/azuma.valid.toml` con `snapshot_dir` temporal; `ready`, `health`, `select`, `report`, `health` (`state_version == 1`); SIGTERM ⇒ snapshot y exit 0; relanzar ⇒ `restored_from_snapshot == true`. Verificación: AC-25, NFR-02 (`warmup_ms < 5000`). Commit `test(daemon): end-to-end subprocess lifecycle`.

## 7. OPE y cierre

- [x] 7.1 T26 `ope.py`: `evaluate`. Fixture `fixtures/ope/decisions.jsonl` generado por la simulación de T11; test `tests/test_ope.py`. Verificación: AC-28; `ess` reportado; `alpha` distinto cambia `ips`/`dr` de forma consistente; `ope.py` no importa `jax`. Commit `feat(ope): ips and doubly-robust off-policy evaluation`.
- [x] 7.2 T27 Documentación y verificación final: `README.md` (quickstart con `azuma.toml`, protocolo), `CLAUDE.md` del proyecto (comandos, layout, convenciones). Ejecutar la sección Verification de la ficha §10 y registrar los números en el PR. Verificación: AC-01…AC-28 verdes; p99 del benchmark registrado; `security-audit` sobre el input handling del daemon. Commit `docs: readme, project claude.md and verification results`.

## Orden de dependencias

`0.1 → 1.1 → 1.2 → 1.3 → 2.1 → 2.2 → 3.1 → … → 3.6`; `4.1 → 4.4` solo dependen de `1.3` y pueden ir en paralelo con el bloque 3; `5.x` depende de `1.2` y `4.1`; `6.1` requiere los bloques 3, 4 y 5; `7.1` requiere `3.6` y `5.1`.
