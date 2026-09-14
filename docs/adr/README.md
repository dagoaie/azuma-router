# Mini-ADRs — azuma-router

Formato: contexto · decisión · porqué · rechazado/diferido. Un ADR por decisión del Grill. Estado `Aceptado` = aprobado por el autor en el Grill (Ronda 1 y Ronda 2, 2026-09-14). Un ADR nunca se edita para cambiar la decisión: se sustituye por uno nuevo que lo marca `Supersedido por`.

| ADR | Decisión | Grill |
|---|---|---|
| [0001](0001-jax-state-namedtuple.md) | `BanditState` como `NamedTuple` + `StaticConfig` hashable | Q1 |
| [0002](0002-cholesky-solve-float64.md) | Solver directo por Cholesky en float64, sin mantener $A^{-1}$ | Q2 |
| [0003](0003-context-dim-32-quota-excluded.md) | $d = 32$; utilización de cuota fuera de $x_t$ | Q3 |
| [0004](0004-block-scaling-fail-fast.md) | Escalado por bloque a $[0,1]$, cota $L=\sqrt d$, NaN → excepción | Q4 |
| [0005](0005-quota-ledger-python-deque.md) | `QuotaLedger` en Python con deque exacto; al JIT solo `utilization[K,W]` | Q5 |
| [0006](0006-log-barrier-per-arm-mu.md) | Barrera logarítmica, $\mu$ por brazo, unidad abstracta `units` | Q6 |
| [0007](0007-reward-external-quality.md) | Recompensa Q-C-L con calidad externa, coste real (0 en suscripción) | Q7 |
| [0008](0008-softmax-sampling-daemon-key.md) | Softmax con temperatura y suelo $\varepsilon/K$; PRNG gestionada por el daemon | Q8 |
| [0009](0009-snapshot-safetensors-atomic.md) | Snapshot safetensors + JSON, rename atómico, sin migraciones | Q9 |
| [0010](0010-daemon-stdio-jsonrpc.md) | Daemon stdio JSON-RPC 2.0 (estilo MCP) | Q10 |
| [0011](0011-report-ttl-idempotent.md) | `report` idempotente; decisiones sin report se descartan tras TTL | Q11 |
| [0012](0012-python312-public-api-apache2.md) | Python ≥ 3.12, API pública mínima, Apache-2.0 | Q12 |
| [0013](0013-select-payload-semistructured.md) | `select` semi-estructurado; el prompt nunca viaja por stdio | Q13 |
| [0014](0014-report-payload-error-codes.md) | `report` con `quality` obligatoria; duplicado `applied:false`; errores −32001…−32006 | Q14 |
| [0015](0015-featurizer-blocks.md) | Bloques del Featurizer v1, enum cerrado, interacción | Q15 |
| [0016](0016-jsonl-wal-pendingstore.md) | JSONL como WAL; `PendingStore` por replay; TTL amortizado | Q16 |
| [0017](0017-toml-config-restart.md) | `azuma.toml`; cambio de config = reinicio; snapshot rechaza brazos distintos | Q17 |
| [0018](0018-repo-layout-ci-precommit.md) | Layout `src/`, CI matriz, pre-commit, x64 en `__init__` | Q18 |
