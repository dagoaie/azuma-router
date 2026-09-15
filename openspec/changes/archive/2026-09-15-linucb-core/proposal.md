## Why

Los orquestadores de agentes asignan petición→modelo de forma estática: ignoran la calidad observada, los precios por token en vivo y las cuotas rodantes (ventanas de ~5 h y techos semanales) de los planes de suscripción, de modo que sobrepagan en tareas triviales y saturan la suscripción justo cuando más se necesita. azuma-router aporta un núcleo de decisión aprendido online (LinUCB disjunto compilado en JAX, regret sublineal por Azuma–Hoeffding) con precios sombra por barrera convexa, latencia < 1 ms, y un daemon JSON-RPC por stdio que Convoy podrá lanzar como proceso hijo.

Contexto completo y evidencia: `proposal/tier-a/0001-azuma-linucb-core.md` (§2–§3). Decisiones: `docs/adr/0001…0018`. Vocabulario: `CONTEXT.md`.

## What Changes

- Nuevo paquete Python `azuma_router` (API pública: `select`, `update`, `BanditState`, `StaticConfig`, `QuotaLedger`, `snapshot`, `restore`; ADR-0012).
- Configuración declarativa `azuma.toml` con brazos, ventanas de cuota y hiperparámetros; cambio de config = reinicio (ADR-0017).
- Featurizer determinista v1 que mapea una petición semi-estructurada a $x_t \in [0,1]^{32}$ con rechazo fail-fast (ADR-0004/0013/0015).
- Bandido LinUCB disjunto: estado inmutable PyTree, `select`/`update` compiladas una sola vez, Cholesky float64, muestreo softmax con propensity no degenerada (ADR-0001/0002/0008).
- Ledger de cuota con ventanas rodantes y de reset fijo, y precio sombra con barrera logarítmica que enmascara brazos agotados (ADR-0005/0006).
- Recompensa Q-C-L escalarizada con calidad aportada por el llamador (ADR-0007).
- Journal de decisiones JSONL que actúa como WAL (fases `select`/`report`/`expire`), `PendingStore` con TTL y `report` idempotente (ADR-0011/0016).
- Snapshot atómico safetensors + JSON con rechazo fail-fast de incompatibilidades, sin migración (ADR-0009).
- Daemon JSON-RPC 2.0 sobre stdio con `ready`, `select`, `report`, `snapshot`, `health` y códigos de error −32001…−32006 (ADR-0010/0014).
- Evaluador off-policy IPS / Doubly-Robust sobre el journal.

Sin cambios **BREAKING**: no existe versión previa.

## Capabilities

### New Capabilities

- `routing-config`: carga y validación estricta de `azuma.toml` (brazos, ventanas, hiperparámetros) y su proyección a la configuración estática del bandido.
- `context-featurizer`: transformación determinista de una petición de enrutamiento a un vector de contexto acotado, con rechazo explícito de entradas inválidas.
- `linucb-bandit`: selección y actualización LinUCB vectorizada con estado inmutable, política de muestreo con propensity, invariante de una única compilación y presupuesto de latencia.
- `quota-shadow-pricing`: ledger de consumo por ventana, utilización, enmascarado por agotamiento, precio sombra por barrera logarítmica y estimación/corrección de consumo.
- `reward-scalarization`: composición calidad–coste–latencia en una recompensa en $[0,1]$ con coste cero para brazos de suscripción.
- `decision-journal`: JSONL append-only como WAL y dataset OPE, decisiones pendientes con TTL, idempotencia de `report`, replay en arranque.
- `state-snapshot`: persistencia atómica y restauración fail-fast del estado del bandido y del ledger.
- `routing-daemon`: protocolo JSON-RPC 2.0 sobre stdio, ciclo de vida (arranque, `ready`, apagado limpio), métodos y errores.
- `off-policy-evaluation`: estimadores IPS y DR sobre el journal para calibrar hiperparámetros sin desplegar.

### Modified Capabilities

(ninguna — primer change del repositorio)

## Impact

- Código nuevo: `src/azuma_router/` (≈1 150 LOC) y `tests/` (≈1 550 LOC); ver `tasks.md`.
- Dependencias de runtime: `jax`/`jaxlib` ≥ 0.11.1, `safetensors`. Sin dependencias de red, sin pickle, sin sockets.
- Consumidor futuro: plugin TypeScript de Convoy (change posterior, follow-up F1). Este change fija el contrato del protocolo que ese plugin implementará.
- Contratos detallados (TOML, payloads JSON-RPC, JSONL, layout del snapshot, edge cases y AC-01…AC-28): `docs/contracts/linucb-core.md`.
