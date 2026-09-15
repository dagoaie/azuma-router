## Purpose

Núcleo LinUCB disjunto compilado en JAX: estado inmutable PyTree con formas estáticas, `select` y `update` puras y vectorizadas sobre brazos, solve por Cholesky en float64, política de muestreo softmax con propensity no degenerada, y garantía de una única compilación por proceso. Decisiones: ADR-0001, ADR-0002, ADR-0008, ADR-0018.

## ADDED Requirements

### Requirement: Estado inicial con prior ridge
`init_state(config) -> BanditState` SHALL devolver `A[a] = ridge_lambda·I` para todo `a < k_max`, `b = 0`, `n_pulls = 0`, `mask[a] = True` para `a < K` y `False` para el padding, y `rng_key = PRNGKey(seed)`. Los dtypes SHALL ser `float64` para `A` y `b`, `int32` para `n_pulls`, `bool` para `mask` y `uint32[2]` para `rng_key` (FR-07).

#### Scenario: formas y dtypes
- **WHEN** se llama `init_state` con `k_max = 32`, `d = 32`, `K = 2`, `ridge_lambda = 1.0`
- **THEN** `A.shape == (32, 32, 32)`, `A.dtype == float64`, `A[0] == I`, `mask[:2].all()` y `not mask[2:].any()` (AC-06)

### Requirement: Puntuación UCB por brazo
`select(state, x, quota_view, static)` SHALL calcular por brazo, mediante `cho_factor`/`cho_solve` y `solve_triangular` en float64: `radius = sqrt(x·A⁻¹x)`, `exploit = x·θ̂` con `θ̂ = A⁻¹b`, `bonus = alpha·radius`, `shadow = max_W −μ_a·ln(1−u_{a,W})` y `score = exploit + bonus − shadow·est_units`. Brazos inactivos (padding, no permitidos o con `u ≥ 1` en alguna ventana) SHALL recibir `score = −inf`, y la máscara estructural (mask ∧ allowed ∧ ¬exhausted) SHALL exponerse como `diagnostics.active`. `diagnostics.radius` SHALL exponer la magnitud sin escalar por `alpha`, para que el journal (decision-journal) y `ope.evaluate` (off-policy-evaluation) puedan reconstruir el bonus de un `alpha` distinto (FR-08).

#### Scenario: bonus con prior identidad
- **WHEN** `A = I`, `b = 0`, `alpha = 1` y todas las utilizaciones son `0`
- **THEN** `bonus[a] == ‖x‖₂` (±1e-12) y `shadow_price[a] == 0` para todo brazo activo (AC-07, AC-08)

#### Scenario: penalización por precio sombra
- **WHEN** un brazo tiene `u = 0.9` en una ventana y `μ = 1`
- **THEN** `diagnostics.shadow_price[a] == −ln(0.1)` (±1e-9) (AC-08)

#### Scenario: bonus decreciente
- **WHEN** se aplica `update` con observaciones alineadas con `x` al brazo `a`
- **THEN** `bonus[a]` decrece monótonamente tras cada `update`

### Requirement: Política de muestreo con propensity
`select` SHALL muestrear el brazo según $\pi = (1-\varepsilon)\,\mathrm{softmax}(\text{score}/\tau) + \varepsilon/K_{act}$ restringida a brazos activos, devolver `propensity = π[arm]` y `diagnostics.policy` completa. Ningún brazo activo SHALL tener `π < ε/K_act` y la suma sobre activos SHALL ser `1` (±1e-12) (FR-08). El softmax SHALL restar el máximo antes de dividir por `τ`, de modo que `π` sea finita para cualquier `τ > 0` no subnormal aunque `score/τ` desbordara.

#### Scenario: temperatura extrema
- **WHEN** `τ = 1e-300` y un `score` de `1e9`
- **THEN** `policy` es finita y suma `1`

#### Scenario: un solo brazo
- **WHEN** solo hay un brazo activo
- **THEN** `arm` es ese brazo y `propensity == 1.0`

#### Scenario: frecuencia empírica
- **WHEN** se muestrean 20 000 decisiones con el mismo estado y claves consecutivas
- **THEN** la frecuencia de cada brazo coincide con `policy` ±0.02

### Requirement: Pureza y avance de la clave aleatoria
`select` SHALL ser pura: la misma tupla `(state, x, quota_view)` SHALL producir salida bitwise idéntica; `state'` SHALL diferir de `state` únicamente en `rng_key` (FR-09).

#### Scenario: determinismo
- **WHEN** se llama `select` dos veces con el mismo `state`
- **THEN** `arm`, `propensity` y `diagnostics` son bitwise iguales y `state'.rng_key != state.rng_key` (AC-09)

### Requirement: Señal de ausencia de brazos
Cuando la máscara del estado, `quota_view.allowed` y el agotamiento de cuota dejan cero brazos activos, `select` SHALL devolver `diagnostics.no_arm_available = True` y MUST NOT propagar `NaN` en `arm` ni en `propensity` (FR-10).

#### Scenario: todos no permitidos
- **WHEN** `quota_view.allowed` es todo `False`
- **THEN** `diagnostics.no_arm_available == True` (AC-10)

### Requirement: Actualización rank-1 del brazo elegido
`update(state, x, arm, reward, static)` SHALL producir `A'[arm] = A[arm] + x xᵀ`, `b'[arm] = b[arm] + reward·x`, `n_pulls'[arm] = n_pulls[arm] + 1` y dejar los demás brazos bitwise iguales. `A` SHALL permanecer simétrica bitwise y SPD, con `‖Aθ̂ − b‖∞ < 1e-9` (FR-11, FR-12).

#### Scenario: aislamiento entre brazos
- **WHEN** se aplican 1 000 `update` aleatorios (Hypothesis)
- **THEN** cada `A[a]` es simétrica bitwise, `cho_factor(A[a])` no falla, `‖A[a]θ̂ − b[a]‖∞ < 1e-9` y los brazos no actualizados son bitwise iguales (AC-11)

### Requirement: Una única compilación por proceso
`select_jit` y `update_jit` SHALL compilarse exactamente una vez por proceso para un `StaticConfig` dado; el número de compilaciones SHALL ser observable para tests y para `health.jit_compilations` (FR-13).

#### Scenario: invariante de forma
- **WHEN** se ejecutan 10 000 llamadas alternando `select_jit` y `update_jit` con contextos y cuotas distintos
- **THEN** el contador de compilación de `select` es `1` y el de `update` es `1` (AC-12)

### Requirement: Presupuesto de latencia
`select_jit` SHALL tener p99 < 1 ms y `update_jit` p99 < 0.5 ms en CPU con `k_max = 32, d = 32, w_max = 4`, medidos con `pytest-benchmark` tras warm-up sobre ≥ 1 000 iteraciones (NFR-01).

#### Scenario: benchmark
- **WHEN** se ejecuta `uv run pytest tests/bench -q --benchmark-only`
- **THEN** se imprimen los p99 de `select_jit` y `update_jit` y cumplen los umbrales

### Requirement: Regret sublineal
El bandido SHALL exhibir regret acumulado sublineal en simulación con brazos sintéticos lineales.

#### Scenario: simulación
- **WHEN** se simulan 20 000 rondas con contextos aleatorios y `θ*` por brazo de `fixtures/sim/arms_synthetic.json`
- **THEN** el exponente de ajuste log-log del regret acumulado es `< 0.6` y cada brazo recibe ≥ 1 % de los pulls
