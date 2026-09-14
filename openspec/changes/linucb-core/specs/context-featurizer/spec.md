## Purpose

Transformación determinista de una `RoutingRequest` semi-estructurada a un vector de contexto $x_t \in [0,1]^{32}$ según el layout v1, con rechazo explícito de entradas inválidas y sin dependencias de NLP. Layout por índice: ADR-0015. Decisiones: ADR-0003, ADR-0004, ADR-0013.

## ADDED Requirements

### Requirement: Layout v1 determinista
`featurize(request) -> f64[32]` SHALL ser determinista y cumplir exactamente la tabla de bloques de ADR-0015: índice 0 bias; 1 `log10(1+input_tokens)/6`; 2 `expected_output_tokens` normalizado; 3 `log10(1+diff_lines)/4`; 4–15 one-hot de `task_kind`; 16–18 `has_code/has_tools/has_images`; 19 `min(agent_depth,8)/8`; 20–22 one-hot de `priority`; 23 interacción `x16·x1`; 24–31 padding a cero. `FEATURIZER_VERSION` SHALL valer `1` (FR-04).

#### Scenario: vector de referencia
- **WHEN** se featuriza la petición de ejemplo de `docs/contracts/linucb-core.md` §5.2
- **THEN** el resultado coincide con `fixtures/featurizer/v1_reference.json` con tolerancia `1e-12`, `x[24:32] == 0` y `x[23] == x[16]*x[1]` (AC-03)

#### Scenario: campos opcionales ausentes
- **WHEN** `expected_output_tokens` y `diff_lines` son `None`
- **THEN** `x[2] == 0.0` y `x[3] == 0.0` sin lanzar

### Requirement: Rechazo fail-fast de contexto inválido
El featurizer SHALL lanzar `InvalidContextError(field, reason)` ante `task_kind` fuera del enum cerrado de 12 valores, `priority` fuera de `{low, normal, high}`, conteos negativos, `agent_depth < 0` o valores no finitos. MUST NOT sustituir por un valor por defecto (FR-05).

#### Scenario: task_kind desconocido
- **WHEN** se construye una petición con `task_kind = "unknown"`
- **THEN** se lanza `InvalidContextError` con `field == "task_kind"` (AC-04)

#### Scenario: conteo negativo
- **WHEN** `input_tokens = -1` o `agent_depth = -1`
- **THEN** se lanza `InvalidContextError` con el `field` correspondiente (AC-04)

### Requirement: Salida acotada y verificada
Todo componente del vector SHALL estar en $[0,1]$ y ser finito; el featurizer SHALL verificarlo antes de devolver y lanzar `InvalidContextError` en caso contrario. Valores por encima del rango de normalización (por ejemplo `input_tokens ≥ 10^6`) SHALL saturar a `1.0` (FR-06).

#### Scenario: propiedad de rango
- **WHEN** se featuriza cualquier `RoutingRequest` válida generada por Hypothesis
- **THEN** `0 ≤ x_i ≤ 1` para todo `i` y `‖x‖₂ ≤ √32` (AC-05)

#### Scenario: saturación
- **WHEN** `input_tokens = 10_000_000`
- **THEN** `x[1] == 1.0`
