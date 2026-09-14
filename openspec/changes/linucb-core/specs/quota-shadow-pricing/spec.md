## Purpose

Ledger exacto de consumo por brazo con ventanas rodantes y de reset fijo, vista de utilización para el bandido, enmascarado por agotamiento, precio sombra por barrera logarítmica y estimación/corrección del consumo por decisión. Decisiones: ADR-0005, ADR-0006, ADR-0011.

## ADDED Requirements

### Requirement: Ventanas y utilización
`QuotaLedger.record(arm_index, units, ts)` y `QuotaLedger.utilization(now) -> f32[k_max, w_max]` SHALL devolver `u = consumido_en_ventana / capacity`, donde `RollingWindow(duration_s, capacity)` cuenta los últimos `duration_s` segundos y `FixedResetWindow(period, reset_anchor, capacity)` cuenta desde el último instante de reset alineado con `reset_anchor`. Slots sin ventana SHALL valer `0.0`. El reloj SHALL ser inyectable (FR-14).

#### Scenario: ventana rodante olvida
- **WHEN** se registran 10 units en `t = 0` con `RollingWindow(duration_s = 100, capacity = 20)` y se consulta en `t = 101`
- **THEN** `utilization == 0.0` (AC-13)

#### Scenario: reset fijo semanal
- **WHEN** se registran units antes del siguiente instante de reset semanal y se consulta después
- **THEN** la utilización de esa ventana es `0.0` (AC-13)

### Requirement: Enmascarado por agotamiento
Un brazo con `u ≥ 1.0` en cualquier ventana SHALL quedar inactivo en la `QuotaView` de ese `select`, con independencia de `barrier_mu` (FR-15).

#### Scenario: cuota agotada
- **WHEN** un brazo alcanza `u = 1.0` en una ventana
- **THEN** `select` lo excluye y `diagnostics.score[a] == −inf` (AC-14)

### Requirement: Estimación y corrección del consumo
El consumo estimado `est_units = units_per_request + units_per_1k_tokens·(input_tokens + expected_output_tokens)/1000` SHALL registrarse en `select`; `report` SHALL reemplazar ese mismo evento por el consumo real calculado con `actual_*_tokens`, MUST NOT añadir un segundo evento (FR-16).

#### Scenario: corrección in-place
- **WHEN** se hace `event_id = record(a, 2.1, ts)` y después `correct(event_id, 2.54)`
- **THEN** el ledger contiene exactamente un evento para `a` con `units == 2.54` (AC-15)

#### Scenario: sin expected_output_tokens
- **WHEN** la petición no informa `expected_output_tokens`
- **THEN** `est_units` usa únicamente `input_tokens`

### Requirement: Brazos per_token sin precio sombra
Brazos `per_token` SHALL tener `shadow = 0` siempre y fila de utilización a cero (FR-17).

#### Scenario: fila a cero
- **WHEN** se consulta `utilization(now)` para un brazo `per_token`
- **THEN** su fila es `[0.0]*w_max` y `diagnostics.shadow_price[a] == 0`

### Requirement: Barrera logarítmica con paridad NumPy/JAX
`quota.log_barrier(u, mu) = −mu·ln(1−clip(u, 0, 1−1e-6))` SHALL existir como función NumPy pura y coincidir con el `shadow_price` calculado dentro del JIT (ADR-0006).

#### Scenario: paridad
- **WHEN** se evalúan ambas implementaciones sobre una malla de `u ∈ [0, 0.999]` y `mu ∈ {0.1, 0.5, 1, 2}`
- **THEN** difieren menos de `1e-12` y la función es convexa y monótona creciente en `u`

### Requirement: Persistencia del ledger
`QuotaLedger.dump()` y `QuotaLedger.load(dump, config)` SHALL preservar todos los eventos vivos para su inclusión en el snapshot.

#### Scenario: round-trip
- **WHEN** se hace `load(dump())` sobre un ledger con eventos en varias ventanas
- **THEN** `utilization(now)` es idéntica antes y después
