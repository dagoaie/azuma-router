## Purpose

Composición de calidad, coste y latencia observados en una recompensa escalar acotada en $[0,1]$, con calidad aportada externamente y coste cero para brazos de suscripción. Decisión: ADR-0007.

## ADDED Requirements

### Requirement: Fórmula de recompensa
`compute_reward(quality, cost_usd, latency_s, static)` SHALL devolver `clip(w_quality·quality − w_cost·min(cost_usd/cost_budget_usd, 1) − w_latency·min(latency_s/latency_budget_s, 1), 0, 1)` (FR-18).

#### Scenario: valor de referencia
- **WHEN** `quality = 0.9`, `cost_usd = 0.0`, `latency_s = 11.2` con los pesos del TOML de ejemplo (`w_quality = 1.0`, `w_cost = 0.3`, `w_latency = 0.2`, `latency_budget_s = 30`)
- **THEN** `reward == 0.8253` (±1e-9) (AC-16)

#### Scenario: propiedad de rango
- **WHEN** se evalúa con cualquier `quality ∈ [0,1]`, `cost_usd ≥ 0`, `latency_s ≥ 0` (Hypothesis)
- **THEN** `0 ≤ reward ≤ 1`

### Requirement: Coste cero en suscripción
Un `report` con `cost_usd > 0` para un brazo `subscription` SHALL rechazarse con `-32602` y `data.field == "cost_usd"` (FR-19).

#### Scenario: coste en brazo de suscripción
- **WHEN** se reporta `cost_usd = 0.01` sobre una decisión cuyo brazo es `subscription`
- **THEN** la respuesta es el error `-32602` y no se aplica `update`
