## Purpose

Evaluación off-policy (IPS y Doubly-Robust) de una política candidata sobre el journal de decisiones, para calibrar `alpha`, `tau` y `epsilon_floor` sin desplegar. Decisión: ADR-0008.

## ADDED Requirements

### Requirement: Estimadores IPS y DR
`ope.evaluate(log_path, candidate_static) -> {ips, dr, n, ess}` SHALL emparejar cada línea `report` con su `select`, usar la `propensity` registrada como política de logging, y reconstruir el `score` candidato como `score_logged + bonus_candidate − bonus_logged`, donde `bonus_candidate` reescala el `bonus` registrado por `alpha_candidate / alpha_logged`; así conserva la penalización de cuota específica de cada brazo ya incluida en el `score` registrado. Deberá calcular la política candidata con la misma fórmula softmax en NumPy y devolver IPS, DR (con `r̂ = exploit` registrado), número de pares `n` y tamaño muestral efectivo `ess`. MUST NOT importar `jax` (FR-34).

#### Scenario: misma política
- **WHEN** se evalúa `fixtures/ope/decisions.jsonl` con la misma configuración que lo generó
- **THEN** `ips ≈ dr ≈ media de reward` (±0.02) y `n` es igual al número de pares (AC-28)

#### Scenario: política distinta
- **WHEN** se evalúa con `alpha` distinto de la registrada
- **THEN** `ips` y `dr` cambian de forma consistente y `ess < n`

### Requirement: Lectura tolerante del journal
El evaluador SHALL ignorar líneas `expire` y `select` sin `report`, y SHALL fallar con un error explícito ante un `report` sin `select` emparejable.

#### Scenario: report huérfano
- **WHEN** el journal contiene un `report` cuyo `decision_id` no aparece en ningún `select`
- **THEN** `evaluate` lanza un error que nombra el `decision_id`
