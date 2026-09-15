# off-policy-evaluation Specification

## Purpose
Evaluación off-policy (IPS y Doubly-Robust) de una política candidata sobre el journal de decisiones, para calibrar `alpha`, `tau` y `epsilon_floor` sin desplegar. Decisión: ADR-0008.

## Requirements

### Requirement: Estimadores IPS y DR
`ope.evaluate(log_path, candidate_static) -> {ips, dr, n, ess}` SHALL emparejar cada línea `report` con su `select`, usar la `propensity` registrada como política de logging, y reconstruir el `score` candidato como `score_logged + bonus_candidate − bonus_logged`, donde `bonus_candidate = alpha_candidate · radius` a partir del `radius` (magnitud de confianza sin escalar) persistido en la línea `select`; así conserva la penalización de cuota específica de cada brazo ya incluida en el `score` registrado. Reescalar el `bonus` registrado por `alpha_candidate / alpha_logged` está prohibido: es indefinido cuando `alpha_logged = 0` y reproduce silenciosamente la política registrada en vez del candidato. Los brazos activos para la política candidata SHALL determinarse por la máscara `active` persistida en la línea `select`, nunca por `policy_logged > 0` (una política registrada puede hacer underflow a 0 bajo un `epsilon_floor` distinto sin que el brazo deje de estar estructuralmente activo). Deberá calcular la política candidata con la misma fórmula softmax en NumPy — restando el `score` máximo entre los brazos activos antes de dividir por `tau`, igual que el paso online (R4-SF-1) — y devolver IPS, DR (con `r̂ = exploit` registrado), número de pares `n` y tamaño muestral efectivo `ess`. MUST NOT importar `jax` (FR-34), ni siquiera transitivamente vía `azuma_router/__init__.py` o un módulo de tipos compartido: los tipos JAX-free consumidos por `ope.evaluate` (`StaticConfig`, `Pending`, etc.) viven en `azuma_router.domain`, que `ope.py`, `wal.py`, `config.py` y `featurizer.py` importan directamente en vez de `azuma_router.types` (que sí importa `jax`), y `azuma_router/__init__.py` expone sus símbolos JAX (`select`, `update`, `QuotaLedger`, `snapshot`, `restore`, `BanditState`, `StaticConfig`) de forma perezosa (PEP 562) para que importar el paquete o cualquiera de sus submódulos no cargue `jax` (JAX-1).

`evaluate` SHALL leer el segmento rotado retenido (`<stem>.1<suffix>`, la misma convención que `wal.DecisionLog.rotate`/`wal.replay_log`) antes del segmento activo, de modo que un par `select`/`report` cerrado por completo antes de una rotación no se pierda silenciosamente de la estimación. Un `select` reenviado byte-idéntico al segmento activo tras la rotación (carry-forward) SHALL contarse una única vez.

#### Scenario: misma política
- **WHEN** se evalúa `fixtures/ope/decisions.jsonl` con la misma configuración que lo generó
- **THEN** `ips ≈ dr ≈ media de reward` (±0.02) y `n` es igual al número de pares (AC-28)

#### Scenario: política distinta
- **WHEN** se evalúa con `alpha` distinto de la registrada
- **THEN** `ips` y `dr` cambian de forma consistente y `ess < n`

#### Scenario: par cerrado antes de una rotación
- **WHEN** un `select` y su `report` se cierran por completo dentro del segmento rotado retenido, y el segmento activo solo contiene pares posteriores
- **THEN** `n` cuenta también el par cerrado en el segmento rotado, y un `select` reenviado byte-idéntico al segmento activo no se cuenta dos veces

#### Scenario: `import azuma_router.ope` en un intérprete nuevo (JAX-1)
- **WHEN** se lanza un subproceso nuevo que únicamente ejecuta `import azuma_router.ope`
- **THEN** ni `jax` ni `jaxlib` aparecen en `sys.modules`

### Requirement: Lectura tolerante del journal
El evaluador SHALL ignorar líneas `expire` y `select` sin `report`, y SHALL fallar con un error explícito ante un `report` sin `select` emparejable. La decodificación SHALL ser estricta — compartiendo o replicando el decodificador de journal de `wal.py` —: rechazar cualquier constante JSON no finita (`NaN`, `Infinity`, `-Infinity`) y cualquier línea malformada salvo la última, que se descarta en silencio únicamente cuando no termina en salto de línea (artefacto de un crash a mitad de escritura, FR-24). Un `except Exception: continue` genérico que descarte cualquier línea malformada sin distinguir esos dos casos está prohibido.

#### Scenario: report huérfano
- **WHEN** el journal contiene un `report` cuyo `decision_id` no aparece en ningún `select`
- **THEN** `evaluate` lanza un error que nombra el `decision_id`

#### Scenario: constante no finita
- **WHEN** una línea del journal contiene `NaN` o `Infinity` en cualquier campo
- **THEN** `evaluate` lanza un error en vez de propagar el valor no finito a la estimación

#### Scenario: línea final truncada
- **WHEN** la última línea del journal está incompleta y no termina en salto de línea (artefacto de un crash a mitad de escritura)
- **THEN** `evaluate` descarta esa línea y evalúa igualmente los pares completos previos
