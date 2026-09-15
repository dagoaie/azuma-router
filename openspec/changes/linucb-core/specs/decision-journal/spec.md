## Purpose

Journal JSONL append-only que actúa simultáneamente como WAL (write-ahead log) para reconstruir decisiones pendientes tras un reinicio y como dataset para evaluación off-policy; incluye el `PendingStore` con TTL, la idempotencia de `report` y el replay en arranque. Schemas de línea: `docs/contracts/linucb-core.md` §4.4. Decisiones: ADR-0011, ADR-0016.

## ADDED Requirements

### Requirement: Escritura del select antes de responder
Antes de responder a `select`, el daemon SHALL haber hecho `append` + `flush` de una línea `phase = "select"` con `decision_id` (UUIDv7 ordenable por tiempo con parte aleatoria; la correspondencia `select`↔`report` es determinista tras normalizar por ordinal, NFR-06), `state_version`, `selection_sequence`, la `rng_key` posterior, `arm_index`, `arm_id`, `propensity`, `x`, `policy`, `score`, `exploit`, `bonus`, `shadow_price`, `utilization`, `est_units` y `featurizer_version`, serializada con `json.dumps(separators=(",", ":"))` (FR-20).

#### Scenario: flush antes de retornar
- **WHEN** se invoca `Router.select` con un `flush` espiado
- **THEN** el `flush` ocurre antes de que `select` retorne y el fichero contiene la línea `select` (AC-17)

### Requirement: Orden del report e idempotencia
`report` sobre un `decision_id` pendiente SHALL, en este orden: escribir `phase = "report"` con `state_version` ya incrementado, corregir el ledger, aplicar `update`, mover la decisión a aplicada. Un `report` repetido sobre una decisión aplicada SHALL responder `{"applied": false, "reason": "duplicate", "state_version": n}` sin escribir ninguna línea (FR-21).

#### Scenario: report duplicado
- **WHEN** se envía el mismo `report` dos veces
- **THEN** el journal contiene exactamente una línea `report` y la segunda respuesta tiene `applied == false` (AC-18)

### Requirement: Expiración por TTL
Decisiones con `now − ts_select > pending_ttl_s` SHALL expirar en el barrido ejecutado al inicio de `select` y de `snapshot`: se escribe `phase = "expire"`, MUST NOT aplicarse `update`, y el consumo estimado permanece en el ledger. El barrido SHALL recorrer el `PendingStore` en orden de inserción y detenerse en la primera decisión no expirada (FR-22).

#### Scenario: expiración
- **WHEN** `pending_ttl_s = 1`, hay una decisión pendiente y el reloj inyectado avanza 2 s antes del siguiente `select`
- **THEN** se escribe una línea `expire` y `n_pulls` no cambia (AC-19)

### Requirement: Errores de report
`report` sobre un `decision_id` desconocido SHALL responder `-32002 UnknownDecision`; sobre uno expirado, `-32003 DecisionExpired` con `data.expired_at` (FR-24).

#### Scenario: desconocido y expirado
- **WHEN** se reporta un id nunca emitido y, por separado, uno ya expirado
- **THEN** las respuestas son `-32002` y `-32003` respectivamente (AC-21)

### Requirement: Replay en arranque
Al arrancar, el daemon SHALL reconstruir el `PendingStore` leyendo el fichero rotado anterior y el activo, en orden: **toda** línea `select` retenida sin `report`/`expire` posterior es un pendiente vivo, sea cual sea su `state_version` (los `select` no lo avanzan), y el ledger se reconstruye de forma idempotente por `ledger_event_id`. Las transiciones PRNG de `select` posteriores a `SnapshotMeta.selection_sequence` SHALL ser contiguas y la última SHALL restaurar la key; líneas arrastradas por rotación con una secuencia ya incluida en el snapshot se ignoran para este fin. Solo las líneas `report` con `state_version > snapshot.state_version` SHALL reaplicarse (`update` con la `x` y `arm_index` de su `select`). Antes de mutar estado, cada línea SHALL validarse contra su esquema: `decision_id` string no vacío; en `select`, `arm_index` dentro de la config y ligado a `arm_id`, `x` con `d` componentes finitas, `propensity` en `(0, 1]`, `est_units ≥ 0`, `ts` finito, `state_version ≥ 0`, `ledger_event_id ≥ 1`; en `report`, `reward` en `[0, 1]`, `actual_units ≥ 0`, `state_version` entero; en `expire`, `ts_select` finito. Un `select` duplicado SHALL ser válido solo si es la copia idéntica arrastrada por rotación, y una segunda fase terminal para la misma decisión SHALL rechazarse. Las constantes JSON `NaN`/`Infinity` SHALL rechazarse (el escritor nunca las emite). Un `report` o `expire` sin `select` retenido, una transición PRNG ausente/discontinua o cualquier campo inválido ⇒ journal corrupto ⇒ `replay_failed` ⇒ salida 3, nunca una traza ni un `update` con datos inválidos. Una última línea truncada SHALL descartarse; una línea corrupta no final SHALL abortar el arranque con código 3 (FR-23). Los `report` reaplicados SHALL ser contiguos en `state_version` a partir de `snapshot.state_version + 1`; un hueco ⇒ `replay_failed` ⇒ salida 3. Sin `snapshot/`, un `decisions.1.jsonl` presente ⇒ inconsistencia (`wal_inconsistent`, la rotación siempre sigue a un snapshot) ⇒ salida 3; un `decisions.jsonl` sin rotar SHALL reproducirse íntegro desde el estado inicial.

#### Scenario: kill antes del primer snapshot
- **WHEN** un daemon nuevo aplica un `report` y muere sin snapshot ni rotación
- **THEN** al relanzar, `state_version == 1` y el brazo reportado tiene `n_pulls == 1`

#### Scenario: hueco en los reports
- **WHEN** el journal conserva el `report` con `state_version == 2` pero no el `1`
- **THEN** stderr contiene `replay_failed` y el proceso sale con código 3

#### Scenario: pendiente reconstruida
- **WHEN** se hace replay de un JSONL con un `select` sin `report`
- **THEN** el `PendingStore` contiene 1 decisión; si el JSONL incluye el `report` posterior, contiene 0 (AC-20)

#### Scenario: línea semánticamente inválida
- **WHEN** una línea parsea como JSON pero no es un registro válido (`decision_id` ausente, `x` de dimensión incorrecta o con `NaN`, `arm_index` que no corresponde a `arm_id`, `propensity == 0`, `reward` fuera de `[0, 1]`, `expire` sin `select`)
- **THEN** stderr contiene `replay_failed` y el proceso sale con código 3 sin haber aplicado ningún `update`

#### Scenario: cola corrupta
- **WHEN** la última línea del fichero está truncada
- **THEN** el replay la descarta y continúa; si la línea corrupta no es la última, lanza error

### Requirement: Rotación acotada
`DecisionLog` SHALL rotar el fichero activo a `decisions.1.jsonl` (conservando como máximo un fichero rotado) cuando su tamaño supere `decision_log_max_bytes`, evaluado tras escribir cualquier línea (`select`, `report` o `expire`). Antes de rotar, el daemon SHALL escribir un snapshot (ningún `update` queda solo en el fichero rotado) y el fichero activo nuevo SHALL comenzar con las líneas `select` de todos los pendientes vivos, arrastradas íntegras.

#### Scenario: rotación
- **WHEN** el fichero supera `max_bytes` tras una línea `select` o `report`
- **THEN** existe `decisions.1.jsonl` con el contenido anterior, hay un snapshot con el `state_version` actual y el activo contiene solo las líneas `select` de los pendientes vivos
