## Purpose

Persistencia atómica del estado del bandido y del ledger en safetensors + JSON, y restauración fail-fast que rechaza cualquier incompatibilidad de forma o versión sin migrar. Layout en disco: `docs/contracts/linucb-core.md` §4.3. Decisiones: ADR-0009, ADR-0017.

## ADDED Requirements

### Requirement: Escritura atómica
`snapshot(dir, state, ledger, meta)` SHALL escribir `state.safetensors` (tensores `A`, `b`, `n_pulls`, `mask`, `rng_key`) y `state.json` (`SnapshotMeta`) en `<dir>/snapshot.tmp/`, hacer `fsync` de ambos ficheros y del directorio, y publicar mediante rename de directorio: `snapshot/ → snapshot.old/`, `snapshot.tmp/ → snapshot/`, borrar `snapshot.old/`. MUST NOT usar `pickle` (FR-25, NFR-07).

#### Scenario: layout tras snapshot
- **WHEN** se invoca `snapshot`
- **THEN** existen `snapshot/state.safetensors` y `snapshot/state.json` y no existe `snapshot.tmp/` (AC-22)

#### Scenario: crash entre renames
- **WHEN** al arrancar existe `snapshot.old/` y no `snapshot/`
- **THEN** `restore` recupera `snapshot.old/` como snapshot válido

### Requirement: Restauración fail-fast
`restore(dir, config) -> (state, ledger, meta)` SHALL lanzar `IncompatibleSnapshotError` si difiere cualquiera de `schema_version`, `d`, `k_max`, `w_max`, `featurizer_version` o `arm_ids` (con el diff de listas en el mensaje), antes de leer ningún tensor. Un `config_sha256` distinto con formas y `arm_ids` iguales SHALL aceptarse con un `warning` en stderr. MUST NOT migrar (FR-26).

#### Scenario: incompatibilidades
- **WHEN** el snapshot difiere en `arm_ids`, en `d` o en `featurizer_version`
- **THEN** se lanza `IncompatibleSnapshotError` en cada caso (AC-23)

#### Scenario: hiperparámetros cambiados
- **WHEN** solo cambia `alpha` en la config
- **THEN** `restore` acepta el snapshot y emite `warning`

### Requirement: Round-trip bitwise
`restore(snapshot(s)) == s` SHALL cumplirse bitwise para todos los arrays y el ledger (FR-27).

#### Scenario: round-trip
- **WHEN** se hace snapshot de un estado tras 50 updates y se restaura
- **THEN** `A`, `b`, `n_pulls`, `mask`, `rng_key` son bitwise iguales y `utilization(now)` coincide (AC-24)

### Requirement: Residuos al arrancar
Al arrancar, un `snapshot.tmp/` residual SHALL borrarse. Si no existe `snapshot/` pero el journal contiene `state_version > 0`, el daemon SHALL abortar con código 3 por inconsistencia.

#### Scenario: journal sin snapshot
- **WHEN** no hay `snapshot/` y `decisions.jsonl` contiene una línea `report` con `state_version = 5`
- **THEN** el arranque falla con código de salida 3 y un error en stderr
