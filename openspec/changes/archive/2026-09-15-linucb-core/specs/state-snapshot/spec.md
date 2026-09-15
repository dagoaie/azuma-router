## Purpose

Persistencia atómica del estado del bandido y del ledger en safetensors + JSON, y restauración fail-fast que rechaza cualquier incompatibilidad de forma o versión sin migrar. Layout en disco: `docs/contracts/linucb-core.md` §4.3. Decisiones: ADR-0009, ADR-0017.

## ADDED Requirements

### Requirement: Escritura atómica
`snapshot(dir, state, ledger, meta)` SHALL escribir `state.safetensors` (tensores `A`, `b`, `n_pulls`, `mask`, `rng_key`) y `state.json` (`SnapshotMeta`, schema 2 con `selection_sequence`) en `<dir>/snapshot.tmp/`, hacer `fsync` de ambos ficheros y del directorio, y publicar mediante rename de directorio: `snapshot/ → snapshot.old/`, `snapshot.tmp/ → snapshot/`, borrar `snapshot.old/`. El directorio de estado SHALL crearse con modo `0700` desde el primer arranque, con independencia del `umask` y antes de cualquier snapshot (lo crea quien lo necesite primero: el WAL o `snapshot`); un directorio padre preexistente SHALL NOT modificarse. Los ficheros persistidos SHALL crearse con `0600`; `restore` SHALL NOT alterar permisos. MUST NOT usar `pickle` (FR-25, NFR-07).

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
`restore` SHALL verificar antes de convertir a JAX el conjunto exacto de tensores, sus formas y dtypes, finitud de `A`/`b`, simetría y SPD de `A`, contadores no negativos, máscara canónica y forma de `rng_key`; también SHALL validar el esquema canónico del ledger, sus brazos configurados exactos, ids positivos únicos, allocator consistente y unidades/timestamps finitos no negativos. Cualquier incumplimiento SHALL producir `IncompatibleSnapshotError`. `restore(snapshot(s)) == s` SHALL cumplirse bitwise para todos los arrays y el ledger (FR-27).

#### Scenario: round-trip
- **WHEN** se hace snapshot de un estado tras 50 updates y se restaura
- **THEN** `A`, `b`, `n_pulls`, `mask`, `rng_key` son bitwise iguales y `utilization(now)` coincide (AC-24)

### Requirement: Residuos al arrancar
Al arrancar, un `snapshot.tmp/` residual SHALL borrarse. Si no existe `snapshot/` pero sí `decisions.1.jsonl`, el daemon SHALL abortar con código 3 (`wal_inconsistent`): la rotación siempre sigue a un snapshot, así que el snapshot se ha perdido. Un `decisions.jsonl` sin rotar y sin `snapshot/` SHALL reproducirse íntegro desde el estado inicial.

#### Scenario: segmento rotado sin snapshot
- **WHEN** no hay `snapshot/` y existe `decisions.1.jsonl`
- **THEN** el arranque falla con código de salida 3 y `wal_inconsistent` en stderr

#### Scenario: journal sin rotar sin snapshot
- **WHEN** no hay `snapshot/` y `decisions.jsonl` contiene un `report` con `state_version = 1`
- **THEN** el daemon arranca con `state_version == 1`
