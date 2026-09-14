## Purpose

Daemon single-thread que expone el router como proceso hijo por JSON-RPC 2.0 sobre stdio: ciclo de vida (arranque, `ready`, apagado limpio), métodos `select`/`report`/`snapshot`/`health`, códigos de error y observabilidad. Payloads exactos: `docs/contracts/linucb-core.md` §5. Decisiones: ADR-0010, ADR-0013, ADR-0014.

## ADDED Requirements

### Requirement: Framing por líneas
El daemon SHALL leer una petición JSON-RPC 2.0 por línea UTF-8 de `stdin` y escribir una respuesta por línea en `stdout` con `flush`; `stderr` SHALL contener únicamente logs JSON estructurados (`ts`, `level`, `event`). Líneas vacías SHALL ignorarse; líneas > 1 MiB SHALL responderse con `-32600` y `data.reason = "line_too_long"` (FR-29, NFR-04).

#### Scenario: línea vacía y línea larga
- **WHEN** llega una línea vacía y después una de 2 MiB
- **THEN** la primera no produce respuesta y la segunda produce `-32600` con `data.reason == "line_too_long"`

### Requirement: Arranque y notificación ready
`Router.boot()` SHALL, en orden: cargar config, restaurar snapshot o inicializar estado, hacer replay del journal, ejecutar warm-up (una llamada real a `select_jit` y a `update_jit`) y emitir la notificación `ready` con `state_version`, `arms`, `restored_from_snapshot`, `pending_replayed` y `warmup_ms`. Toda petición anterior a `ready` SHALL responder `-32005 NotReady`. El tiempo hasta `ready` SHALL ser < 5 s en CPU (FR-30, NFR-02).

#### Scenario: petición prematura
- **WHEN** llega un `health` antes de que el loop marque `ready`
- **THEN** la respuesta es el error `-32005` (AC-26)

#### Scenario: arranque con snapshot y pendientes
- **WHEN** existen `snapshot/` y un journal con 3 `select` sin `report`
- **THEN** `ready.restored_from_snapshot == true` y `ready.pending_replayed == 3`

### Requirement: Métodos y errores de protocolo
Los métodos SHALL ser `select`, `report`, `snapshot` y `health`. Método desconocido ⇒ `-32601`; JSON inválido ⇒ `-32700`; falta `jsonrpc: "2.0"` o `method` ⇒ `-32600`; batch (array) ⇒ `-32600`; `params` presente y no objeto ⇒ `-32602` con `data.field = "params"`; params inválidos ⇒ `-32602` con `data.field`. Una línea que el decodificador no puede anidar ⇒ `-32700` y el bucle continúa. Una petición **sin** `id` SHALL procesarse sin responder y registrarse en stderr; un `"id": null` explícito SHALL responderse con `id: null` (FR-31, FR-33).

#### Scenario: id nulo explícito
- **WHEN** llegan `{"jsonrpc":"2.0","id":null,"method":"health"}` y `{"jsonrpc":"2.0","method":"health"}`
- **THEN** la primera recibe una respuesta con `id: null` y la segunda ninguna

#### Scenario: params que no es objeto
- **WHEN** llega `{"jsonrpc":"2.0","id":1,"method":"select","params":[]}`
- **THEN** la respuesta es `-32602` con `id: 1` y `data.field == "params"`

#### Scenario: batch rechazado
- **WHEN** llega `[{"jsonrpc":"2.0","id":1,"method":"health"}]`
- **THEN** la respuesta es `-32600` (AC-27)

#### Scenario: método desconocido
- **WHEN** llega `method = "predict"`
- **THEN** la respuesta es `-32601`

### Requirement: Procesamiento secuencial
Las peticiones SHALL procesarse una a una en orden de llegada en un único hilo; el daemon MUST NOT abrir sockets (FR-32, NFR-07).

#### Scenario: orden preservado
- **WHEN** se escriben tres peticiones con `id` 1, 2, 3 en una sola escritura
- **THEN** las respuestas llegan en el orden 1, 2, 3

### Requirement: Secuencia de select
`Router.select(params)` SHALL: barrer expirados y evaluar la rotación del journal (los `expire` cuentan aunque la petición se rechace); validar `params.request` (`-32602`); featurizar (`-32001 InvalidContext` con `data.field` y `data.reason`); resolver `allowed_arms` (`-32602` con `data.field = "allowed_arms[i]"` si un id es desconocido); calcular `est_units`; construir `QuotaView`; invocar `select_jit`; si `no_arm_available`, responder `-32004` con `data.masked_reason` por brazo (`quota_exhausted | not_allowed | disabled`) sin escribir journal ni avanzar estado; en caso contrario registrar el consumo estimado, escribir la línea `select`, guardar la pendiente, avanzar `state` y responder con `decision_id`, `arm`, `propensity` y `diagnostics` (omitido si `verbose = false`). `select` SHALL verificar `isfinite(score)` sobre los brazos activos fuera del JIT y abortar con código 4 si falla.

#### Scenario: contexto inválido
- **WHEN** `params.request.task_kind = "unknown"`
- **THEN** la respuesta es `-32001` con `data.field == "task_kind"` y no se escribe línea `select`

#### Scenario: sin brazo disponible
- **WHEN** `allowed_arms` contiene solo brazos con cuota agotada
- **THEN** la respuesta es `-32004` con `masked_reason` por brazo y el journal no crece

### Requirement: Secuencia de report y snapshot automático
`Router.report(params)` SHALL validar `quality ∈ [0,1]`, `cost_usd ≥ 0`, `latency_s ≥ 0`, tokens enteros `≥ 0`; aplicar el orden de `decision-journal`; responder `{applied, reward, state_version}`; y después evaluar `maybe_snapshot()`, que SHALL hacer snapshot si `updates_since_snapshot ≥ snapshot_every_updates` o `now − last_snapshot ≥ snapshot_every_s` (FR-28).

#### Scenario: snapshot por conteo y por tiempo
- **WHEN** se aplican 100 `report`, o bien 1 `report` con el reloj avanzado 31 s
- **THEN** en ambos casos existe un snapshot nuevo y `health.updates_since_snapshot == 0`

### Requirement: Métodos snapshot y health
`snapshot` SHALL barrer expirados, escribir el snapshot y responder `{state_version, path, bytes, expired_pending}`. `health` SHALL responder `{ready, state_version, pending, updates_since_snapshot, arms: {arm_id: {mask, n_pulls, utilization}}, jit_compilations: {select, update}}`.

#### Scenario: health tras un ciclo
- **WHEN** se ejecuta `select` → `report` → `health`
- **THEN** `health.state_version == 1`, `health.pending == 0` y `jit_compilations == {"select": 1, "update": 1}`

### Requirement: Apagado y fallos
Ante EOF en `stdin`, `SIGTERM` o `SIGINT`, el daemon SHALL terminar la petición en curso, hacer snapshot y salir con código 0: una señal con petición en curso SHALL diferirse (`shutdown_deferred`) hasta que esa petición haya respondido, y una señal en reposo SHALL cerrar de inmediato. `IncompatibleSnapshotError` en arranque ⇒ `incompatible_snapshot` en stderr y código 3; snapshot o journal ilegibles ⇒ `snapshot_read_failed` / `replay_failed` y código 3, nunca un traceback. Excepción no controlada durante una petición ⇒ log en stderr, snapshot solo si el estado es finito, y código 4.

#### Scenario: señal durante un report
- **WHEN** llega `SIGTERM` mientras `report` está aplicando `update_jit`
- **THEN** el `report` responde `applied: true`, no se procesa ninguna petición posterior, el proceso sale con código 0 y el snapshot contiene `state_version == 1`

#### Scenario: snapshot de otro esquema
- **WHEN** `snapshot/state.json` declara un `schema_version` distinto
- **THEN** stderr contiene `incompatible_snapshot` y el proceso sale con código 3

#### Scenario: ciclo de vida end-to-end
- **WHEN** se lanza `python -m azuma_router.daemon --config fixtures/config/azuma.valid.toml` por subprocess, se espera `ready`, se envían `health`, `select`, `report`, `health` y se envía `SIGTERM`
- **THEN** `health.ready == true`, el segundo `health.state_version == 1`, el proceso sale con código 0 y existe `snapshot/`; al relanzar, `ready.restored_from_snapshot == true` y `warmup_ms < 5000` (AC-25, NFR-02)

### Requirement: Determinismo del journal
Con el mismo `seed` y la misma secuencia de peticiones y reports, el journal SHALL ser bitwise idéntico salvo en los campos `ts` (NFR-06).

#### Scenario: dos ejecuciones
- **WHEN** se ejecuta la misma secuencia de 50 `select`/`report` en dos routers con `seed = 42`
- **THEN** los journals coinciden línea a línea tras eliminar `ts`
