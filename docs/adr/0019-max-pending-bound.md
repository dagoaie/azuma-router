# ADR-0019 — `bandit.max_pending` acota el `PendingStore`; `select` rechaza con `-32007`

**Estado:** Aceptado · 2026-10-23 · Hallazgo #1 de la auditoría de seguridad del daemon

## Contexto
Un cliente que llama a `select` sin llamar nunca a `report` crece sin límite durante `pending_ttl_s`: cada `select` añade una entrada en memoria (`PendingStore`), una línea en el journal y un evento en el ledger. La rotación copia además todos los pendientes al nuevo segmento (`carry_forward`). El tope de línea de 1 MiB protege cada petición, pero no el agregado.

## Decisión
- Clave de config obligatoria `bandit.max_pending` (entero `> 0`; ejemplo `10000`, igual que la LRU de `applied_ids`). Obligatoria por la política de corte duro: una sola ruta de código, sin valor por defecto silencioso.
- Al inicio de `select`, tras el barrido TTL y la rotación, si `len(PendingStore) >= max_pending` ⇒ `-32007 TooManyPending` con `data = {max_pending, pending}`. No se escribe journal, no avanza el estado ni el ledger.
- Se comprueba antes de validar `params.request`: es el camino barato para rechazar una inundación. Un cliente legítimo se recupera con `report` (libera hueco) o esperando al TTL.
- El replay NO descarta pendientes: bajar `max_pending` por debajo de los pendientes retenidos solo bloquea nuevos `select` hasta que se drenen; nunca pierde decisiones en vuelo (coherente con ADR-0016).

## Porqué `-32007`
`-32004` (sin brazo) y `-32005` (no listo) describen otras condiciones; `-32006` es solo de arranque. Un código propio permite al cliente distinguir «reporta antes de pedir más» de «no hay brazo».

## Rechazado
- Valor por defecto si falta la clave: es un camino de compatibilidad; las configs existentes deben añadir la clave (error explícito `bandit.max_pending`).
- Expulsar el pendiente más antiguo: perdería una decisión cuyo `report` puede llegar, y rompería la propiedad de que toda decisión registrada se aprende o expira por TTL.
