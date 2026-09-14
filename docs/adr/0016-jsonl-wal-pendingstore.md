# ADR-0016 — `DecisionLog` JSONL como WAL; `PendingStore` en memoria reconstruido por replay; barrido TTL amortizado

**Estado:** Aceptado (Grill R2 · Q16) · 2026-09-14 · **Decisión arquitectónica definitiva**

## Decisión
- Cada `select` escribe **antes de responder** una línea `{"phase":"select", …}` (append + flush) y cada `report` aplicado una línea `{"phase":"report", …}`; expiraciones escriben `{"phase":"expire", …}`.
- `PendingStore` = `dict[decision_id, Pending]` en memoria + orden de inserción (UUIDv7 es cronológico) para el barrido TTL. Se reconstruye al arrancar con replay del JSONL desde el `state_version` del snapshot: `select` sin `report`/`expire` posterior ⇒ pendiente.
- Barrido de TTL amortizado: al inicio de cada `select` y en cada `snapshot`. Sin hilos ni timers; el daemon es secuencial sobre stdio.
- Rotación por tamaño (`decision_log_max_bytes`, default 256 MiB); el replay lee el fichero activo y el anterior.

## Porqué
El JSONL ya existe para OPE; convertirlo en WAL cuesta un campo `phase`. El OPE prefiere las fases separadas (la decisión con su propensity queda registrada aunque el report nunca llegue). Single-thread elimina toda concurrencia.

## Rechazado / diferido
- `PendingStore` solo en memoria: rechazado (pierde decisiones en vuelo en un crash).
- SQLite: rechazado (segunda fuente de verdad junto al JSONL).
