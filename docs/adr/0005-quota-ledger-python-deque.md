# ADR-0005 — `QuotaLedger` en Python con deque exacto; al JIT solo entra `utilization[K,W]`

**Estado:** Aceptado (Grill R1 · Q5) · 2026-09-14

## Contexto
Las ventanas rodantes pueden representarse con deques exactos (Python, forma dinámica) o buckets circulares como array JAX (forma fija, aproximado).

## Decisión
- `QuotaLedger` es un objeto Python fuera del JIT: por brazo y ventana, un `deque[(timestamp_s, units)]` podado por el lado antiguo en cada consulta.
- `QuotaWindow` tiene dos variantes explícitas: `RollingWindow(duration_s, capacity)` y `FixedResetWindow(period, reset_anchor, capacity)` (semana con reset en día fijo).
- El ledger produce `QuotaView.utilization: f32[K_max, W_max]` ($u_{a,W}$, 0 en slots sin ventana) justo antes de cada `select`. `W_max` es forma estática.
- El ledger se persiste con el snapshot (ADR-0009).

## Porqué
Exacto, trivial de testear con reloj inyectado, sin aproximación por resolución de bucket. La reducción a un escalar por ventana es $O(\text{eventos expirados})$ amortizado y ocurre fuera de la ruta JIT.

## Rechazado / diferido
- Buckets en JAX: rechazado (aproximación innecesaria, complejidad de reloj dentro del JIT).
- Ventana única por brazo: rechazado; los proveedores combinan corta + semanal.
