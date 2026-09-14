# ADR-0011 — `report` idempotente por `decision_id`; decisiones sin report se descartan tras TTL sin imputar

**Estado:** Aceptado (Grill R1 · Q11) · 2026-09-14

## Contexto
La recompensa llega asíncronamente tras ejecutar la llamada LLM. Agentes que crashean o expiran nunca reportan; los reintentos pueden duplicar reports.

## Decisión
- Cada `select` genera `decision_id` (UUIDv7, ordenable por tiempo) y guarda la decisión pendiente (`x_t`, `arm`, `propensity`, `ts`) en el `PendingStore`.
- `report(decision_id, quality, cost_usd, latency_s)`: primera vez → calcula $r$ (ADR-0007), aplica `update`, mueve el registro al `DecisionLog` con `r`. Segunda vez → `warning` en stderr, respuesta OK sin efecto. `decision_id` desconocido/expirado → error JSON-RPC específico.
- Orden libre entre decisiones.
- TTL configurable (default 3600 s). Al expirar: la decisión se escribe al `DecisionLog` con `reward = null` y **no** se aplica `update`. El consumo de cuota ya se registró en el ledger en el `select` (estimación) y se corrige con el `report` real si llega.

## Porqué
Imputar recompensa 0 o la media sesga $\hat\theta_a$ (castiga/premia al brazo por fallos del agente, no del modelo). Registrar la decisión expirada con `reward = null` preserva la trazabilidad y permite analizar tasa de abandono por brazo.

## Rechazado / diferido
- Imputación de 0 o media: rechazado.
- Reports parciales (solo latencia sin calidad): rechazado v1; todo o nada.
