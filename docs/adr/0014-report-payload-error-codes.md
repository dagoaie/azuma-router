# ADR-0014 — `report` con `quality` obligatoria; duplicado = `applied:false`; códigos de error −32001…−32006

**Estado:** Aceptado (Grill R2 · Q14) · 2026-09-14

## Decisión
- `params` de `report`: `decision_id`, `quality ∈ [0,1]` (**obligatoria**), `cost_usd ≥ 0`, `latency_s ≥ 0`, `actual_input_tokens ≥ 0`, `actual_output_tokens ≥ 0`. Los tokens reales corrigen la estimación de consumo anotada en el ledger en `select`.
- Duplicado (mismo `decision_id` ya aplicado): `result: {"applied": false, "reason": "duplicate"}`, sin error, `warning` en stderr.
- Errores de aplicación JSON-RPC: `-32001 InvalidContext`, `-32002 UnknownDecision`, `-32003 DecisionExpired`, `-32004 NoArmAvailable`, `-32005 NotReady`, `-32006 IncompatibleSnapshot`. Los estándar (`-32700` parse, `-32600` invalid request, `-32601` method not found, `-32602` invalid params) se usan tal cual.

## Porqué
Sin `quality` no hay recompensa válida (ADR-0007/0011: todo o nada). Idempotencia silenciosa hace que un reintento del cliente sea seguro. Códigos fijos permiten que el plugin TS los mapee a tipos.

## Rechazado / diferido
- `report` parcial (solo coste/latencia): rechazado v1.
- Duplicado como error: rechazado (rompe reintentos).
