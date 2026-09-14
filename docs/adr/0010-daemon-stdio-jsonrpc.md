# ADR-0010 — Daemon stdio JSON-RPC 2.0 (estilo MCP) como protocolo canónico

**Estado:** Aceptado (Grill R1 · Q10) · 2026-09-14 · **Decisión arquitectónica definitiva**

## Contexto
Convoy (TypeScript) necesita invocar `select` y `report` con latencia mínima. Alternativas: UDS + JSON-lines, gRPC, HTTP local, stdio JSON-RPC.

## Decisión
- El daemon se lanza como **proceso hijo** de Convoy (`python -m azuma_router.daemon`) y habla **JSON-RPC 2.0 sobre stdio**, un mensaje por línea (`\n`-delimited), UTF-8. `stderr` reservado a logs estructurados.
- Métodos: `select`, `report`, `snapshot`, `health`; notificaciones: `ready` (tras warm-up JIT). Schema exacto se cierra en Grill R2.
- Un único protocolo. Sin UDS/HTTP paralelos.

## Porqué
Convoy ya conoce el patrón MCP (lanzar hijo, hablar JSON-RPC por stdio): cero puertos, cero auth, cero configuración de red, funciona en los tres SO, y el ciclo de vida del daemon queda ligado al del orquestador. La latencia de un round-trip por pipe (~50–100 µs) es compatible con el SLO.

## Rechazado / diferido
- gRPC: dependencias pesadas en ambos lados. Rechazado.
- HTTP local: más latencia y superficie de red innecesaria. Rechazado.
- UDS: no portable a Windows. **Follow-up F1** si algún consumidor no puede lanzar procesos hijo.
