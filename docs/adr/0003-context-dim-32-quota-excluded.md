# ADR-0003 — $d = 32$; la utilización de cuota **no** forma parte de $x_t$

**Estado:** Aceptado (Grill R1 · Q3) · 2026-09-14 · **Decisión arquitectónica definitiva**

## Contexto
El contexto $x_t$ podría incluir el estado del router (utilización de cuota por proveedor) además de la descripción de la petición. Hacerlo mezcla *restricción* con *aprendizaje*.

## Decisión
- $d = 32$ fijo (bloques útiles ≈ 21, resto padding a cero; el detalle de bloques se cierra en Grill R2).
- $x_t$ describe **exclusivamente la petición**: nada del estado del router, del ledger ni de las cuotas.
- La cuota actúa solo a través del precio sombra en la penalización del score (ADR-0006).

## Porqué
Si $u_{a,W}$ entrara en $x_t$, el bandido "aprendería" que un brazo es malo cuando está saturado y necesitaría rondas para desaprenderlo al liberarse la ventana. La separación mantiene $\hat\theta_a$ como estimador puro de calidad/coste y hace que la respuesta a la cuota sea instantánea y explicable.

## Rechazado / diferido
- Utilización en $x_t$: rechazado.
- $d$ dinámico o > 32: recompilación; rechazado. Cambiar $d$ = nueva versión de estado.
- Embeddings semánticos en $x_t$: follow-up F4 (viola presupuesto de latencia).
