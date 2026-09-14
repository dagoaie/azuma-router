# ADR-0006 — Barrera logarítmica, $\mu$ por brazo, unidad abstracta `units`

**Estado:** Aceptado (Grill R1 · Q6) · 2026-09-14

## Contexto
El precio sombra debe crecer sin cota cuando la utilización $u \to 1$ y ser suave lejos de la saturación. Los proveedores no publican la fórmula exacta de consumo de cuota.

## Decisión
- $\phi(u) = -\mu_a \ln(1 - u)$ con $u$ recortado a $[0, 1-10^{-6}]$ dentro del JIT; $u \ge 1$ ⇒ `mask[a] = False`.
- $\lambda_a(t) = \max_W \phi(u_{a,W})$; penalización $= \lambda_a \cdot c_a(x_t)$ con $c_a$ = consumo estimado en `units`.
- `barrier_mu` por brazo en `StaticConfig` (default global aplicado a los brazos que no lo declaran). Brazos `PER_TOKEN` tienen $\mu = 0$ y ninguna ventana.
- Unidad de cuota: `units` abstracta; cada brazo `SUBSCRIPTION` declara `units_per_request` y `units_per_1k_tokens` en su config.

## Porqué
La barrera logarítmica es la barrera interior clásica (convexa, $C^\infty$, gradiente $\mu/(1-u)$): penalización despreciable hasta $u \approx 0.7$ y explosiva cerca de 1, que es el perfil deseado. La unidad abstracta reconoce que la fórmula real del proveedor es desconocida y la hace configurable en vez de fingir precisión.

## Rechazado / diferido
- Potencia inversa $\mu/(1-u)^k$: rechazado (segundo hiperparámetro $k$ sin beneficio claro).
- $\mu$ aprendido: follow-up F7.
