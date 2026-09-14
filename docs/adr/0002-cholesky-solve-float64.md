# ADR-0002 — Solver directo por Cholesky en float64; no se mantiene $A^{-1}$

**Estado:** Aceptado (Grill R1 · Q2) · 2026-09-14

## Contexto
LinUCB necesita $\hat\theta_a = A_a^{-1} b_a$ y $x^\top A_a^{-1} x$ por brazo en cada `select`. Alternativas: (a) resolver desde $A$ en cada llamada; (b) mantener $A^{-1}$ con Sherman–Morrison, $O(d^2)$; (c) mantener factor Cholesky con rank-1 update propio.

## Decisión
- El estado guarda **solo $A_a$** (SPD por construcción con `ridge_lambda > 0`).
- `select` calcula `L = cho_factor(A)` y `cho_solve(L, ·)` vía `vmap`, en **float64** (`jax.config.update("jax_enable_x64", True)` en el módulo de entrada; test de arranque lo verifica).
- `update` hace `A.at[arm].add(outer(x, x))`, `b.at[arm].add(r * x)`.
- Test de propiedad: tras $10^5$ updates aleatorios, $A$ simétrica bitwise y $\|A\hat\theta - b\|_\infty < 10^{-9}$.

## Porqué
Con $K \le 32$, $d = 32$, el coste $O(Kd^3) \approx 1$ M flops es irrelevante frente al SLO < 1 ms. Se elimina toda la clase de bugs de deriva numérica / pérdida de simetría de (b) y el código propio de `cholupdate` de (c), que no existe en `jax.scipy`.

## Rechazado / diferido
- Sherman–Morrison: rechazado por deriva acumulada.
- Cholesky rank-1 update: **follow-up diferido** solo si un benchmark demuestra necesidad de $d \ge 256$.
- float32 con re-factorización periódica: rechazado; introduce un segundo path.
