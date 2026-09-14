# ADR-0004 — Escalado por bloque a $[0,1]$, cota $L = \sqrt{d}$, NaN/inf → excepción

**Estado:** Aceptado (Grill R1 · Q4) · 2026-09-14

## Contexto
Las cotas UCB asumen $\|x_t\| \le L$. Normalizar a $\|x\|_2 = 1$ destruye la magnitud relativa de los bloques; escalar por bloque mantiene la semántica.

## Decisión
- Cada bloque del Featurizer se escala independientemente a $[0,1]$ con rangos **declarados y versionados** (`featurizer_version`).
- Bias fijo $x_0 = 1$. Cota declarada $L = \sqrt{d}$; $\alpha$ se calibra con OPE para esa escala.
- El Featurizer (fuera del JIT) valida: sin NaN/inf, todos los componentes en $[0,1]$, `len == d`. Violación → `InvalidContextError` (fail-fast).

## Porqué
Explícito sobre clever: rangos declarados son auditables; `nan_to_num` sería un silent fallback (prohibido por política hard-cut).

## Rechazado / diferido
- Normalización L2 global: rechazado.
- Normalización online (running mean/std): cambia la semántica de $x$ con el tiempo y rompe la estacionariedad que asume LinUCB. Rechazado.
