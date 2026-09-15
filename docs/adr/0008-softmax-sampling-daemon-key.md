# ADR-0008 — Muestreo softmax con temperatura y suelo $\varepsilon/K$; PRNG gestionada por el daemon

**Estado:** Aceptado (Grill R1 · Q8) · 2026-09-14

## Contexto
LinUCB con argmax es determinista → propensity ∈ {0,1} → IPS/DR inservibles. Se necesita una política de muestreo con propensity registrable y no degenerada.

## Decisión
- $\pi(a \mid x) = (1-\varepsilon)\,\mathrm{softmax}(\text{score}/\tau)_a + \varepsilon / K_{\text{activos}}$ sobre brazos con `mask = True`; `tau ≈ 0.05` (escala de score en $[0,1]$), `epsilon_floor ≈ 0.02`, ambos en `StaticConfig`.
- `select` devuelve `arm`, `propensity = π(arm|x)` y el vector completo `π` en `diagnostics`.
- `rng_key` vive en `BanditState`; `select` hace `split` y devuelve el estado con la nueva key. El snapshot persiste la key y su `selection_sequence`; cada `select` posterior persiste en el WAL la secuencia contigua y la key resultante. El replay instala la última transición posterior al snapshot → reproducibilidad exacta tras crash.

## Porqué
Softmax da propensities continuas y > 0 para todos los brazos activos; el suelo $\varepsilon/K$ acota los pesos IPS a $\le K/\varepsilon$ y por tanto la varianza. Key en el estado mantiene `select` pura y reproducible.

## Rechazado / diferido
- ε-greedy puro: rechazado (propensity casi binaria, varianza alta).
- Argmax sin OPE: rechazado.
- Thompson Sampling lineal: follow-up F3.
- Key aportada por el llamador: rechazado (el plugin TS no debería conocer PRNG de JAX).
