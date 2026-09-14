# ADR-0007 — Recompensa Q-C-L con calidad externa, coste monetario real (0 en suscripción)

**Estado:** Aceptado (Grill R1 · Q7) · 2026-09-14

## Contexto
La recompensa escalar $r_t \in [0,1]$ debe resumir calidad, coste y latencia. La calidad no es observable por el router.

## Decisión
- $r = \mathrm{clip}\big(w_q\, q - w_c\, \tilde c - w_l\, \tilde l,\ 0,\ 1\big)$ con $w_q + w_c + w_l$ documentados en `StaticConfig`.
- $q \in [0,1]$ la aporta el llamador (Convoy: LLM-judge, tests pasados, aceptación humana). El router **nunca** la infiere.
- $\tilde c = \min(\text{coste\_usd} / \text{cost\_budget\_usd}, 1)$; para brazos `SUBSCRIPTION`, `coste_usd = 0` — la cuota es restricción (ADR-0006), no recompensa.
- $\tilde l = \min(\text{latencia\_total\_s} / \text{latency\_budget\_s}, 1)$.
- Pesos y budgets se calibran offline con OPE (IPS/DR).

## Porqué
Separar restricción (cuota) de recompensa (calidad/coste/latencia) evita doble penalización y mantiene $\hat\theta_a$ interpretable. Calidad externa evita acoplar el router a un judge concreto.

## Rechazado / diferido
- Coste imputado para suscripción: rechazado (doble contabilidad con el precio sombra).
- TTFT o tokens/s como latencia: rechazado para la v1; latencia total es lo que ve el agente.
- Reward delayed a nivel de tarea multi-llamada: follow-up F8.
