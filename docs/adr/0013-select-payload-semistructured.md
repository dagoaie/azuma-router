# ADR-0013 — `select` recibe una petición semi-estructurada; el prompt nunca viaja por stdio

**Estado:** Aceptado (Grill R2 · Q13) · 2026-09-14

## Contexto
El Featurizer podría recibir el texto/diff crudo (el router tokeniza y clasifica) o conteos y enums ya calculados por Convoy.

## Decisión
- `params.request` de `select` contiene solo campos escalares y enums: `task_kind`, `input_tokens`, `expected_output_tokens?`, `has_code`, `has_tools`, `has_images`, `agent_depth`, `diff_lines?`, `priority`. Opcional `allowed_arms` (subconjunto de `arm_id`) y `verbose` (default `true`).
- Convoy es responsable de contar tokens y clasificar la tarea (es quien construye el prompt).
- Respuesta: `decision_id`, `arm`, `propensity`, `diagnostics` completo (scores, exploit, bonus, shadow_price, policy, utilization) salvo `verbose: false`.

## Porqué
Cero strings de tamaño arbitrario por el pipe; latencia acotada; el router no arrastra tokenizadores ni clasificadores NLP (follow-up F4). Mantiene la frontera de responsabilidad: Convoy conoce la tarea, azuma conoce los brazos.

## Rechazado / diferido
- Texto/diff crudo por stdio: rechazado.
- Clasificación del `task_kind` dentro del router: follow-up F4.
