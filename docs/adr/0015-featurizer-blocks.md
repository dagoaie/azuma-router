# ADR-0015 — Bloques del Featurizer v1 ($d = 32$), enum cerrado de `task_kind`, término de interacción

**Estado:** Aceptado (Grill R2 · Q15) · 2026-09-14

## Decisión
`featurizer_version = 1`. Índices y transformaciones (todas en $[0,1]$):

| Idx | Bloque | Transformación |
|---|---|---|
| 0 | bias | 1.0 |
| 1 | `input_tokens` | $\min(\log_{10}(1+n)/6,\,1)$ |
| 2 | `expected_output_tokens` | idem; 0 si ausente |
| 3 | `diff_lines` | $\min(\log_{10}(1+n)/4,\,1)$; 0 si ausente |
| 4–15 | `task_kind` one-hot | `code_edit, code_review, code_gen, test_gen, planning, summarize, qa, extract, classify, agentic_tool_loop, chat, other` (en este orden) |
| 16–18 | `has_code`, `has_tools`, `has_images` | 0/1 |
| 19 | `agent_depth` | $\min(n,8)/8$ |
| 20–22 | `priority` one-hot | `low, normal, high` |
| 23 | interacción | $x_{16} \cdot x_1$ |
| 24–31 | padding | 0 |

- `task_kind` fuera del enum ⇒ `InvalidContextError` (nunca "other" silencioso).
- Cota $L = \sqrt{32}$; cualquier componente fuera de $[0,1]$, NaN o inf ⇒ `InvalidContextError`.

## Porqué
LinUCB es lineal: sin el producto (23), "prompt largo con código" no es separable de "prompt largo" + "con código". Enum cerrado = fail-fast ante cambios de Convoy no coordinados.

## Rechazado / diferido
- Embeddings: F4. Cambiar cualquier bloque ⇒ `featurizer_version = 2` y snapshot incompatible (ADR-0009).
