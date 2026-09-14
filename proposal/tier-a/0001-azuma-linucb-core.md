# Ficha de Propuesta 0001 — azuma-router: núcleo LinUCB en JAX con shadow pricing

| Campo | Valor |
|---|---|
| Tier | A (arquitectura fundacional) |
| Estado | CERRADA (Fase 0) — Grill R1 y R2 aprobados 2026-09-14; especificación en `specs/core/` |
| Autor | David González |
| Fecha | 2026-09-14 |
| Vocabulario | `CONTEXT.md` (fuente de verdad de términos) |
| ADRs | `docs/adr/0001` … `0018` (ver `docs/adr/README.md`) |

> Convención: cada decisión cita su ADR como `(ADR-000N)`. Contratos exactos en `specs/core/spec.md`.

---

## §1 Resumen

azuma-router es un motor de enrutamiento de peticiones a LLMs/SLMs que, en cada llamada, elige el modelo (brazo) que maximiza una recompensa compuesta calidad–coste–latencia, aprendiendo online mediante **LinUCB disjunto** y respetando **cuotas de suscripción con ventanas rodantes** mediante **precios sombra** con barrera convexa. El núcleo es un conjunto de funciones puras JAX (`select`, `update`) compiladas con `jax.jit` y vectorizadas con `jax.vmap` sobre el eje de brazos, con estado inmutable (PyTree) y presupuesto de latencia **< 1 ms p99** en CPU. Esta ficha cubre exclusivamente el núcleo matemático + estado + snapshot; la integración con Convoy se difiere a una ficha posterior.

## §2 Evidencia y motivación técnica

**Hechos durables (verificables en la fuente citada):**

1. **Dispersión de precios entre modelos de 1–2 órdenes de magnitud.** Los listados públicos de OpenRouter muestran, para una misma tarea, modelos con precio por millón de tokens de salida que difieren en más de 50× entre la gama frontier y los SLM abiertos (comprobar valores concretos en `openrouter.ai/models` en el momento del MR; los precios cambian, la dispersión no).
2. **Un router aprendido reduce coste sin perder calidad.** RouteLLM (Ong et al., 2024, arXiv:2406.18665) reporta reducciones de coste de hasta ~85 % en MT-Bench manteniendo ~95 % de la calidad del modelo fuerte, usando routers entrenados sobre datos de preferencia. Evidencia de que la decisión "qué modelo" es aprendible a partir del contexto de la petición.
3. **Las suscripciones planas tienen cuotas rodantes, no precio marginal.** Codex Pro / Claude Max exponen límites en ventanas de ~5 h más techos semanales (valores exactos: parámetros de configuración, cambian por plan). Un router que solo mira `$ / token` ve estos brazos como coste 0 y los satura en minutos, forzando un hard-cutoff justo cuando más se necesitan.
4. **Los pipelines estáticos son frágiles.** En orquestadores de agentes (Convoy incluido) el modelo suele fijarse en configuración por rol/agente. Cada cambio de precio, cuota o disponibilidad exige un cambio manual; no existe retroalimentación de la calidad observada hacia la decisión.
5. **La latencia de decisión debe ser despreciable frente a la de inferencia.** Una llamada LLM típica dura 500 ms–30 s; un router que añade >10 ms en cadenas de decenas de llamadas por tarea de agente es un coste apreciable. JAX compilado en CPU ejecuta un `select` de `(K≤32, d≤64)` en el orden de decenas de µs (a medir en §10 Verification).

**Por qué bandidos contextuales y no clasificación supervisada:** un clasificador (RouteLLM) requiere dataset etiquetado offline y no se adapta a cambios de precio/cuota/calidad en producción. Un bandido aprende **online**, con feedback parcial (solo observamos el brazo elegido), y su regret está acotado (Azuma–Hoeffding → $\tilde O(\sqrt{dT})$).

**Por qué JAX y no NumPy/PyTorch:** `jit` + `vmap` dan un único kernel XLA sin overhead de intérprete por brazo; el estado inmutable es idiomático (no un añadido); y el mismo código corre en CPU hoy y en acelerador si $K \cdot d^2$ crece.

## §3 Problema y Valor

**Problema.** Un orquestador de agentes emite miles de llamadas heterogéneas; hoy la asignación petición→modelo es estática, ignora cuotas rodantes y no aprende de la calidad observada. Resultado: sobrecoste (frontier para tareas triviales), saturación de cuotas de suscripción, y ausencia de trazabilidad de por qué se eligió un modelo.

**Valor.**
- Coste: reducción medible de $/tarea con calidad ≥ umbral (métrica: regret offline sobre `DecisionLog` y coste real acumulado).
- Cuotas: utilización de ventanas rodantes suave, sin hard-cutoffs (métrica: fracción de tiempo con $u_{a,W} \ge 0.95$; objetivo ~0).
- Explicabilidad: cada decisión lleva scores UCB, precio sombra y propensity → auditable y evaluable off-policy.
- Latencia: < 1 ms por decisión.

## §4 Alcance (In-Scope)

1. `BanditState` como `NamedTuple` de arrays + `StaticConfig` hashable (ADR-0001); $K_{\max}$ brazos con máscara, $d = 32$ (ADR-0003).
2. `select(state, x, quota_view) -> (state', arm, propensity, diagnostics)` compilado; vmap sobre brazos; Cholesky float64 (ADR-0002); scores UCB + penalización por precio sombra; muestreo softmax con suelo y `rng_key` en el estado (ADR-0008).
3. `update(state, x, arm, reward) -> state` compilado; rank-1 sobre $A_a$, $b_a$ (ADR-0002).
4. `QuotaLedger` Python con deque exacto, `RollingWindow` + `FixedResetWindow` (ADR-0005); barrera logarítmica con $\mu$ por brazo y `units` abstractas (ADR-0006).
5. Recompensa Q-C-L con calidad externa y coste 0 para suscripción (ADR-0007).
6. Snapshot safetensors + JSON con rename atómico, `state_version`, rechazo fail-fast (ADR-0009).
7. `DecisionLog` JSONL append-only + `PendingStore` con TTL y `report` idempotente (ADR-0011). JSONL como WAL con fases `select`/`report`/`expire`, `PendingStore` reconstruido por replay (ADR-0016).
8. Evaluador OPE offline (IPS y DR) sobre `DecisionLog` para calibrar $\alpha$, $\tau$, pesos de recompensa.
9. Featurizer determinista, escalado por bloque, sin dependencias NLP (ADR-0004). Bloques v1 con enum cerrado de `task_kind` e interacción código×tokens (ADR-0015).
10. Daemon stdio JSON-RPC 2.0: métodos `select`/`report`/`snapshot`/`health`, notificación `ready` (ADR-0010). Petición semi-estructurada, `quality` obligatoria en `report`, códigos −32001…−32006 (ADR-0013/0014). Configuración `azuma.toml` (ADR-0017).
11. Suite de tests: propiedades numéricas (SPD, monotonía del bonus), invariantes de forma (recompilaciones = 1), latencia (benchmark p99), determinismo con `seed` fija. Python ≥ 3.12, Apache-2.0 (ADR-0012).

## §5 Fuera de alcance y Follow-ups diferidos explícitos

| # | Diferido | Motivo | Criterio para reabrir |
|---|---|---|---|
| F1 | Integración real con Convoy (plugin TS, transporte, auth) | Ficha 0002; requiere contrato del daemon cerrado (Q10/Q11) | 0001 mergeada y contrato estable |
| F2 | LinUCB *hybrid* (features compartidas entre brazos) | Rompe la vmap trivial; beneficio no demostrado para K ≤ 32 | Evidencia OPE de cold-start lento en brazos nuevos |
| F3 | Thompson Sampling lineal como política alternativa | Misma estructura de estado; añadir después sin tocar el núcleo | Si la propensity de LinUCB+ε resulta mala para OPE |
| F4 | Featurizer con embeddings semánticos (modelo de embedding en la ruta caliente) | Viola presupuesto <1 ms; requiere caché | Regret offline insuficiente con features simbólicas |
| F5 | Ejecución en GPU/TPU | CPU basta para $K\cdot d^2 \le 131{,}072$ flops-equivalentes | $K$ o $d$ crecen 10× |
| F6 | Multi-tenant (estados por usuario/equipo) | Un estado global es suficiente para un orquestador | Convoy exige aislamiento por tenant |
| F7 | Aprendizaje del precio sombra (λ aprendido vs. calculado) | Barrera determinista es explicable y suficiente | Evidencia de sobre/infra-utilización sistemática |
| F8 | Reward delayed / atribución a nivel de tarea multi-llamada | Requiere semántica de Convoy | Al abordar F1 |
| F9 | Persistencia distribuida / réplicas del daemon | Un daemon por host | Convoy multi-host |
| F10 | Migraciones de snapshot entre versiones de estado | Política hard-cut: se rechaza y se re-entrena | Nunca, salvo decisión explícita |
| F11 | Añadir/quitar brazos en caliente conservando aprendizaje | Requiere remapeo explícito de índices (`azuma migrate-arms`); contradice el rechazo estricto v1 (ADR-0017) | Rotación frecuente de modelos en Convoy |

## §6 Diseño propuesto

### 6.1 Arquitectura funcional (capas)

```
┌──────────────────────────────── proceso daemon (Python) ────────────────────────────────┐
│  stdio JSON-RPC 2.0  →  Handler (I/O, validación)  →  Featurizer  →  ┌───── JIT boundary ─────┐ │
│                                                                    │ select_jit / update_jit │ │
│  QuotaLedger (Python, reloj)  → QuotaView (arrays)  ───────────►   │  vmap sobre eje K       │ │
│  DecisionLog (append-only)  ◄─ diagnostics ────────────────────    └────────────────────────┘ │
│  Snapshotter (safetensors+JSON, rename atómico)  ◄─ BanditState (PyTree) + QuotaLedger        │
│  PendingStore (TTL 3600 s)  ◄─ decision_id ─── report(decision_id, q, cost, latency) → update │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

Regla de oro: **dentro del JIT boundary solo hay arrays de forma fija y funciones puras**. Reloj, I/O, configuración y ledger viven fuera y entran como arrays.

### 6.2 Estado inmutable

```
BanditState (PyTree, frozen):
  A        : f64[K_max, d, d]   # solo A; nunca A^{-1} (ADR-0002)
  b        : f64[K_max, d]
  n_pulls  : i32[K_max]
  mask     : bool[K_max]
  rng_key  : u32[2]             # avanzada por select (ADR-0008)
StaticConfig (hashable, argumento static_argnums):
  K_max=32, d=32, W_max, ridge_lambda=1.0, alpha, tau=0.05, epsilon_floor=0.02,
  barrier_mu: tuple[float, ...] (por brazo), w_q, w_c, w_l, cost_budget_usd, latency_budget_s
QuotaView (PyTree):
  utilization : f32[K_max, W_max]   # u_{a,W}, 0 si el brazo no tiene esa ventana
  est_cost    : f32[K_max]          # consumo estimado c_a(x_t)
```

dtype de $A$, $b$: **float64** (`jax_enable_x64`, ADR-0002); `QuotaView` en float32. `state_version` vive en `state.json`, no en el PyTree.

### 6.3 `select` (vectorizado)

Por brazo, con `vmap` sobre el eje 0:
1. `L_a = cho_factor(A_a)`; $\hat\theta_a$ = `cho_solve(L_a, b_a)`; $x^\top A_a^{-1}x = \|L_a^{-1}x\|^2$ (ADR-0002)
2. $\text{exploit} = x^\top\hat\theta_a$; $\text{bonus} = \alpha\sqrt{x^\top A_a^{-1} x}$
3. $\lambda_a = \max_W -\mu_a\ln(1-u_{a,W})$; $\text{penalty} = \lambda_a \cdot c_a$ (ADR-0006)
4. $\text{score}_a = \text{exploit} + \text{bonus} - \text{penalty}$; `where(mask, score, -inf)`
5. $\pi = (1-\varepsilon)\,\mathrm{softmax}(\text{score}/\tau) + \varepsilon/K_{\text{act}}$; `arm ~ π` con `rng_key` (split, key nueva en `state'`); `propensity = π[arm]` (ADR-0008)
6. `diagnostics` = todos los intermedios (para `DecisionLog`).

### 6.4 `update`

$A_a \leftarrow A_a + xx^\top$, $b_a \leftarrow b_a + r x$ solo para `arm`, expresado como `A.at[arm].add(...)` (índice dinámico permitido, forma fija). No se mantiene ningún factor entre llamadas (ADR-0002).

### 6.5 Shadow pricing
Barrera logarítmica con `barrier_mu` por brazo (ADR-0006); $u$ producida por el `QuotaLedger` Python de deque exacto (ADR-0005). Ventana más restrictiva domina (`max` sobre eje W). El consumo se registra en el ledger como **estimación** en `select` y se corrige con el `report` real (ADR-0011).

### 6.6 Snapshot
`state.safetensors` + `state.json` (config, ledger, `state_version`, `featurizer_version`, `seed`) escritos en `snapshot.tmp/`, fsync, rename atómico a `snapshot/`. Cada `max(100 updates, 30 s)` y en shutdown. Formas/schema incompatibles ⇒ `IncompatibleSnapshotError` (ADR-0009).

## §7 Objeciones esperadas y mitigación

| Objeción | Mitigación |
|---|---|
| **Cold-start de XLA/JIT** (primer `select` tarda 0.5–3 s) | Warm-up en arranque del daemon con formas reales; healthcheck no responde `ready` hasta completar. Test que verifica exactamente 1 compilación por función en toda la suite. AOT (`jax.export`) como opción si el warm-up supera 3 s. |
| **Estabilidad numérica**: mantener $A^{-1}$ vía Sherman–Morrison acumula error; factorizar $A$ cada vez es $O(d^3)$ | ADR-0002: no se invierte; Cholesky sobre $A$ SPD en float64 en cada `select` ($O(Kd^3) \approx 1$ M flops, irrelevante). Test de propiedad: $\|A\hat\theta - b\|_\infty < 10^{-9}$ y simetría bitwise tras $10^5$ updates. |
| **Contextos no acotados** rompen la teoría y el condicionamiento | Featurizer normaliza a $\|x\|\le1$ y lo asegura con assert fuera del JIT; test de propiedad. |
| **Propensity degenerada** hace inútil el OPE | Softmax con suelo $\varepsilon/K$ (ADR-0008); pesos IPS acotados a $K/\varepsilon$; propensity registrada en cada decisión. |
| **La recompensa llega tarde** (calidad se conoce tras la ejecución) | `select`/`report` desacoplados por `decision_id` (UUIDv7) en `PendingStore`; TTL 3600 s sin imputación; `report` idempotente (ADR-0011). Reward delayed multi-llamada diferido (F8). |
| **Precio sombra mal calibrado** (`barrier_mu`) → infra-utilización de la suscripción | Métrica de utilización media por ventana en diagnostics; `barrier_mu` ajustable en caliente (es valor, no forma). |
| **float64 en JAX** es opt-in global y puede sorprender | `jax_enable_x64` se fija en el módulo de entrada del daemon y se verifica en un test de arranque. |
| **Un solo brazo domina y los demás nunca aprenden** | Es exactamente lo que el bonus UCB previene; test de simulación con brazos sintéticos verifica que todos reciben pulls y regret sublineal. |
| **"Es una matriz $d\times d$ por brazo; ¿por qué no NumPy?"** | Benchmark en §10 compara NumPy vs JAX jit para $(K,d)$ objetivo; JAX se justifica por vmap + inmutabilidad + ausencia de intérprete por brazo, no solo por velocidad. |

## §8 Dependencias técnicas

Hechos verificados hoy (2026-09-14) en la máquina de desarrollo:

| Dependencia | Versión | Nota |
|---|---|---|
| Python | `>=3.12`; CI en 3.12 y 3.13 (ADR-0012). 3.14.7 disponible localmente pero no es objetivo | uv gestiona la versión |
| uv | 0.11.19 | Único gestor de paquetes y entornos |
| jax / jaxlib | 0.11.1 (última en PyPI) | CPU wheels arm64 macOS y x86_64 Linux |
| numpy | la que exija jax | Solo en Featurizer / tests |
| Serialización snapshot | `safetensors` + JSON stdlib (ADR-0009) | |
| Transporte daemon | stdlib (`sys.stdin`/`stdout`, `json`, `uuid`) — JSON-RPC 2.0 sin librería externa (ADR-0010). Validación manual contra dataclasses tipadas, sin `pydantic`/`jsonschema` | |
| Test / calidad | `pytest`, `hypothesis` (property tests), `pytest-benchmark`, `ruff`, `mypy --strict` | |

Sin dependencias de red en la ruta caliente. Sin PyTorch, sin scikit-learn.

## §9 Tamaño y complejidad estimada

| Bloque | LOC estimadas (src) | Tests | Riesgo |
|---|---|---|---|
| `state` (PyTree, config, validación) | ~120 | ~150 | bajo |
| `linucb` (`select`, `update`, solver) | ~180 | ~350 (propiedades + numérica) | medio (ADR-0002 elimina la deriva) |
| `quota` (ledger, ventanas, barrera) | ~200 | ~250 | medio (reloj inyectado, dos tipos de ventana) |
| `reward` (composición Q-C-L) | ~60 | ~100 | bajo |
| `featurizer` | ~100 | ~120 | bajo |
| `snapshot` | ~120 | ~150 | medio (atomicidad en disco) |
| `ope` (IPS, DR) | ~120 | ~150 | bajo |
| `daemon` (JSON-RPC stdio + PendingStore + TTL) | ~250 | ~250 | medio (replay del WAL) |
| **Total** | **~1 150** | **~1 550** | |

Complejidad algorítmica por decisión: $O(K d^3)$ con Cholesky directo — para $K=32, d=32$ ≈ 1 M flops → decenas de µs en CPU.

Estimación de esfuerzo: 3–4 slices verticales (ver §11), cada uno mergeable de forma independiente.

## §10 Borrador de PR (inglés)

**Title:** `feat(core): disjoint LinUCB router in JAX with quota-aware shadow pricing`

```markdown
## Why

Static request→model assignment in agent orchestrators ignores three things at once:
observed quality, rolling subscription quotas (5h windows / weekly caps) and live
per-token prices. This PR introduces the azuma-router core: a disjoint LinUCB
contextual bandit compiled with `jax.jit`/`jax.vmap`, with a convex-barrier shadow
price that keeps flat-rate arms below quota exhaustion. Regret is sublinear
(Azuma–Hoeffding, Õ(√(dT))); decision latency budget is < 1 ms p99 on CPU.

## Summary

- `BanditState` immutable PyTree (`A: [K,d,d]`, `b: [K,d]`, `mask`, `n_pulls`), static
  `(K_max, d)` shapes with arm masking — exactly one XLA compilation per function.
- `select`: vmapped UCB scores + shadow-price penalty + non-degenerate sampling policy;
  returns arm, propensity and full diagnostics.
- `update`: ridge rank-1 update of the pulled arm; `select` solves via float64 Cholesky each call, no cached inverse (ADR-0002).
- `QuotaLedger` with multi-level rolling windows and convex barrier (ADR-0005/0006).
- Atomic snapshot/restore with `state_version`; incompatible shapes are rejected, never migrated.
- `DecisionLog` + IPS/DR off-policy evaluator for offline α calibration.
- Property tests (SPD invariance, bonus monotonicity, determinism), shape-invariance test,
  p99 latency benchmark.

## Verification

    uv sync --all-extras
    uv run pytest -q                                   # all green
    uv run pytest tests/test_jit_invariants.py -q      # asserts compile count == 1 per fn
    uv run pytest tests/bench -q --benchmark-only      # p99 select < 1 ms @ K=32,d=32
    uv run python -m azuma_router.ope --log fixtures/decisions.jsonl --alpha 0.5,1.0,2.0

Cross-checks:
- Simulated bandit (fixtures/sim_arms.yaml): cumulative regret curve is sublinear
  (fit exponent < 0.6 over 1e5 rounds); every arm pulled ≥ 1 % of rounds.
- Quota sim: with 5h window at 100 units and Poisson demand of 150 units/5h, no
  window reaches u ≥ 1.0; mean utilization ≥ 0.85.
- Snapshot round-trip: `restore(snapshot(s)) == s` bitwise for A, b, mask, rng_key; restoring
  a snapshot with d' ≠ d raises `IncompatibleSnapshotError`.
- Daemon: `printf '{"jsonrpc":"2.0","id":1,"method":"health"}\n' | uv run python -m azuma_router.daemon`
  answers after the `ready` notification; duplicate `report` for one `decision_id` is a no-op.
- NumPy baseline vs JAX jit timing table included in PR body.
```

## §11 Siguientes pasos

1. ~~Grill Ronda 1~~ — aprobado 2026-09-14; ADR-0001…0012 registrados.
2. ~~Grill Ronda 2~~ — aprobado 2026-09-14; ADR-0013…0018 registrados.
3. Commit de Fase 0 en `main` (`docs: phase 0 proposal, glossary and adrs`) y rama `feat/0001-linucb-core`.
4. `specs/core/{spec,design,tasks}.md` (Fase 1) para los contratos con tipos exactos (`BanditState`, `QuotaView`, mensajes del daemon).
5. `write-plan` → slices verticales: (S1) state + linucb + tests numéricos; (S2) quota + reward; (S3) snapshot + OPE; (S4) daemon contract.
6. `implement` en worktree `wt switch --create azuma-router.0001-feat-linucb-core` bajo TDD estricto.
