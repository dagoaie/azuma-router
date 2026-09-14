# CONTEXT.md — azuma-router

> High-performance Contextual Bandit LLM Router compiled in JAX with quota-aware shadow pricing.

**Estado del documento:** CERRADO para Fase 0 (Grill Ronda 1 aprobado 2026-09-14; decisiones en `docs/adr/0001…0012`). Este fichero es la fuente de verdad del vocabulario del dominio; toda ficha, ADR, test y símbolo público debe usar estos nombres.

**Nombre del proyecto.** *Azuma* honra la desigualdad de Azuma–Hoeffding: para una martingala $(S_n)$ con incrementos acotados $|S_i - S_{i-1}| \le c_i$, $\Pr(|S_n - S_0| \ge t) \le 2\exp\!\left(-\tfrac{t^2}{2\sum_i c_i^2}\right)$. Es la herramienta de concentración con la que se demuestra que el regret acumulado de LinUCB/SupLinUCB es sublineal, $\tilde O(\sqrt{dT})$ (Chu, Li, Reyzin & Schapire, 2011): la secuencia de errores de estimación del bandido es una diferencia de martingala respecto a la filtración de contextos observados.

---

## 1. Glosario del dominio

### 1.1 Arm (Brazo)

Una **opción de enrutamiento atómica** entre la que el bandido elige en cada ronda. Un brazo es la tupla `(model_id, provider, billing_mode)`, p. ej. `("gpt-5-codex", "openai-codex", SUBSCRIPTION)` o `("qwen3-coder", "openrouter", PER_TOKEN)`. Dos brazos pueden apuntar al mismo modelo por proveedores distintos: son brazos distintos porque su coste y latencia lo son.

- `K` = número de brazos configurados. Es un **invariante de forma** (§1.8): fijo entre recompilaciones JIT.
- Cada brazo mantiene estadísticas **disjuntas** (§1.3): no comparte parámetros con los demás.
- **Brazo inactivo / enmascarado**: brazo con `mask[a] = False` (cuota agotada, proveedor caído, o slot de padding). Sigue existiendo en el estado para preservar la forma; su score UCB se sustituye por $-\infty$ antes del `argmax`.

### 1.2 Context Vector ($x_t$)

Vector $x_t \in \mathbb{R}^d$ que describe la **petición** en la ronda $t$, construido por el `Featurizer` **fuera** del JIT boundary. Es la única entrada de datos al `select`. $d = 32$ fijo (ADR-0003). $x_t$ describe **exclusivamente la petición** — bloques escalados a $[0,1]$ con bias $x_0 = 1$ (ADR-0004): log-tokens de entrada, tipo de tarea (one-hot), flags (código / herramientas / multimodal), profundidad del agente en Convoy, padding a cero. El detalle exacto de bloques se cierra en Grill Ronda 2. **Nunca** contiene utilización de cuota ni ningún estado del router: la cuota actúa solo a través del precio sombra.

Invariantes:
- $d$ fijo por despliegue; cambiar $d$ = nueva versión de estado (§1.9) y recompilación.
- Cada componente en $[0,1]$ ⇒ cota declarada $\|x_t\|_2 \le L = \sqrt{d}$ (ADR-0004). Las cotas UCB (§1.4) asumen contextos acotados. NaN/inf o componente fuera de rango ⇒ `InvalidContextError` fuera del JIT; nunca `nan_to_num`.
- Contexto **compartido** entre brazos (variante *disjoint*): el mismo $x_t$ se puntúa contra los $K$ modelos lineales. No hay features específicas de brazo dentro de $x_t$ (eso sería LinUCB *hybrid*, fuera de alcance).

### 1.3 LinUCB (Disjoint) & Ridge Regularization

Algoritmo de bandido contextual lineal (Li, Chu, Langford & Schapire, 2010). Supone recompensa esperada lineal por brazo: $\mathbb{E}[r_{t,a} \mid x_t] = x_t^\top \theta_a^*$.

Estado por brazo $a$:
- $A_a = \lambda_{\text{ridge}} I_d + \sum_{s: a_s = a} x_s x_s^\top \in \mathbb{R}^{d\times d}$ (matriz de diseño regularizada, SPD por construcción).
- $b_a = \sum_{s: a_s = a} r_s x_s \in \mathbb{R}^d$.

Estimador ridge: $\hat\theta_a = A_a^{-1} b_a$, calculado en cada `select` por **Cholesky** (`cho_factor` / `cho_solve`) en float64 sobre $A_a$; el estado nunca guarda $A_a^{-1}$ (ADR-0002).

**Ridge regularization** ($\lambda_{\text{ridge}} > 0$, típicamente 1.0): garantiza que $A_a$ sea invertible desde $t=0$ (cold start del brazo) y acota el número de condición al inicio. Es un prior gaussiano isotrópico sobre $\theta_a$ en la lectura bayesiana. No confundir con el precio sombra $\lambda(t)$ (§1.5); en código y documentación se nombran `ridge_lambda` y `shadow_price` respectivamente, nunca `lambda` a secas.

**Disjoint** = un $\theta_a$ independiente por brazo. Ventaja: `jax.vmap` sobre el eje de brazos es trivial (estado con forma `(K, d, d)` y `(K, d)`). Coste: no hay transferencia de conocimiento entre brazos; cada brazo nuevo arranca frío.

### 1.4 Exploration Parameter ($\alpha$) y cotas UCB

Score de selección: $p_{t,a} = x_t^\top \hat\theta_a + \alpha \sqrt{x_t^\top A_a^{-1} x_t}$.

- Primer término: **explotación** (recompensa predicha).
- Segundo término: **bonus de exploración** = anchura del intervalo de confianza en la dirección $x_t$. Decrece a medida que el brazo acumula observaciones alineadas con $x_t$.
- $\alpha$ controla el trade-off. Teoría (Li et al. 2010, Thm. 1 vía Walsh et al.): $\alpha = 1 + \sqrt{\ln(2/\delta)/2}$ da un intervalo con probabilidad $1-\delta$. En práctica $\alpha$ se trata como hiperparámetro y se ajusta offline con OPE (§1.7).
- **Regret**: $R_T = \sum_t (x_t^\top\theta^*_{a_t^*} - x_t^\top\theta^*_{a_t})$. Cota $\tilde O(\sqrt{dT\ln K})$ para SupLinUCB; LinUCB práctico hereda el comportamiento empírico sin la cota exacta.

En azuma-router la recompensa $r_t \in [0,1]$ es un **escalar compuesto** del trilema: $r = \mathrm{clip}(w_q q - w_c \tilde c - w_l \tilde l, 0, 1)$ con $q$ aportada por el llamador, $\tilde c$ = coste USD real normalizado por `cost_budget_usd` (0 en brazos `SUBSCRIPTION`) y $\tilde l$ = latencia total normalizada por `latency_budget_s` (ADR-0007). El precio sombra (§1.5) entra como término de penalización aditivo al score, no como parte de $\hat\theta_a$, para que el aprendizaje del modelo de calidad no se contamine con el estado transitorio de cuotas.

### 1.5 Shadow Price ($\lambda(t)$) & Convex Barrier

**Precio sombra**: coste marginal *implícito* de consumir una unidad de un recurso con restricción, aunque su precio monetario sea cero. Para un modelo bajo suscripción plana (Codex Pro, Claude Max…) el token no cuesta dinero en el margen, pero consume **cuota** de una ventana rodante (§1.6); agotarla tiene un coste de oportunidad real (fallback forzado a brazos de pago o degradación).

Formalmente es el multiplicador de Lagrange de la restricción de cuota en el problema $\max \sum_t \text{calidad}_t - \text{coste}_t \;\text{s.a.}\; \sum_{t \in W} u_t \le Q_W$ para cada ventana $W$. Se calcula **dinámicamente** en cada ronda, no se aprende.

**Convex Barrier** (barrera convexa): función $\phi(u)$ de la utilización $u \in [0,1)$ de la ventana, convexa, creciente y con $\phi(u)\to\infty$ cuando $u\to 1$. Elegida (ADR-0006): **logarítmica** $\phi(u) = -\mu_a\ln(1-u)$, con $u$ recortado a $[0, 1-10^{-6}]$ dentro del JIT y $\mu_a$ configurable por brazo (default global; $\mu = 0$ en brazos `PER_TOKEN`). Garantiza que el router **nunca** elige un brazo cuya cuota está a punto de agotarse salvo que la ganancia de calidad lo compense de forma abrumadora, y hace el consumo suave en lugar de "todo hasta el 100 % y luego hard-cutoff".

$\lambda_a(t) = \max_{W \in \text{ventanas}(a)} \phi(u_{a,W}(t))$ (la ventana más restrictiva domina).

Score final: $\text{score}_{t,a} = p_{t,a} - \lambda_a(t)\cdot c_a(x_t)$, donde $c_a(x_t)$ es el consumo estimado de la petición en el brazo $a$, en `units` abstractas declaradas por brazo (`units_per_request`, `units_per_1k_tokens`).

### 1.6 Rolling Quota Window (ventanas de 5h y cuotas semanales)

Límite de consumo definido sobre una **ventana temporal deslizante**, no sobre un calendario fijo. Planes como Codex Pro o Claude Max exponen (a fecha de redacción; los valores concretos son **parámetros de configuración**, no constantes del código) dos capas simultáneas:

- **Ventana corta rodante** (~5 h): límite de mensajes/tokens en las últimas 5 h. Se recupera continuamente a medida que las peticiones antiguas salen de la ventana.
- **Cuota semanal** (~7 días rodantes o reset fijo semanal, según proveedor): techo global que las ventanas cortas no pueden superar en agregado.

Modelo en azuma-router (ADR-0005): cada brazo `SUBSCRIPTION` declara ventanas de dos variantes explícitas, `RollingWindow(duration_s, capacity)` y `FixedResetWindow(period, reset_anchor, capacity)`. El **QuotaLedger** es Python puro fuera del JIT: un `deque[(timestamp_s, units)]` exacto por brazo y ventana, podado en cada consulta, que produce `QuotaView.utilization: f32[K_max, W_max]` con $u_{a,W}(t) = \text{consumido}_{W}(t) / \text{capacity}_W$ justo antes de cada `select`. $W_{\max}$ es forma estática; slots sin ventana valen 0.

Regla dura: cuando cualquier $u_{a,W} \ge 1$ el brazo se enmascara (§1.1). La barrera (§1.5) actúa antes de llegar ahí.

### 1.7 Inverse Propensity Scoring (IPS) & Doubly Robust Estimation

Herramientas de **evaluación off-policy (OPE)**: estimar el valor de una política nueva $\pi$ usando logs generados por la política de comportamiento $\mu$ (la que estaba en producción). Necesarias para (a) ajustar $\alpha$ y la función de recompensa sin desplegar, (b) comparar versiones del router en CI con datos reales.

- **Propensity** $\mu(a_t \mid x_t)$: probabilidad con la que la política de logging eligió $a_t$. LinUCB puro es determinista (argmax) → propensity $\in\{0,1\}$ → IPS degenerado. Por eso el router registra una propensity no degenerada (ADR-0008): $\pi(a\mid x) = (1-\varepsilon)\,\mathrm{softmax}(\text{score}/\tau)_a + \varepsilon/K_{\text{activos}}$ sobre brazos activos, con `tau ≈ 0.05` y `epsilon_floor ≈ 0.02`; el suelo acota los pesos IPS a $\le K/\varepsilon$.
- **IPS**: $\hat V_{\text{IPS}}(\pi) = \frac1n\sum_i \frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)} r_i$. Insesgado si $\mu>0$ donde $\pi>0$; varianza alta cuando las propensities son pequeñas.
- **Doubly Robust (DR)**: $\hat V_{\text{DR}}(\pi) = \frac1n\sum_i \Big[\hat r(x_i,\pi) + \frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}\big(r_i - \hat r(x_i,a_i)\big)\Big]$ con un modelo de recompensa $\hat r$. Insesgado si **o bien** $\hat r$ **o bien** $\mu$ es correcto; varianza menor que IPS. El propio $\hat\theta_a$ de LinUCB sirve como $\hat r$.

Todo registro de decisión (`DecisionLog`) incluye: $x_t$, $a_t$, $\mu(a_t\mid x_t)$, $r_t$ (cuando llega), `state_version`, timestamp.

### 1.8 JIT Boundary & Shape Invariance en JAX

**JIT boundary**: la frontera entre código Python ordinario y una función trazada por `jax.jit`. Dentro, el código se traza una vez a XLA con **formas y dtypes concretos**; los valores son abstractos (tracers). Consecuencias de diseño:

- Todo lo que es *forma* (`K`, `d`, número de ventanas de cuota por brazo, longitud del deque) debe ser **estático** y conocido en el trazado. Cambiarlo dispara recompilación (cientos de ms → segundos), incompatible con el SLO de <1 ms.
- Todo lo que es *valor* (matrices $A_a$, $b_a$, utilización, máscaras, timestamp) es dinámico y viaja como argumento/retorno.
- Control de flujo dependiente de datos se expresa con `jnp.where` / `lax.cond` / máscaras, nunca con `if` de Python sobre tracers.
- Python no puede vivir dentro: sin I/O, sin dicts de tamaño variable, sin listas de brazos.

**Shape invariance**: invariante de que **todas** las llamadas a `select` y `update` en un despliegue usan exactamente las mismas formas. Se garantiza con **padding** de brazos hasta `K_max` + máscara, y se verifica con un test que cuenta compilaciones (`jax.jit(...).lower(...).compile()` una sola vez; `_cache_size()` estable).

Latencia objetivo: `select` compilado, forma `(K, d, d)` con $K \le 32$, $d \le 64$, en CPU: presupuesto **< 1 ms p99** incluida la transferencia host↔device (que en CPU es nula). El cold-start del JIT se paga una vez en arranque del daemon, nunca en la ruta caliente.

### 1.9 Immutable PyTree State

Todo el estado del bandido es un **PyTree inmutable** (contenedor anidado de arrays JAX: `NamedTuple` o `dataclass` congelada registrada con `jax.tree_util`). Las operaciones son **funciones puras**:

```
select : (BanditState, x_t, QuotaView) -> (BanditState', arm, propensity, Diagnostics)   # BanditState' lleva la rng_key avanzada
update : (BanditState, x_t, arm, r_t)       -> BanditState'
```

Nunca se muta `A_a` in situ; `update` devuelve un estado nuevo con `A_a' = A_a + x x^\top`. Ventajas: (a) JIT/vmap requieren pureza; (b) snapshots atómicos = serializar un valor, sin locks; (c) rollback trivial (conservar el estado anterior); (d) reproducibilidad exacta dado `(state, x, key)`.

Representación (ADR-0001/0002): `BanditState(NamedTuple)` con solo arrays dinámicos — `A: f64[K_max,d,d]`, `b: f64[K_max,d]`, `n_pulls: i32[K_max]`, `mask: bool[K_max]`, `rng_key: u32[2]`. Hiperparámetros y formas (`K_max`, `d`, `ridge_lambda`, `alpha`, `tau`, `epsilon_floor`, `barrier_mu`, pesos de recompensa) en `StaticConfig`, un `dataclass(frozen=True)` hashable pasado por `static_argnums`. `state_version` vive en los metadatos del snapshot (ADR-0009), no en el PyTree.

`state_version` es un entero monótono que acompaña cada snapshot; un snapshot con `d` o `K_max` distintos a los del binario que lo carga se **rechaza** (fail-fast, sin migración implícita — política hard-cut).

---

## 2. Términos auxiliares

| Término | Definición |
|---|---|
| **Reward ($r_t$)** | Escalar en $[0,1]$ que resume calidad, coste y latencia observados tras ejecutar la petición en el brazo elegido (ADR-0007). |
| **Regret** | Diferencia acumulada entre la recompensa del mejor brazo en retrospectiva y la obtenida. Métrica primaria de evaluación offline. |
| **Featurizer** | Componente Python (fuera del JIT) que convierte una `RoutingRequest` en $x_t$. Determinista y versionado. |
| **QuotaLedger** | Registro de consumo por brazo y ventana; produce la `QuotaView` (utilizaciones $u_{a,W}$) que entra al `select`. |
| **DecisionLog** | JSONL append-only con rotación; cada línea: `decision_id`, $x_t$, `arm`, `propensity`, `reward` (o `null` si expiró), `state_version`, timestamps. Insumo de OPE (ADR-0009/0011). |
| **PendingStore** | Decisiones emitidas por `select` a la espera de `report`; TTL default 3600 s; al expirar pasan al `DecisionLog` con `reward = null` y **no** actualizan el bandido (ADR-0011). |
| **decision_id** | UUIDv7 emitido por `select`; clave de idempotencia de `report` (ADR-0011). |
| **Snapshot** | `state.safetensors` (arrays) + `state.json` (metadatos, config, ledger) escritos en directorio temporal y publicados con `rename` atómico; formas incompatibles ⇒ `IncompatibleSnapshotError`, sin migración (ADR-0009). |
| **Sidecar / Plugin** | Proceso hijo `python -m azuma_router.daemon` que expone `select`/`report`/`snapshot`/`health` a Convoy por **JSON-RPC 2.0 sobre stdio**, un mensaje por línea; `stderr` para logs (ADR-0010). |
| **Convoy** | Orquestador de agentes en TypeScript, consumidor futuro del router. Fuera de este repositorio. |
| **Trilema Q-C-L** | Calidad–Coste–Latencia. El router no maximiza uno: optimiza una combinación escalarizada y penalizada por cuota. |

---

## 3. Convenciones de nombres (para código futuro)

- `ridge_lambda` (regularización), `shadow_price` (multiplicador de cuota), `alpha` (exploración). Nunca `lambda`.
- Ejes de arrays: `(K, ...)` siempre primero (eje de brazos), luego `(d, ...)`.
- Funciones puras en snake_case sin efectos; cualquier I/O vive en módulos con sufijo `_io`.
- Tiempo siempre en segundos UNIX float64 UTC; nunca `datetime` dentro del JIT.

## 4. Referencias

- Li, Chu, Langford, Schapire (2010). *A Contextual-Bandit Approach to Personalized News Article Recommendation*. WWW.
- Chu, Li, Reyzin, Schapire (2011). *Contextual Bandits with Linear Payoff Functions*. AISTATS.
- Abbasi-Yadkori, Pál, Szepesvári (2011). *Improved Algorithms for Linear Stochastic Bandits*. NeurIPS.
- Dudík, Langford, Li (2011). *Doubly Robust Policy Evaluation and Learning*. ICML.
- Azuma (1967). *Weighted sums of certain dependent random variables*. Tôhoku Math. J.
- Ong et al. (2024). *RouteLLM: Learning to Route LLMs with Preference Data*. arXiv:2406.18665.
- JAX docs: *JIT mechanics*, *Pytrees*, *Sharp bits*.
