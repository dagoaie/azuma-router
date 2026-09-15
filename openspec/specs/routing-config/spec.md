# routing-config Specification

## Purpose
Carga y validación estricta de `azuma.toml` (brazos, ventanas de cuota e hiperparámetros) y su proyección a la configuración estática e inmutable del bandido. Contrato exacto del fichero: `docs/contracts/linucb-core.md` §4.1. Decisiones: ADR-0012, ADR-0017.

## Requirements

### Requirement: Carga de configuración inmutable
`load_config(path)` SHALL devolver un `Config` inmutable cuyos brazos quedan indexados por orden de aparición en `[[arms]]`, con `arm_id = "{model_id}@{provider}"` único (FR-01).

#### Scenario: TOML válido con dos brazos
- **WHEN** se carga `fixtures/config/azuma.valid.toml`
- **THEN** `Config.arms` tiene longitud 2, `arms[0].arm_id == "gpt-5-codex@openai-codex"` y `arms[1].arm_id == "qwen3-coder@openrouter"` (AC-01)

### Requirement: Rechazo fail-fast con ruta de campo
El cargador SHALL lanzar `ConfigError` con la ruta exacta del campo ofensivo (por ejemplo `arms[2].windows[0].capacity`) ante `arm_id` duplicado, sección obligatoria ausente, tipo incorrecto, valor numérico no finito o con `|v| > 2³¹ − 1` (incluida `capacity`), `tau < 1e-6`, `billing_mode = "subscription"` sin ventanas, o `billing_mode = "per_token"` con ventanas o con `barrier_mu > 0`. El cargador MUST NOT sustituir valores por defecto silenciosamente (FR-02).

#### Scenario: temperatura degenerada
- **WHEN** se carga un TOML con `bandit.tau = 1e-300`
- **THEN** se lanza `ConfigError` con `field == "bandit.tau"`

#### Scenario: brazo de suscripción sin ventanas
- **WHEN** se carga un TOML donde un brazo `subscription` no declara `[[arms.windows]]`
- **THEN** se lanza `ConfigError` con `field == "arms[i].windows"` (AC-02)

#### Scenario: brazo per_token con barrera
- **WHEN** se carga un TOML donde un brazo `per_token` declara `barrier_mu = 0.5`
- **THEN** se lanza `ConfigError` con `field == "arms[i].barrier_mu"` (AC-02)

### Requirement: Límites de forma estática
El cargador SHALL verificar `1 ≤ K ≤ k_max` y `max(len(arm.windows)) ≤ w_max`, y `d == 32` para `featurizer_version = 1` (FR-03).

#### Scenario: más brazos que k_max
- **WHEN** el TOML declara `k_max = 2` y tres `[[arms]]`
- **THEN** se lanza `ConfigError` con `field == "arms"` y la razón menciona `k_max`

### Requirement: Proyección a StaticConfig hashable
`to_static_config(config)` SHALL producir un `StaticConfig` frozen y hashable con `barrier_mu` e `is_subscription` como tuplas de longitud `k_max`, con `0.0`/`False` en las posiciones de padding y `barrier_mu = 0.0` en brazos `per_token`.

#### Scenario: padding coherente
- **WHEN** `k_max = 32` y hay 2 brazos, el primero `subscription` con `barrier_mu = 0.5`
- **THEN** `len(static.barrier_mu) == 32`, `static.barrier_mu[0] == 0.5`, `static.barrier_mu[2:] == (0.0,)*30` y `hash(static)` no lanza
