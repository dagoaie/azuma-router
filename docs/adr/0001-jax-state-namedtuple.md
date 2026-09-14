# ADR-0001 — `BanditState` como `NamedTuple` + `StaticConfig` hashable

**Estado:** Aceptado (Grill R1 · Q1) · 2026-09-14

## Contexto
JAX exige distinguir lo que es *forma* (estático, fija el trazado) de lo que es *valor* (dinámico, viaja como array). Los contenedores candidatos eran `NamedTuple` (PyTree nativo), `flax.struct.dataclass` / `chex.dataclass` (campos `pytree_node=False`) y `equinox.Module`.

## Decisión
- `BanditState(NamedTuple)` contiene **solo arrays dinámicos**: `A: f64[K_max,d,d]`, `b: f64[K_max,d]`, `n_pulls: i32[K_max]`, `mask: bool[K_max]`, `rng_key: u32[2]`.
- `StaticConfig` es un `@dataclass(frozen=True)` hashable con `K_max`, `d`, `ridge_lambda`, `alpha`, `tau`, `epsilon_floor`, `barrier_mu: tuple[float, ...]`, pesos de recompensa; se pasa por `static_argnums` a `select_jit` / `update_jit`.
- `state_version` y metadatos de snapshot viven fuera del PyTree (ADR-0009).

## Porqué
Cero dependencias por un contenedor; la frontera estático/dinámico queda codificada en **dos tipos distintos**, no en un decorador por campo; `NamedTuple` es PyTree sin registro y es trivialmente serializable campo a campo.

## Rechazado / diferido
- flax/chex/equinox: arrastrar un framework para un contenedor. Rechazado.
- `state_version` dentro del PyTree: cambiaría el hash de trazado o viajaría como array sin sentido. Rechazado.
