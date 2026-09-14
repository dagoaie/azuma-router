# ADR-0017 — Configuración declarativa `azuma.toml`; cambio de config = reinicio; snapshot rechaza si cambia el conjunto de brazos

**Estado:** Aceptado (Grill R2 · Q17) · 2026-09-14 · **Decisión arquitectónica definitiva**

## Decisión
- Fichero TOML (stdlib `tomllib`) pasado por `--config`. Secciones `[bandit]`, `[[arms]]`, `[[arms.windows]]`. Schema exacto en `docs/contracts/linucb-core.md`.
- `arm_id = "{model_id}@{provider}"`, único. El orden de `[[arms]]` define el índice $a$.
- El snapshot guarda `arm_ids: list[str]` ordenada; en `restore`, si difiere del config ⇒ `IncompatibleSnapshotError`.
- Cualquier cambio en `azuma.toml` requiere reiniciar el daemon.

## Porqué
TOML sin dependencias; validación estricta con `tomllib` + dataclasses. La lista de brazos es forma (índice en `A[K_max]`), no valor: cambiarla en caliente sin remapeo silencioso es imposible bajo hard-cut.

## Rechazado / diferido
- YAML: rechazado (dependencia).
- Hot-reload / añadir brazo conservando aprendizaje: **follow-up F11** (requiere remapeo explícito de índices y una herramienta `azuma migrate-arms`; contradice el rechazo estricto v1).
