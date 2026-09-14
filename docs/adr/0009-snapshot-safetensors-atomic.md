# ADR-0009 — Snapshot = safetensors + JSON sidecar, rename atómico, rechazo fail-fast, sin migraciones

**Estado:** Aceptado (Grill R1 · Q9) · 2026-09-14

## Contexto
`BanditState` + `QuotaLedger` + metadatos deben persistir de forma atómica y recuperable; el `DecisionLog` tiene un ciclo de vida distinto (append-only).

## Decisión
- Arrays (`A`, `b`, `n_pulls`, `mask`, `rng_key`) → `state.safetensors`.
- Metadatos → `state.json`: `state_version`, `schema_version`, `StaticConfig` completo, `featurizer_version`, ledger serializado (eventos vivos por ventana), `created_at`, `seed`.
- Escritura: directorio temporal → `fsync` de ambos ficheros y del temporal → `rename` atómico del directorio `snapshot.tmp` → `snapshot/` (dos ficheros bajo un mismo rename) → `fsync` del directorio padre, sin el cual el rename no es durable → borrar `snapshot.old/`.
- Disparo: cada `max(N = 100 updates, T = 30 s)` — el primero que ocurra tras el mínimo — más snapshot en shutdown limpio.
- `restore`: si `d`, `K_max`, `W_max` o `schema_version` difieren del binario → `IncompatibleSnapshotError`. Sin migración (política hard-cut). Recuperación explícita: borrar snapshot y re-entrenar, o exportar `DecisionLog` y re-aprender.
- `DecisionLog` → JSONL independiente con rotación por tamaño; nunca dentro del snapshot.

## Porqué
safetensors es un formato zero-copy, sin pickle, con header JSON; el sidecar mantiene los metadatos legibles. Rename de directorio garantiza que los dos ficheros son coherentes entre sí.

## Rechazado / diferido
- orbax-checkpoint: rechazado (pensado para modelos grandes, dependencia pesada).
- `np.savez` / msgpack / pickle: rechazado (pickle inseguro; los otros sin ventaja sobre safetensors).
- Migraciones de esquema: follow-up F10 (nunca, salvo decisión explícita).
