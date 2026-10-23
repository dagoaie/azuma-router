# ADR-0020 — El arranque solo confía en estado propio y no escribible por terceros (verificar, no reparar)

**Estado:** Aceptado (auditoría de seguridad del daemon, hallazgo #2) · 2026-10-23

## Contexto
`restore_snapshot` borra `snapshot.tmp/`, renombra `snapshot.old/` y lee `state.json` / `state.safetensors`; el replay lee el WAL activo y el rotado. Los permisos `0700/0600` se aplican al escribir, pero nada comprobaba al leer que el estado siguiera siendo nuestro: un directorio escribible por otro usuario permite falsificar decisiones que el replay aplicaría al bandit, o plantar un `snapshot.tmp/` que `rmtree` seguiría.

## Decisión
- `wal.require_trusted_paths(paths)`: para cada ruta **existente** exige `st_uid == geteuid()` y ningún bit de escritura de grupo/otros (`mode & 0o022 == 0`). Las lecturas abiertas (0644) son aceptables; las rutas ausentes se omiten; se siguen symlinks (se juzga lo que realmente se leería).
- `restore_snapshot` lo ejecuta **primero**, antes de `rmtree`/`rename`/lectura, sobre la raíz, `snapshot/`, `snapshot.tmp/`, `snapshot.old/` y sus `state.*`. `Router.boot` lo ejecuta sobre el WAL activo y el rotado.
- Fallo ⇒ `UnsafeStateError` ⇒ evento `unsafe_state_permissions`, salida 3 (igual que `incompatible_snapshot`).
- No se comprueba el directorio padre del WAL: puede ser compartido (`/tmp`) y su contenido ya se juzga fichero a fichero.

## Porqué
Verificar es coherente con FR-25 («`restore` nunca altera permisos»): no se amplía el contrato a modificar nada ajeno. Reparar con `chmod` ocultaría una posible manipulación y alteraría un directorio externo.

## Rechazado / diferido
- Corregir permisos automáticamente (`chmod`/`chown`): rechazado (silent fallback; el operador debe investigar).
- Exigir exactamente `0700/0600`: rechazado; rompería backups restaurados con 0644 sin ganar seguridad, el riesgo está en la escritura.
- Comprobar el directorio padre del WAL: rechazado (falsos positivos en directorios compartidos).
