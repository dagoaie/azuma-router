# ADR-0012 — Python ≥ 3.12, API pública mínima, licencia Apache-2.0

**Estado:** Aceptado (Grill R1 · Q12) · 2026-09-14

## Decisión
- `requires-python = ">=3.12"`; CI en 3.12 y 3.13. uv como único gestor.
- Paquete PyPI `azuma-router`, módulo `azuma_router`. API pública exactamente: `select`, `update`, `BanditState`, `StaticConfig`, `QuotaLedger`, `snapshot`, `restore`; daemon vía `python -m azuma_router.daemon`. Todo lo demás con prefijo `_`.
- Licencia Apache-2.0.

## Porqué
3.12 maximiza cobertura de wheels jaxlib; API mínima permite refactor interno sin romper a Convoy; Apache-2.0 aporta cláusula de patentes, relevante en un proyecto algorítmico.

## Rechazado / diferido
- 3.14 como mínimo: rechazado hasta que jaxlib publique wheels estables para él.
- MIT: rechazado.
