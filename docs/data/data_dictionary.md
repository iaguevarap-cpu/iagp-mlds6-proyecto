# Diccionario de datos

## 1. Diccionario de Variables — Archivos CSV

Los archivos CSV tienen **3 columnas sin encabezado**. Después de renombrar:

| # | Nombre de columna | Tipo | Descripción | Valores posibles | Ejemplo |
|---|---|---|---|---|---|
| 0 | `image_original_name` | `str` | Ruta relativa de la **firma de referencia** dentro de la carpeta `train/` o `test/` | Cadena de texto con formato `<id>/<nombre_archivo>.png` | `"055/055_01.png"` |
| 1 | `image_comparison_name` | `str` | Ruta relativa de la **firma a comparar** contra la referencia | Cadena de texto con formato `<id>/<nombre_archivo>.png` | `"055/055_forg_01.png"` |
| 2 | `label` | `int` | Etiqueta del par: indica si las dos firmas son del mismo autor | `0` = genuino (mismo autor) · `1` = falsificado (imitación) | `1` |

> **Nota sobre la etiqueta:** La convención `0 = genuino / 1 = falsificado` es opuesta a la intuición de "similitud". Un score alto del modelo indica que las firmas son **diferentes** (posible falsificación), no similares. Esta convención se mantiene consistente con el dataset original.

---

## 2. Diccionario de Variables — Imágenes

Cada imagen individual representa una firma manuscrita escaneada con las siguientes características:

| Atributo | Descripción | Valor |
|---|---|---|
| `canal` | Modo de color | Escala de grises (1 canal) |
| `formato` | Extensión del archivo | PNG |
| `resolución_original` | Dimensiones antes del preprocesamiento | Variable (~400–1500 px de ancho) |
| `resolución_preprocesada` | Dimensiones después del preprocesamiento | 105 × 105 px |
| `rango_pixeles_raw` | Valores de píxel sin normalizar | [0, 255] (uint8) |
| `rango_pixeles_norm` | Valores de píxel normalizados | [0.0, 1.0] (float32) |
| `convención_color` | Interpretación del valor de píxel | 0 = negro (tinta) · 255 = blanco (papel) |
| `tipo_firma_genuina` | Identificador en el nombre del archivo | `<id>_<num>.png` (sin sufijo `forg`) |
| `tipo_firma_falsificada` | Identificador en el nombre del archivo | `<id>_forg_<num>.png` (con sufijo `forg`) |

---

- **Variable**: nombre de la variable.
- **Descripción**: breve descripción de la variable.
- **Tipo de dato**: tipo de dato que contiene la variable.
- **Rango/Valores posibles**: rango o valores que puede tomar la variable.
- **Fuente de datos**: fuente de los datos de la variable.


