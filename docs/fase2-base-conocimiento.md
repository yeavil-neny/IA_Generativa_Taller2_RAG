# Fase 2 — Creación de la Base de Conocimiento de Documentos

**Caso de estudio:** EcoMarket — el rendimiento de un sistema RAG depende, en primer lugar, de la calidad de la información que puede recuperar.

**Equipo:** Cristian Humberto Yepes Marin y Yenny Alexandra Villarreal Suaza

---

## 1. Identificación de documentos

El sistema de atención al cliente de EcoMarket necesita responder consultas de cuatro naturalezas distintas. Para mantenernos en la realidad de una empresa de e-commerce (y no inventar datos), se identificaron **cuatro tipos de documentos** — dos reutilizados del Taller Práctico #1, dos creados para este taller:

| Tipo de documento | Archivo | Formato | Contenido | Por qué es crucial para soporte |
|---|---|---|---|---|
| 1. Política de devoluciones | `data/politicas_devolucion.md` | Markdown, por secciones | Condiciones generales, categorías no devolvibles, excepciones, proceso de retorno, plazos de reembolso | Reglas del negocio que el asistente debe citar con fidelidad: una respuesta incorrecta aquí genera reclamos reales |
| 2. Registro de pedidos | `data/pedidos.json` | JSON estructurado | 12 pedidos con número de seguimiento, cliente, productos, estado, fechas, transportadora, enlace de rastreo | La consulta más frecuente del canal es "¿dónde está mi pedido?"; los datos son por-pedido y cambian con el tiempo |
| 3. Inventario de productos | `data/inventario_productos.csv` | CSV tabular | 18 SKUs con nombre, categoría, precio, stock, disponibilidad y notas | Permite responder preguntas de catálogo: precios, disponibilidad, categorías de los productos |
| 4. Preguntas frecuentes | `data/faqs.json` | JSON, pares pregunta-respuesta | 12 FAQ de envíos, pagos, suscripciones, sostenibilidad y atención humana | Cubre el tipo de consultas repetitivas que justificaron el proyecto (80% de los tickets en el planteamiento del caso) |

**Criterio de identificación:** un documento entra a la base de conocimiento si el asistente puede necesitarlo para responder una solicitud razonable de un cliente. Se descartaron conscientemente datos sensibles que el asistente no debe ni necesita consultar (historial de pagos con datos de tarjetas, datos de contacto personales completos), continuando el análisis de privacidad de las Fase 2 del Taller Práctico #1: el documento de pedidos solo expone el nombre en formato corto ("María G.") y el número de seguimiento sirve como llave de consulta.

**Nota:** en producción, el registro de pedidos y el inventario estarían en una base de datos del ERP que se sincronizaría al índice vectorial (o se consultarían como *tools* del agente, como se discute en la Fase 3); para este taller se trabaja con archivos estáticos, que cumplen el mismo rol didáctico.

---

## 2. Estrategia de segmentación (chunking)

### 2.1 El problema del chunking

El modelo de embeddings no procesa documentos completos: procesa fragmentos cortos. La calidad de la recuperación depende de una propiedad de los chunks: **deben ser semánticamente autónomos** — un chunk debe contener por sí solo la respuesta a (por lo menos) un tipo de consulta. Si un chunk corta a mitad de una regla, el recuperador puede encontrar el chunk pero la respuesta estará incompleta, y el LLM tenderá a completar el hueco con conocimiento general: reaparece la alucinación que el RAG venía a resolver.

### 2.2 Estrategias comparadas

| Estrategia | Cómo funciona | Ventajas | Desventajas en este caso |
|---|---|---|---|
| **Tamaño fijo** | Corta cada N caracteres o tokens y solapa (overlap) | Simple, rápido, homogéneo para índices grandes | Puede cortar reglas, filas o pares Q-A por la mitad; con corpora heterogéneos produce chunks de calidad arbitraria |
| **Por párrafos / delimitadores** | Corta en separadores estructurales (`\n\n`, fin de fila, fin de objeto) | Respeta la unidad semántica natural de cada formato | Puede producir chunks demasiado largos o demasiado cortos; por sí sola no maneja párrafos gigantes |
| **Recursiva con jerarquía de separadores** | Intenta los separadores más fuertes (`\n\n`) y, si el trozo sigue siendo grande, baja a separadores más finos (`\n`, `.`, `»)`, con overlap | Compromiso estándar de la industria: respeta la estructura casi siempre y garantiza un tamaño máximo | Requiere calibrar tamaño y overlap con el modelo de embeddings elegido |

Es la elección del script de referencia `rag_ejemplo.py` (`RecursiveCharacterTextSplitter`), y también la del notebook de clase. El equipo coincide en que es la mejor de las tres **pero con un ajuste importante para este caso: separadores personalizados por tipo de documento**, porque los cuatro documentos son de formatos distintos y sus unidades semánticas naturales no son las mismas.

### 2.3 Decisiones por tipo de documento

| Documento | Separadores jerárquicos | Tamaño de chunk objetivo |
|---|---|---|
| Política (Markdown) | `##` (sección) → `###` → `¶` (párrafo) → fin de oración | Cada sección (≈200-600 caracteres) es un chunk propio |
| Pedidos (JSON) | Un objeto JSON por pedido (no text-splitter: serialización directa) | Un pedido completo por chunk, con metadatos (`tipo`, `tracking`) |
| Inventario (CSV) | Una fila por ítem (no text-splitter: parse tabular) | Un producto por chunk, con metadatos (`tipo`, `sku`) |
| FAQ (JSON) | Un par pregunta-respuesta por FAQ | Una FAQ completa por chunk, con metadatos (`tipo`, `id`) |

**Justificación de cada decisión:**

- **Política por secciones:** cada sección ("Condiciones generales", "Productos que NO se pueden devolver", "Excepciones", "Proceso") responde una familia distinta de consultas. Con el tamaño de section cabe holgadamente dentro del contexto de entrada del modelo de embeddings elegido (128 tokens, véase la Fase 1) — un chunk típico de esa política ronda los 400-800 caracteres, es decir, el final de un chunk jamás quedaría truncado.
- **Pedidos e inventario "por registro" en lugar de text-splitter:** un pedido o un SKU es el átomo de información que el cliente consulta. Cortarlos con un splitter de texto mezclaría campos de pedidos distintos en un mismo chunk (por ejemplo, el estado de ECO-10003 junto con productos de ECO-10004), lo que induciría a error al LLM. Además, la metadata por chunk (`tracking`, `sku`, `tipo`) permite filtros estructurados y trazabilidad.
- **FAQ por par Q-A:** la pregunta y su respuesta deben ir juntas — la pregunta aporta la superficie semántica que más se parece a la consulta real del usuario, y el chunk que la contiene llega al prompt **con su respuesta dentro**. Separarlos obligaría a que dos chunks vecinos coincidieran en la recuperación.
- **Overlap 0 en los chunking de documentos estructurados:** cada registro es completo por sí mismo, por lo que no requiere solapamiento. En los chunks de tipo texto se usa un solapamiento pequeño (80 caracteres) para que un pasaje transversal no pierda su cierre.

### 2.4 Calibración con el modelo de embeddings elegido

La decisión de chunking no es independiente del componente de la Fase 1: `paraphrase-multilingual-MiniLM-L12-v2` trunca su entrada a 128 tokens **silenciosamente** (sin advertirlo), y un chunk truncado pierde el final de su contenido tanto en el índice como en la recuperación. La calibración, en consecuencia, no se estimó por caracteres sino **midiendo el tokenizador real del modelo**:

- Techo de texto: 400 caracteres por chunk (en español, ~95-115 tokens); etiquetas compactas de referencia (`[Pedido ECO-10007]`, `[Política EcoMarket | sección]`, `[Producto SKU-000]`, `[FAQ id | categoría]`) al inicio, para que el modelo sepa de dónde proviene cada chunk sin gastar techo.
- Overlap 80 de los chunks de texto de la política.
- Los enlaces de rastreo no entran en el texto del chunk (una URL consume ~12 tokens y su formato es determinista a partir del número de seguimiento); viven en la metadata y la plantilla del prompt instruye construirla cuando la respuesta es sobre un pedido.
- **Verificación token a token:** los 50 chunks del corpus quedan entre 60 y 122 tokens — cero truncamiento silencioso. El notebook incluye la celda que repite esta medición.

Este ajuste también tuvo un efecto de privacidad apreciable: el nombre del cliente salió del texto de los chunks de pedidos (va solo en la metadata, no se vectoriza), reduciendo los datos personales que quedan en el índice — coherente con la postura de privacidad de la Fase 1 y del Taller Práctico #1.

Esto se verifica experimentalmente en la Fase 3 (se imprime el número y tamaño de chunks por documento).

---

## 3. Indexación (embedding + carga de la base vectorial)

El proceso de indexación — qué pasa desde los archivos hasta la base de datos vectorial — tiene estos pasos:

1. **Carga:** se leen los cuatro archivos desde `data/` y se normalizan a documentos con `page_content` (texto a vectorizar) y `metadata` (tipo de documento, identificador: `tracking`, `sku`, `id de FAQ`).
2. **Segmentación:** según la tabla 2.3 — splitter recursivo para la política; splitters por registro para pedidos, inventario y FAQ.
3. **Vectorización (embedding):** cada chunk se pasa por `paraphrase-multilingual-MiniLM-L12-v2`, que lo convierte en un vector de 384 dimensiones. Chunks con significado similar quedan con vectores cercanos en el espacio de 384 dimensiones: "¿puedo devolver el café?" queda cerca de "los productos perecederos no son elegibles para devolución" aunque ninguna palabra coincida exactamente.
4. **Carga (upsert) en ChromaDB:** los vectores, junto con su texto original y metadata, se insertan en una colección de Chroma, que indexa internamente las similitudes con HNSW (*Hierarchical Navigable Small World*, un grafo de capas que permite encontrar los vecinos más cercanos sin comparar contra todos los vectores; consultas por aproximación, eficientes). Se configura el espacio de similitud por **coseno**, cuyo rango es interpretable (0 = textos idénticos en dirección, mayor = menos similares) y permite definir el umbral de escalamiento de la Fase 3.
5. **Persistencia:** la colección se escribe en disco (`chroma_ecomarket/`): reindexaciones posteriores solo añaden los documentos nuevos; en el prototipo se reindexa desde cero en cada corrida porque el corpus es minúsculo (menos de un segundo de cómputo) y así se garantiza la reproducibilidad.

Diagrama del flujo de indexación:

```
politicas_devolucion.md ┐
pedidos.json            ├─> Carga y normalización (page_content + metadata)
inventario_productos.csv┤         │
faqs.json               ┘         v
                          Segmentación por tipo de documento
                                    │
                                    v
                          Embeddings (MiniLM multilingüe, 384 dim.)
                                    │
                                    v
                          Upsert en ChromaDB (cosíneo) ─> persistencia en disco
```

Al terminar la indexación, el sistema tiene una **base de conocimiento consultable por similitud semántica**, no solo por palabras clave. El detalle de código y su ejecución en Colab se documenta en la Fase 3 y en el notebook.

---

## 4. Por qué esta fase determina el rendimiento del sistema

- Si un documento clave no está en la base de conocimiento, no hay consulta que lo recupere: el asistente responderá mal o tendrá que escalar. La identificación de la tabla 1 define el *techo* de lo que el asistente puede responder.
- Si el chunking fragmenta las reglas (tamaño fijo), el recuperador puede traer contexto incompleto, y el LLM "completa" con conocimiento general — alucinación reconstruida. Es la misma lección del experimento sin contexto del Taller Práctico #1 (`--basico`: la calidad de lo que entra al prompt define la calidad de lo que sale.
- Si los chunks son demasiado grandes (o el modelo trunca), el prompt se llena con ruido y el LLM pierde de vista lo relevante (dilución del contexto).
- La metadata permitirá en la Fase 3 un comportamiento más "agency-like": el asistente puede razonar sobre *qué tipo* de documento recuperó y decir con honestidad cuándo nada recuperado es pertinente.