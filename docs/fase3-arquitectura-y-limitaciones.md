# Fase 3 — Integración y Ejecución del Sistema RAG

**Caso de estudio:** EcoMarket — código con el que el asistente del Taller Práctico #1 pasa de atender dos flujos predefinidos a responder cualquier solicitud sobre la empresa, apoyado en el RAG diseñado en las Fases 1 y 2.

**Equipo:** Cristian Humberto Yepes Marin y Yenny Alexandra Villarreal Suaza

---

## 1. Arquitectura del flujo

El sistema conecta las piezas decididas en la Fase 1 (embeddings + base vectorial) y construidas en la Fase 2 (base de conocimiento segmentada e indexada) alrededor del asistente del Taller Práctico #1, que conserva su patrón de prompt **Rol → Contexto → Instrucciones → Restricciones**:

```
                 Consulta del cliente (cualquier tema)
                                  |
                                  v
                Embeddings del modelo elegido en la Fase 1
                (paraphrase-multilingual-MiniLM-L12-v2)
                                  |
                                  v
        Busqueda por similitud coseno en ChromaDB (Fase 2)
                -> top-k chunks con su distancia
                                  |
                    +-------------+-------------+
                    |                           |
              distancia <= umbral          distancia > umbral
            hay material pertinente      nada pertinente
                    |                           |
                    v                           v
   Plantilla del Taller #1 rellenada con    Respuesta honesta:
   los chunks recuperados (Contexto) y      el asistente informa que
   consulta (Consulta del cliente)          no tiene las herramientas
                    |                       y ofrece escalar a humano
                    v
            LLM generador (Gemini, capa gratuita,
            endpoint compatible con OpenAI)
                    |
                    v
              Respuesta al cliente
```

**El punto de inflexión frente al Taller Práctico #1 es el umbral.** Antes, el contexto llegaba siempre (era la inyección directa del pedido); ahora el sistema *verifica* si lo recuperado es pertinente antes de dejar que el LLM hable. Si la similitud coseno del mejor chunk cae por debajo del umbral, no se consulta al LLM: se responde con transparencia. Esto implementa el comportamiento exigido por el enunciado — "responda con la respuesta que parece correcta **o haciéndole saber que no tiene las herramientas**" — y es, visto con el marco de la Semana 4, el proceso de *reflexión* del agente: el sistema evalúa la pertinencia de su herramienta de recuperación antes de ejecutar la generación.

### 1.1 Lectura desde la Semana 4 (agentes y agentic workflows)

El enunciado pide extender el asistente para "cualquier tipo de solicitud"; la unidad 3 aporta el marco para nombrar ese comportamiento:

- **Herramienta de ampliación de conocimiento:** el recuperador RAG es exactamente la clase de herramienta descrita en clase ("recuperación de datos, APIs, navegación web") — dota al asistente de la base de conocimiento interna de EcoMarket.
- **Herramientas definen el alcance:** el umbral de similitud materializa en código la idea de que un conjunto de herramientas "determina las tareas que el agente puede realizar": lo que no está en la base de conocimiento, el asistente lo declara fuera de su alcance y ofrece escalar a un asesor humano (FAQ-11).
- **Reflection:** la verificación del umbral es autocrítica durante la ejecución (el sistema monitorea el resultado de su herramienta antes de la siguiente acción), y la opción de escalar corresponde a "identificar desviaciones o fallos tempranamente".

### 1.2 Adaptaciones respecto del script de referencia (`rag_ejemplo.py`)

| Elemento de `rag_ejemplo.py` | Versión EcoMarket (este taller) | Motivo |
|---|---|---|
| Embeddings con `OllamaEmbeddings` (llama2 local) | `HuggingFaceEmbeddings` open-source (Fase 1) | Sin descargas de GB ni dependencia de un servidor local; corpus pequeño, costo cero |
| Vectorstore `Chroma` en memoria | `Chroma` **persistente en disco** con espacio **coseno** | Reproducibilidad y umbral interpretable para el escalado |
| Un único documento de texto estático | 4 tipos de documentos con **chunking por tipo** (Fase 2) | Base de conocimiento realista y recuperable por unidades semánticas |
| `RetrievalQA.from_chain_type` (chain opaca "stuff") | Flujo explícito: `similarity_search_with_score` → verificación de umbral → plantilla propia del Taller #1 → LLM | La cadena genérica no permite (a) leer distancias para decidir el escalado ni (b) conservar el patrón de prompts del Taller #1; el flujo explícito también es lo que la rúbrica pide "explicar sección por sección" |
| `Ollama` local como LLM | Gemini capa gratuita vía endpoint compatible con OpenAI (`ChatOpenAI` con `base_url` de Gemini) | Continuidad con el Taller #1; solución 100% gratuita; el mismo modelo que ya demostró buen comportamiento con el patrón de prompts |
| Bucle infinito de chat sin verificación | Bucle de chat + casos de prueba reproducibles en el notebook | Evidencia de cumplimiento de la rúbrica (cada sección explicada, cambios observables) |

Se mantiene la decisión de guiar la implementación por `rag_ejemplo.py` (LangChain) y no por el notebook de clase con LlamaIndex: la Fase 1 del taller pide justificar embeddings y base vectorial, no el framework, y `rag_ejemplo.py` es la referencia explícita del enunciado. Sí se tomó del notebook de clase su estilo: markdown que explica antes de cada bloque de código.

---

## 2. Cómo se ejecuta

**En Colab (recomendado):** abrir el notebook `notebooks/Taller2_RAG_EcoMarket.ipynb` con el botón "Abrir en Colab" del README. El notebook clona este repositorio, instala dependencias, construye la base de conocimiento desde `data/`, ejecuta los casos de prueba y pide la clave API de Gemini por `getpass` (nunca se guarda ni se imprime).

**Localmente:** `pip install -r requirements.txt`, copiar `.env.example` a `.env` con la clave `GEMINI_API_KEY`, y `python src/rag_ecomarket.py` (bucle de chat interactivo). `python src/rag_ecomarket.py --pregunta "..." --sin-rag` ejecuta el modo control del experimento (misma pregunta, sin contexto recuperado).

**Detalle técnico** (documentado como suposición, ver sección 3): el modelo LLM es `gemini-3.6-flash` con el patrón de configuración del Taller Práctico #1 (endpoint OpenAI-compatible de Gemini). Si al ejecutar el modelo reporta disponibilidad distinta, se ajusta el nombre en `src/prompts.py` sin impacto en el resto del sistema.

---

## 3. Limitaciones y suposiciones

El enunciado pide detallarlas expresamente:

1. **Corpus simulado y pequeño.** La base de conocimiento (4 documentos, menos de 100 chunks) es una muestra del dominio de EcoMarket, no un sistema de producción. La suposición es que los archivos de `data/` representan lo que el ERP/publico de la empresa expondría con calidad de datos suficiente. Con este volumen, las conclusiones son válidas para el prototipo, no para producción.
2. **Sin conexión en vivo a un backend.** Los pedidos e inventario son estáticos: estados como "EN TRANSITO" no cambian al responder. La suposición didáctica es que la frescura de los datos se manejaría con un pipeline de sincronización (reindexación programada) que no forma parte del alcance del taller.
3. **Umbral de similitud calibrado empíricamente.** El valor de umbral del escalado se calibró con los casos de prueba del notebook. Con otro corpus habría que recalibrarlo (verificación de la Fase 3); en producción una alternativa más robusta es combinar la similaridad con un clasificador de intención o con el propio LLM juez.
4. **Embeddings open-source en recursos gratuitos.** `text-embedding-3-small` o `gemini-embedding` obtienen mejores puntajes en benchmarks (Fase 1); se asume que con el corpus pequeño la calidad de recuperación es suficiente, y esto se observa en los resultados de los casos. Si EcoMarket creciera a decenas de miles de documentos con ambigüedad alta, la conversación sobre cambiar ese componente se reabriría.
5. **Consultas monovuelta (sin memoria de conversación).** El asistente responde cada pregunta de forma independiente; no mantiene el hilo de un diálogo ni pide aclaraciones adicionales. Añadir memoria sería un paso natural de mejora (y cabe en la conversación de agentes de la unidad 3).
6. **Idioma:** el corpus está en español y las pruebas se ejecutan en español. El modelo de embeddings es multilingüe, así que consultas en otros idiomas no se romperían, pero no fueron un criterio de diseño.
7. **Recursos del equipo:** el computador local del equipo no tiene GPU y el disco está limitado, motivo por el que la ejecución se realiza en Colab. La solución free completa evita: (a) descargar modelos locales grandes (Ollama), (b) servicios de pago (Ollama Cloud, APIs de embeddings) — suposición: la capa gratuita de Gemini es suficiente para el volumen de pruebas del taller.

---

## 4. Escalabilidad y riesgos éticos (breve reflexión)

**Escalabilidad.** El diseño escala razonablemente de prototipo a piloto: ChromaDB soporta millones de vectores en un solo servidor y el pipeline de indexación es agnóstico del volumen. Los puntos de quiebre identificables son dos: (a) al crecer a documentos heterogéneos en miles/diecenas de miles, la búsqueda por similitud pura gana ruido y conviene pasar a búsqueda híbrida (semántica + palabras clave, que Weaviate ofrece self-hosted) o a un servicio gestionado (Pinecone), evaluando costo como en la Fase 1; (b) la frescura de datos (pedidos, stock) exige sincronización continua con el ERP, porque un RAG con datos viejos es peor que sin RAG: responde con confianza a preguntas cuyo estado real ya cambió.

**Riesgos éticos** (continuando la Fase 2 del Taller Práctico #1):

- **Alucinación residual:** el RAG la reduce drásticamente (el contexto viene de la base de conocimiento y el umbral la bloquea cuando no hay material), pero no la elimina: siempre un LLM puede reformular mal lo recuperado. Mitigación: restricciones del prompt ("responde solo con lo indicado en el contexto..."), casos de prueba, y escalamiento a humano.
- **Privacidad:** el corpus incluye pedidos con nombres de clientes. En el prototipo el acceso está controlado por el número de seguimiento; en producción, los chunks pedidos deben consultarse solo tras autenticación del cliente, y los embeddings no viajan a servicios externos (decisión de la Fase 1). Tratamiento según Ley 1581 de 2012.
- **Transparencia con el cliente:** el cliente debe saber que conversa con un asistente virtual y cuando este no puede resolver su caso el sistema lo dice y ofrece a un asesor humano — no hay "falso asesor".
- **Impacto laboral:** el asistente absorbe las consultas repetitivas (el 80% del planteamiento del caso) y redirige a las asesores humanos hacia casos de mayor valor; la automatización empodera al equipo de soporte, no lo reemplaza.
- **Sesgo:** el corpus de FAQ y política define qué puede responder el asistente; si esa base refleja solo a ciertos tipos de clientes (por ejemplo, sin respuestas para personas con discapacidad visual), el servicio queda desbalanceado. La curaduría de la base de conocimiento (esta fase) es también una decisión ética.