# Fase 1 — Selección de Componentes Clave del Sistema RAG

**Caso de estudio:** EcoMarket (e-commerce de productos sostenibles) — extender el asistente de atención al cliente construido en el Taller Práctico #1 con un sistema RAG (Retrieval Augmented Generation).

**Equipo:** Cristian Humberto Yepes Marin y Yenny Alexandra Villarreal Suaza

---

## 1. ¿Qué decide esta fase y por qué importa?

En el Taller Práctico #1 el asistente de EcoMarket funcionaba con un *mini-RAG* didáctico: el código buscaba directamente el pedido en un JSON inyectaba su contenido en la plantilla del prompt. Eso resolvió dos flujos específicos (consultar un pedido y evaluar una devolución), pero el enunciado del Taller Práctico #2 exige más: el asistente debe responder **cualquier tipo de solicitud** sobre la empresa, o reconocer que no tiene las herramientas para atenderla.

Para lograrlo se necesita un sistema RAG real, y su eficacia depende de dos decisiones de arquitectura que se toman *antes* de codificar:

1. **Modelo de embeddings**: convierte texto de los documentos en vectores numéricos. Determina *qué tan bien el sistema entiende que una pregunta del cliente ("¿puedo devolver el café?") es semánticamente igual a un pasaje de la política ("los productos perecederos no son elegibles para devolución")*. Un embedding deficiente en español produce recuperaciones irrelevantes, y con contexto irrelevante el LLM alucina o responde mal: el RAG se convierte en un generador de ruido.
2. **Base de datos vectorial**: almacena y consulta esos vectores por similitud. Determina la *escalabilidad, el costo operativo y la latencia* del sistema a medida que la base de conocimiento crece.

La regla usada para ambas decisiones fue la misma que en la Fase 1 del Taller Práctico #1: **la arquitectura debe corresponder al tamaño real del problema y a las restricciones de la empresa, no a la popularidad de la herramienta**.

---

## 2. Modelo de embeddings

### 2.1 Opciones evaluadas

| Opción | Tipo | Idioma español | Costo | Dónde corre | Privacidad |
|---|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-Transformers / Hugging Face) | Código abierto | Bueno: modelo entrenado en más de 50 idiomas con buen desempeño en tareas de similitud semántica multilingüe | Gratuito | Local o en Colab (CPU suficiente para corpus pequeños) | Ningún texto sale del entorno de la empresa |
| `intfloat/multilingual-e5-base` (Hugging Face) | Código abierto | Muy bueno: entrenado específicamente para multilingüismo (MTEB) | Gratuito | Local o en Colab; más pesos que MiniLM (~278M vs ~118M parámetros) | Ningún texto sale del entorno de la empresa |
| OpenAI `text-embedding-3-small` | Propietario | Excelente (MTEB multilingüe superior a los anteriores) | De pago por token (~USD 0.02 / 1M tokens) | API externa | El texto viaja a servidores de OpenAI |
| Google `gemini-embedding` (text-embedding-004) | Propietario | Excelente | Capa gratuita con cuota; después de pago | API externa | El texto viaja a servidores de Google |

*Nota:* las cifras de calidad se referencian al benchmark público **MTEB** (Massive Text Embedding Benchmark), que evalúa recuperación semántica en 10+ idiomas.

### 2.2 Opción seleccionada: `paraphrase-multilingual-MiniLM-L12-v2` (código abierto)

**Razones, en orden de peso para EcoMarket:**

1. **Adecuación al volumen real del problema.** La base de conocimiento de EcoMarket es una política de devoluciones, un catálogo de inventario, un archivo de FAQ y el registro de pedidos: esto produce decenas a cientos de *chunks*, no millones. La diferencia de calidad estadística entre MiniLM y modelos de frontera es relevante en corpora masivos con ambigüedad alta; en un corpus pequeño, curado y específico del dominio, MiniLM recupera el pasaje correcto de forma confiable. Pagar (en dinero o en complejidad) por capacidad que el volumen no explota es mala ingeniería.
2. **Privacidad y protección de datos.** Con un modelo open-source, la vectorización ocurre *dentro* del entorno de ejecución del proyecto (notebook en Colab o servidor propio): el contenido de la base de conocimiento — que incluye nombres de clientes en el registro de pedidos — nunca viaja a un servicio externo solo para calcular embeddings. Esto es coherente con el análisis de privacidad de la Fase 2 del Taller Práctico #1 (Ley 1581 de 2012) y reduce la superficie de riesgo del sistema final.
3. **Costo cero y reproducibilidad.** El equipo no tiene presupuesto para el proyecto y la versión gratuita de una API propietaria impone cuotas que impiden experimentos repetidos (chunking alternativo, calibración de umbrales) — que justamente son la parte científica de esta fase. El modelo se descarga una vez y ejecuta las veces que el experimento requiera.
4. **Español sólido para el caso de uso.** El modelo fue entrenado con datos en más de 50 idiomas, incluido el español, con el objetivo explícito de similitud de pares de oraciones (paraphrase mining), que es exactamente la tarea de un recuperador RAG: medir si una pregunta del cliente "parece" un pasaje de la política.
5. **Costo computacional manejable.** Es el modelo más ligero de la comparación (~118M parámetros, vectores de 384 dimensiones), corre en CPU en los recursos gratuitos de Colab, y su propia limitación (contexto máximo de entrada corto, 128 tokens) encaja bien con la estrategia de segmentación elegida en la Fase 2 (chunks cortos y semánticamente autónomos).

**Qué se gana con alternativas y por qué no se eligieron:** `multilingual-e5-base` obtiene puntajes MTEB algo superiores, pero duplica el costo de cómputo por consulta a cambio de una mejora que un corpus pequeño no explota. Las APIs propietarias suben el techo de calidad y quitan al equipo la carga de descargar pesos, pero introducen costo variable, cuotas para experimentar y el traslado de datos de clientes fuera del entorno de la empresa. Sobre las APIs propietarias se discute el mismo criterio que en el Taller Práctico #1 (sección 6): tienen sentido cuando la escala justifica su costo, y el punto de equilibrio se vuelve a evaluar si EcoMarket crece órdenes de magnitud.

**Nota sobre el límite de 128 tokens:** este detalle no es menor — dicta la estrategia de segmentación. Si los chunks fueran de 1000 caracteres (como los del script de referencia `rag_ejemplo.py`, diseñado para `llama2`), el modelo truncaría silenciosamente el final de cada chunk y la recuperación se degradaría. Por eso en la Fase 2 el tamaño de chunk se calibra alrededor del contexto del modelo elegido. Este es un ejemplo concreto de cómo una elección de componente **condiciona** la siguiente decisión del sistema.

---

## 3. Base de datos vectorial

### 3.1 Opciones evaluadas

| Criterio | ChromaDB | Pinecone | Weaviate |
|---|---|---|---|
| Tipo | Código abierto, librería/local | Servicio gestionado (SaaS) | Código abierto, self-hosted o Cloud |
| Costo | Gratuito | Gratuito con límites; de pago al crecer | Gratuito self-hosted (con costo de operación); Cloud de pago |
| Configuración | `pip install chromadb`, lista en segundos | Cuenta y claves de API | Docker/orquestación (self-hosted) o cuenta cloud |
| Escalabilidad | Millones de vectores en un solo servidor; no distribuido | Escala elástico y gestionado | Escala bien self-hosted si se opera bien |
| Persistencia | Archivos en disco (`persist_directory`) | Gestionada | Gestionada (self-hosted la propia base) |
| Índices de similitud | HNSW; espacio L2/cosíneo/IP | Gestionado | HNSW; más opciones (BM25 híbrido, modul­os) |
| Carga operativa (MLOps) | Nula en prototipo; manual en producción | Nula (el proveedor la asume) | Alta en self-hosted; media en Cloud |

### 3.2 Opción seleccionada: ChromaDB (código abierto)

**Razones para EcoMarket:**

1. **El volumen no justifica un servicio gestionado.** Con decenas-cientos de chunks, la ventaja diferencial de Pinecone (distribución automática, réplicas multi-región) es un servicio que EcoMarket no consume todavía. La rúbrica de costo es explícita: pagar gestionamiento antes de necesitarlo es costo sin retorno.
2. **Costo cero y cero fricción en el prototipo:** se instala con pip, corre en memoria o persiste en disco en la misma carpeta del proyecto, y funciona igual en Colab, laptop o servidor. La barra para reproducir el experimento del notebook es la más baja posible: sin cuentas, sin claves, sin red.
3. **Datos bajo control:** igual que con los embeddings, la base de conocimiento (incluidos pedidos con nombres de clientes) vive en el entorno del proyecto, no en un servidor de terceros — coherente con la postura de privacidad definida en el Taller Práctico #1.
4. **Integración idiomática con LangChain** (framework elegido para la Fase 3, guiado por el script de referencia): la clase `Chroma` es una opción documentada del ecosistema, con soporte de `similarity_search_with_score`, lo que permite implementar la lógica de escalado honesto ("no tengo herramientas para esto") leyendo la distancia de cada resultado.

**Por qué no Pinecone:** es la mejor opción cuando la escala justifica (millones de vectores, picos de tráfico, multi-región). Para EcoMarket hoy, añadiría una cuenta externa, cuotas de capa gratuita y datos de clientes en un tercero, todo para resolver un problema de escala que no existe. Queda registrada como ruta de migración si el negocio crece (la hoja de ruta de "migrar cuando el volumen lo exija" es la misma lógica de la sección 6 de la Fase 1 del Taller Práctico #1).

**Por qué no Weaviate (self-hosted):** ofrece capacidades superiores (búsqueda híbrida con BM25, módulos de desnormalización) que el caso no necesita, a cambio de asumir la operación de un servicio (Docker, backups, actualizaciones) — exactamente la carga de MLOps que en el Taller Práctico #1 se identificó como el costo real del autohospedaje. Weaviate es un candidato fuerte para una versión de producción con volúmenes medios, pero excede el prototipo.

---

## 4. Resumen de la arquitectura decidida

| Componente | Decisión | Decisores alternativos descartados | Factor decisivo |
|---|---|---|---|
| Modelo de embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (Hugging Face, open-source) | e5-base (open-source, más pesado), OpenAI/Google (propietarios, cuotas y datos fuera) | Corpus pequeño + costo cero + privacidad + calidad suficiente en español |
| Base de datos vectorial | ChromaDB (open-source, persistente en disco) | Pinecone (gestionado, de pago al crecer), Weaviate (self-hosted con carga MLOps) | Volumen pequeño + costo cero + simplicidad + datos bajo control |
| Framework de integración | LangChain (siguiendo el script de referencia `rag_ejemplo.py`) | LlamaIndex (usado en el notebook de referencia de clase) | Referencia explícita del enunciado y continuidad con el desarrollo del Taller Práctico #1 |
| LLM generador | Gemini en su capa gratuita, vía endpoint compatible con OpenAI (el mismo modelo del Taller Práctico #1) | Ollama local (disco/CPU del equipo del equipo), Ollama Cloud (licencia de pago) | Continuidad con el Taller Práctico #1 y solución 100% gratuita |

**Principio transversal:** cada componente es open-source, gratuito y autocontenido, para conservar la privacidad de los datos de la empresa y la reproducibilidad del experimento; el único servicio externo es el LLM en su capa gratuita, justificado con el mismo argumento de costo/beneficio presentado en la Fase 1 del Taller Práctico #1.