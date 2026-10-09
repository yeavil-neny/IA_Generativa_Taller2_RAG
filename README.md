# Taller Práctico #2 — Sistema RAG para la atención al cliente de EcoMarket

[![Abrir en Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeavil-neny/IA_Generativa_Taller2_RAG/blob/main/notebooks/Taller2_RAG_EcoMarket.ipynb)

**Curso:** IA Generativa — Unidad 3: Automatización de Tareas con IA Generativa
**Caso de estudio:** EcoMarket, e-commerce de productos sostenibles
**Equipo:** Cristian Humberto Yepes Marin y Yenny Alexandra Villarreal Suaza

---

## ¿Qué es esta solución?

En el [Taller Práctico #1](https://github.com/yeavil-neny/IA_Generativa) el asistente de atención al cliente de EcoMarket resolvía dos flujos definidos (estado de pedido y devolución) mediante plantillas de prompt y un mini-RAG didáctico: el código inyectaba directamente el JSON del pedido o la política como contexto.

En este Taller Práctico #2 se extiende ese mismo asistente con un **sistema RAG completo** (Retrieval Augmented Generation) para que pueda responder **cualquier tipo de solicitud** sobre la empresa, o **comunicar con honestidad** que no tiene las herramientas para atenderla y ofrecer escalar a un asesor humano — el comportamiento exacto que exige el enunciado.

## Arquitectura (resumen)

| Componente | Elección | Justificación (detalle en la Fase 1) |
|---|---|---|
| Modelo de embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (Hugging Face, open-source) | Buena calidad semántica en español, costo cero, la vectorización ocurre dentro del entorno (privacidad de datos de clientes, Ley 1581) |
| Base de datos vectorial | ChromaDB (open-source, persistente, similitud coseno) | Sin costo ni operación, volumen acorde al caso, datos bajo control; Pinecone/Weaviate se evalúan y se descartan para esta escala |
| Framework | LangChain (guiado por el script de referencia `rag_ejemplo.py`) | Referencia explícita del enunciado; componentes intercambiables |
| LLM generador | Gemini (capa gratuita) vía endpoint compatible con OpenAI | El mismo modelo del Taller Práctico #1; solución 100% gratuita |
| Recuperador y umbral | `similarity_search_with_score` + umbral de distancia coseno | Permite verificar la pertinencia antes de generar: si nada supera el umbral, respuesta honesta de escalamiento sin llamar al LLM |

## Las tres fases (documentación)

| Fase | Documento | Contenido |
|---|---|---|
| Fase 1 | [`docs/fase1-componentes-rag.md`](docs/fase1-componentes-rag.md) | Selección y justificación del modelo de embeddings y de la base de datos vectorial |
| Fase 2 | [`docs/fase2-base-conocimiento.md`](docs/fase2-base-conocimiento.md) | Los 4 tipos de documentos, estrategia de segmentación (chunking) justificada, proceso de indexación |
| Fase 3 | [`docs/fase3-arquitectura-y-limitaciones.md`](docs/fase3-arquitectura-y-limitaciones.md) | Arquitectura del flujo, adaptaciones sobre `rag_ejemplo.py`, limitaciones y suposiciones, reflexión de escalabilidad y riesgos éticos |

## Cómo ejecutar

### Opción recomendada: Google Colab

1. Abrir el notebook con el botón **Abrir en Colab** (arriba de este README).
2. Ejecutar las celdas en orden; el notebook clona este repositorio, instala dependencias y pide la clave API de Gemini (se ingresa por teclado, oculta, y no queda registrada en el notebook ni en el repositorio).
3. Los casos de prueba de la sección 7 producen la evidencia: cada respuesta muestra las fuentes recuperadas con su distancia coseno.

### Local

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env    # y pegar la clave GEMINI_API_KEY en .env
python src/rag_ecomarket.py                                  # bucle de chat interactivo
python src/rag_ecomarket.py --pregunta "¿Cuánto tarda el reembolso?"   # consulta puntual
python src/rag_ecomarket.py --pregunta "¿Cuánto tarda el reembolso?" --sin-rag  # modo control
```

## Estructura del repositorio

```
├── README.md                     esta página
├── requirements.txt              dependencias (Colab y local)
├── .env.example                  plantilla de configuración local (sin claves reales)
├── data/                         base de conocimiento de EcoMarket (Fase 2)
│   ├── politicas_devolucion.md   política de devoluciones (por secciones)
│   ├── pedidos.json              registro de 12 pedidos
│   ├── inventario_productos.csv  catálogo de 18 productos
│   └── faqs.json                 12 preguntas frecuentes
├── src/
│   ├── prompts.py                plantillas (Rol -> Contexto -> Instrucciones -> Restricciones)
│   └── rag_ecomarket.py          pipeline RAG completo + CLI
├── notebooks/
│   └── Taller2_RAG_EcoMarket.ipynb   solución documentada y ejecutable en Colab
└── docs/                         documentación de las tres fases
```

## Casos de prueba (evidencia de la Fase 3)

| Caso | Consulta | Comportamiento esperado |
|---|---|---|
| 1 | "Quiero devolver el café orgánico molido que compré. ¿Puedo?" | Cita la política: alimentos/bebidas no se devuelven, con empatía y la excepción por defecto de fábrica |
| 2 | "¿Cuánto tarda el reembolso de una devolución?" | Cita la FAQ: 5 días hábiles desde la llegada del producto |
| 3 | "¿Qué pasa con mi pedido ECO-10003? El envío lleva muchos días." | Disculpa + motivo real (interrupción logística Cali-Medellín) + nueva fecha + enlace de rastreo |
| 4 | "¿A qué precio está el kit de compostaje de cocina y está disponible?" | Cita el inventario: $58.000 COP, disponibilidad |
| 5 | "¿Cuál es el clima en Bogotá para el fin de semana?" | Ningún chunk supera el umbral de pertinencia: mensaje determinista de "no tengo las herramientas" + oferta de escalar a asesor humano |
| 6 (control) | Caso 1 sin RAG | El modelo responde con conocimiento general y no puede citar la política: alucinación con apariencia de respuesta oficial |

Los resultados ejecutados quedan documentados en el notebook (secciones 7 y 8).