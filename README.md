# 🌌 Gemini Embedding 2: Motor de Búsqueda Multimodal & Guía de Estudio

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Google GenAI SDK](https://img.shields.io/badge/Google%20GenAI-v2.34-green.svg)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **Proyecto de estudio y portafolio técnico** basado en el tutorial oficial de DataCamp: *["Gemini Embedding 2: A Guide With Python Examples"](https://www.datacamp.com/tutorial/gemini-embedding-2-guide)* (por Vaibhav Mehra).
> 
> Este repositorio implementa una solución completa y modular que sirve tanto como **referencia teórica exhaustiva** sobre embeddings multimodales nativos, como una **aplicación práctica ejecutable** con base de datos vectorial local (ChromaDB), interfaz CLI enriquecida, cuaderno Jupyter interactivo y suite de pruebas unitarias.

---

## 📑 Tabla de Contenidos

- [1. Visión General y Arquitectura](#1-visión-general-y-arquitectura)
- [2. Guía Teórica de Estudio](#2-guía-teórica-de-estudio)
  - [¿Qué es un Embedding?](#qué-es-un-embedding)
  - [Similitud Coseno: La Métrica Fundamental](#similitud-coseno-la-métrica-fundamental)
  - [Modelos Tradicionales vs. Multimodalidad Nativa](#modelos-tradicionales-vs-multimodalidad-nativa)
  - [Matryoshka Representation Learning (MRL)](#matryoshka-representation-learning-mrl)
  - [Prefijos de Tarea (*Task Prefixes*)](#prefijos-de-tarea-task-prefixes)
  - [La Trampa Crítica: Vectores Fusionados (*Merged*) vs. Separados](#la-trampa-crítica-vectores-fusionados-merged-vs-separados)
- [3. Estructura del Proyecto](#3-estructura-del-proyecto)
- [4. Instalación y Puesta en Marcha](#4-instalación-y-puesta-en-marcha)
- [5. Guía de Uso de la CLI](#5-guía-de-uso-de-la-cli)
- [6. Cuaderno Interactivo de Aprendizaje](#6-cuaderno-interactivo-de-aprendizaje)
- [7. Pruebas Automatizadas y Calidad de Código](#7-pruebas-automatizadas-y-calidad-de-código)
- [8. Limitaciones del Modelo](#8-limitaciones-del-modelo)
- [9. Casos de Uso en Producción](#9-casos-de-uso-en-producción)
- [10. Estándar de Buenas Prácticas: Conventional Commits](#10-estándar-de-buenas-prácticas-conventional-commits)

---

## 1. Visión General y Arquitectura

**Gemini Embedding 2** (`gemini-embedding-2` / `gemini-embedding-2-preview`) es el primer modelo de embeddings de Google capaz de procesar de manera nativa **texto, imágenes, audio, video y documentos PDF**, mapeándolos a un **único espacio geométrico común**.

```mermaid
flowchart TD
    subgraph Fuentes de Datos Multimodales
        T["Texto (Consultas / Párrafos)"]
        I["Imágenes (PNG, JPEG)"]
        A["Audio (WAV, MP3)"]
        D["Documentos (PDF)"]
        M["Intercalado (Texto + Imagen)"]
    end

    subgraph Procesamiento con Gemini Embedding 2
        PRE["Formateador de Prefijos de Tarea (Task Prefixes)"]
        API["Google GenAI Client (gemini-embedding-2)"]
        MRL["Matryoshka Truncation (768 dims) + L2 Auto-Norm"]
    end

    subgraph Base de Datos Vectorial Persistente
        CHROMA[("ChromaDB Persistent Store\nhnsw:space = cosine")]
    end

    subgraph Recuperación Cruzada (Cross-Modal Search)
        Q1["Texto ➔ Imagen"]
        Q2["Imagen ➔ Imagen (Búsqueda Visual)"]
        Q3["Texto ➔ Audio (Recuperación Sonora)"]
        UI["CLI (Rich) & Jupyter Notebook"]
    end

    T --> PRE --> API
    I --> API
    A --> API
    D --> API
    M --> API
    API --> MRL --> CHROMA

    CHROMA --> Q1 --> UI
    CHROMA --> Q2 --> UI
    CHROMA --> Q3 --> UI
```

---

## 2. Guía Teórica de Estudio

### ¿Qué es un Embedding?
Un **embedding** es una proyección matemática que transforma información no estructurada (palabras, píxeles, frecuencias de audio) en un vector denso de números reales:

$$\mathbf{v} = [x_1, x_2, \dots, x_d] \in \mathbb{R}^d$$

En este espacio geométrico, la distancia o el ángulo entre vectores codifica el **significado semántico**. Dos conceptos relacionados se ubican cerca entre sí, mientras que conceptos disímiles quedan lejanos.

### Similitud Coseno: La Métrica Fundamental
Para comparar dos vectores $\mathbf{u}$ y $\mathbf{v}$, se calcula el coseno del ángulo formado entre ellos:

$$\text{Similitud Coseno}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \\mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

- **$+1.0$**: Vectores colineales con idéntica orientación (máxima afinidad semántica).
- **$0.0$**: Vectores perpendiculares/ortogonales (sin relación semántica detectable).
- **$-1.0$**: Vectores con orientaciones opuestas.

ChromaDB calcula internamente la **distancia coseno**:

$$\text{Distancia Coseno} = 1 - \text{Similitud Coseno}$$

Por lo tanto, la similitud semántica reportada por nuestra aplicación se obtiene como:

$$\text{Similitud} = 1.0 - \text{Distancia}$$

> [!CAUTION]
> **Peligro en ChromaDB:** Por defecto, ChromaDB utiliza la distancia euclidiana L2 al inicializar colecciones. Evaluar embeddings normalizados de Gemini con L2 distorsiona la ordenación de resultados. Es **obligatorio** configurar `metadata={"hnsw:space": "cosine"}`.

### Modelos Tradicionales vs. Multimodalidad Nativa
Antes de Gemini Embedding 2, los sistemas multimodales se construían conectando múltiples modelos desacoplados (*Two-Tower Encoders* tipo CLIP para imagen-texto, y modelos de transcripción como Whisper para procesar audio).

| Característica | Enfoque Clásico (*Two-Tower* / Stitched) | Gemini Embedding 2 |
| :--- | :--- | :--- |
| **Arquitectura** | 2 o más modelos independientes acoplados con pérdidas contrastivas. | **Un único modelo transformer** nativamente multimodal. |
| **Modalidades** | Típicamente solo Texto e Imagen. | **Texto, Imagen, Audio, Video y PDFs** (más de 100 idiomas). |
| **Entradas Intercaladas** | No soportado directamente (requiere promedios manuales). | **Soporta texto + imagen juntos** en un solo vector conjunto. |
| **Espacio de salida** | Espacios vectoriales dispares o débilmente alineados. | **Mismo espacio latente unificado.** |

### Matryoshka Representation Learning (MRL)
Por defecto, Gemini Embedding 2 genera vectores de **3.072 dimensiones**. Almacenar e indexar 3.072 números float32 por cada elemento en catálogos de millones de ítems representa un costo masivo de memoria y almacenamiento.

Para solucionarlo, el modelo fue entrenado mediante **MRL** (*Matryoshka Representation Learning*), análogo a las muñecas rusas:
- El modelo es forzado durante el entrenamiento a concentrar la mayor parte de la señal semántica en los primeros coeficientes del vector.
- Al solicitar `output_dimensionality = 768` (o `1536`), el modelo recorta el vector y **lo normaliza automáticamente a norma unitaria** ($L_2$).
- **Ahorro:** Truncar de 3.072 a 768 dimensiones reduce el consumo de disco y RAM en un **75%** conservando prácticamente toda la precisión de búsqueda.

### Prefijos de Tarea (*Task Prefixes*)
El modelo anterior (`gemini-embedding-001`) utilizaba el parámetro `task_type`. Gemini Embedding 2 **deprecó dicho parámetro** y adopta una sintaxis de **prefijos explícitos en el texto**:

| Tipo de Tarea | Uso Principal | Formato de Consulta (*Query*) | Formato de Documento (*Document*) |
| :--- | :--- | :--- | :--- |
| **Búsqueda (Asimétrica)** | Búsqueda semántica en corpus | `task: search result \| query: {texto}` | `title: {titulo} \| text: {texto}` |
| **Preguntas y Respuestas** | Sistemas RAG y preguntas directas | `task: question answering \| query: {texto}` | `title: {titulo} \| text: {texto}` |
| **Verificación de Hechos** | Fact-checking | `task: fact checking \| query: {texto}` | `title: {titulo} \| text: {texto}` |
| **Recuperación de Código** | Code search | `task: code retrieval \| query: {texto}` | `title: {titulo} \| text: {texto}` |
| **Clasificación (Simétrica)** | Asignación de categorías | `task: classification \| query: {texto}` | Idéntico a la consulta |
| **Clustering (Simétrica)** | Agrupamiento no supervisado | `task: clustering \| query: {texto}` | Idéntico a la consulta |
| **Similitud de Oraciones** | Comparación directa de textos | `task: sentence similarity \| query: {texto}` | Idéntico a la consulta |

> [!IMPORTANT]
> **Reglas Críticas de Google:**
> 1. Si un documento indexado no tiene título formal, utiliza obligatoriamente `title: none | text: {contenido}`.
> 2. **NO apliques prefijos de tarea** al componente textual de una entrada multimodal intercalada (por ejemplo, descripción + imagen).

### La Trampa Crítica: Vectores Fusionados (*Merged*) vs. Separados
Este es el error más recurrente al migrar desde otros modelos:
- **`contents=[parte1, parte2]`**: Retorna **1 único vector fusionado** que representa la intersección multimodal de ambas partes.
- **`contents=[Content(parts=[p1]), Content(parts=[p2])]`**: Retorna **un vector individual por cada Content**.

Si deseas vectorizar un lote (*batch*) de archivos o textos independientes, **debes envolver cada uno en su propio objeto `types.Content`**.

---

## 3. Estructura del Proyecto

```
gemini-embedding/
├── .env.example                     # Plantilla de variables de entorno
├── .gitignore                       # Filtros de exclusión para Git
├── .python-version                  # Versión de Python fijada (3.12)
├── pyproject.toml                   # Especificación de proyecto, dependencias y scripts con uv
├── README.md                        # Esta guía de estudio y documentación técnica
├── main.py                          # Punto de entrada directo para la CLI
├── media/                           # Directorio de medios multimedia
│   ├── query.jpg                    # Imagen de consulta para búsqueda visual
│   ├── car.png                      # Muestra sintética: Automóvil rojo
│   ├── dog.png                      # Muestra sintética: Perro en el parque
│   ├── fruit.png                    # Muestra sintética: Frutas frescas
│   ├── laptop.png                   # Muestra sintética: Código en pantalla
│   ├── piano.png                    # Muestra sintética: Teclado de piano
│   ├── rain.png                     # Muestra sintética: Lluvia y nubes
│   ├── piano_melody.wav             # Audio sintetizado con arpegios de piano
│   └── rain_storm.wav               # Audio sintetizado con sonido de lluvia
├── notebooks/
│   └── gemini_embedding_2_tutorial.ipynb  # Cuaderno Jupyter interactivo paso a paso
├── src/
│   └── gemini_embedding/
│       ├── __init__.py              # Exportaciones del paquete
│       ├── config.py                # Configuración validada con Pydantic Settings
│       ├── embeddings.py            # SDK de Gemini, MRL, prefijos y lotes
│       ├── vector_store.py          # Wrapper de ChromaDB con HNSW coseno
│       ├── indexer.py               # Indexador por lotes con rate-limit defensivo
│       ├── search_engine.py         # Orquestador de búsquedas cruzadas
│       ├── sample_media.py          # Generador autónomo de imágenes y audios sintéticos
│       ├── visualizer.py            # Renderizado gráfico (Matplotlib) y tablas (Rich)
│       └── cli.py                   # Interfaz de línea de comandos con Typer
└── tests/
    ├── __init__.py
    ├── test_config.py               # Pruebas de configuración y dimensiones
    ├── test_embeddings.py           # Pruebas unitarias de prefijos y mocks del SDK
    ├── test_vector_store.py         # Pruebas de persistencia y métrica coseno en ChromaDB
    ├── test_indexer.py              # Pruebas de indexación y búsqueda integrada
    └── test_sample_media.py         # Validación de integridad de los medios generados
```

---

## 4. Instalación y Puesta en Marcha

### Prerrequisitos
- **Linux / macOS / Windows (WSL recomendado)**
- **Python >= 3.12**
- **uv** instalado (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Una **API Key de Google Gemini** (gratuita en [Google AI Studio](https://aistudio.google.com/))

### Pasos de Instalación

1. **Navegar al directorio del proyecto:**
   ```bash
   cd gemini-embedding
   ```

2. **Sincronizar el entorno virtual con `uv`:**
   ```bash
   uv sync
   ```
   *Esto instalará todas las dependencias necesarias en un entorno virtual aislado `.venv`.*

3. **Configurar la clave de API:**
   Copia el archivo de ejemplo y agrega tu clave de Gemini:
   ```bash
   cp .env.example .env
   ```
   Edita `.env`:
   ```env
   GEMINI_API_KEY="AIzaSyTuClaveDeAPI..."
   GEMINI_EMBEDDING_MODEL="gemini-embedding-2"
   EMBEDDING_DIMENSIONS="768"
   CHROMA_PERSIST_DIRECTORY="./chroma_db"
   ```

---

## 5. Guía de Uso de la CLI

El proyecto incluye comandos interactivos mediante Typer y Rich. Puedes ejecutarlos con `uv run python main.py <comando>` o directamente con `uv run gemini-search <comando>`.

### 1. Generar Medios Sintéticos de Prueba
Crea 6 imágenes PNG con diferentes categorías, 2 clips de audio WAV reproducibles y la imagen de consulta `query.jpg`:
```bash
uv run python main.py samples
```

### 2. Indexar la Carpeta Multimedia en ChromaDB
Vectoriza todos los archivos de `./media/` calculando sus embeddings multimodales con Gemini y guardándolos en ChromaDB:
```bash
uv run python main.py index
```

### 3. Consultar Información de la Base de Datos
Verifica la cantidad de vectores indexados, el modelo y la métrica de distancia:
```bash
uv run python main.py info
```

### 4. Búsqueda Cruzada: Texto ➔ Imagen
Busca fotos usando descripciones en lenguaje natural:
```bash
uv run python main.py search "a red sports car on street" --kind image
```
*(Opcional: agrega `--plot` para abrir la cuadrícula visual de Matplotlib con las miniaturas recuperadas).*

### 5. Búsqueda Visual: Imagen ➔ Imagen
Utiliza una imagen como consulta directa para encontrar imágenes análogas:
```bash
uv run python main.py search-image query.jpg --plot
```

### 6. Búsqueda Cruzada: Texto ➔ Audio
Recupera clips de audio describiendo sonoridades:
```bash
uv run python main.py search "someone playing the piano notes" --kind audio
uv run python main.py search "gentle rainfall and storm" --kind audio
```

---

## 6. Cuaderno Interactivo de Aprendizaje

Para experimentar de forma interactiva con explicaciones detalladas y reproductores de audio embebidos:

```bash
uv run jupyter notebook notebooks/gemini_embedding_2_tutorial.ipynb
```

El cuaderno recorre los 7 pasos del tutorial de DataCamp:
1. Inspección dimensional de embeddings (3.072 vs. 768 MRL).
2. Formateo asimétrico de consultas y documentos.
3. Prueba visual de entradas fusionadas (*merged*) vs. independientes.
4. Construcción del índice ChromaDB con distancia coseno.
5. Comparativa de precisión con y sin prefijo de tarea.
6. Búsqueda visual de fotos con `query.jpg`.
7. Recuperación de sonido y reproducción mediante `IPython.display.Audio`.

---

## 7. Pruebas Automatizadas y Calidad de Código

### Ejecución de Pruebas Unitarias con Pytest
La suite de pruebas utiliza `unittest.mock` para simular las respuestas de la API de Google GenAI, por lo que **se ejecutan al 100% de forma local, rápida y sin consumir cuota de tu API Key**:

```bash
uv run pytest -v
```

### Linter y Formato con Ruff
Garantiza el cumplimiento de los estándares PEP 8 e importaciones limpias:

```bash
uv run ruff check .
uv run ruff format --check .
```

---

## 8. Limitaciones del Modelo

A pesar de ser un avance notable, Gemini Embedding 2 presenta restricciones importantes a considerar en arquitectura de software:

1. **Límites de Entrada Cortos:**
   - **Video:** Máximo 120 segundos (muestreado a máximo 32 fotogramas).
   - **Audio:** Máximo 180 segundos.
   - **PDFs:** Máximo 6 páginas por petición (cada página consume 258 tokens para su codificación visual).
2. **Truncado Silencioso:** Si una petición sobrepasa los **8.192 tokens**, el exceso de contenido se descarta silenciosamente sin generar una excepción. Es vital validar el conteo de tokens en documentos extensos.
3. **El Video Ignora la Pista de Sonido:** El encoder de video procesa únicamente la pista visual. Si el audio del video es relevante, debe extraerse y embedarse por separado.
4. **Espacios Incompatibles:** Los vectores generados por `gemini-embedding-001` y `gemini-embedding-2` viven en espacios latentes distintos. Migrar requiere reindexar la base de datos completa.
5. **Servicio Propietario vía API:** El modelo no es de pesos abiertos (*open weights*). Toda la información viaja a través de los servidores de Google Cloud.

---

## 9. Casos de Uso en Producción

- **Multimodal RAG (Retrieval-Augmented Generation):** Indexar y recuperar directamente diapositivas, capturas de pantalla, gráficos financieros y PDFs escaneados sin necesidad de pipelines previos de OCR.
- **Búsqueda Visual de Catálogo (Visual Commerce):** Permitir a los usuarios subir una foto de una prenda o producto para consultar artículos equivalentes en el inventario.
- **Búsqueda Semántica en Audio y Podcasts:** Localizar pasajes específicos de audio describiendo el contenido temático o sonoro sin requerir transcripción texto-a-texto previa.
- **Clasificación Unificada de Tickets:** Categorizar reclamos de soporte que contienen texto mezclado con capturas de pantalla de error en una sola operación vectorial.

---

## 10. Estándar de Buenas Prácticas: Conventional Commits

Este proyecto adopta rigurosamente el estándar [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/). Cada mensaje de confirmación describe con claridad la intención y el impacto de los cambios.

### Estructura Canónica
```
<tipo>[ámbito opcional]: <descripción en presente>

[cuerpo explicativo opcional]

[pie de página opcional, e.g., BREAKING CHANGE o referencias a issues]
```

### Tipos Utilizados en este Proyecto

| Tipo | Propósito | Ejemplo en este Proyecto |
| :--- | :--- | :--- |
| `feat` | Incorporación de una nueva funcionalidad | `feat(embeddings): implement task prefix formatting and MRL truncation` |
| `fix` | Corrección de un fallo o bug | `fix(vector_store): ensure hnsw:space is explicitly set to cosine` |
| `docs` | Cambios exclusivos en documentación | `docs: add comprehensive multimodal search guide and study README` |
| `test` | Incorporación o refactorización de pruebas | `test(indexer): add unit tests with mocked GenAI client` |
| `refactor` | Cambio de código que no altera el comportamiento externo | `refactor(cli): modularize search command logic and table formatting` |
| `chore` | Tareas de mantenimiento, dependencias o configuración | `chore: configure pyproject.toml with uv dependencies and ruff` |

---

## 👨‍💻 Autor

Desarrollado por **Juan Jose** como proyecto de estudio avanzado y referencia práctica para ingeniería de Inteligencia Artificial y embeddings multimodales.
