# Búsqueda multimodal con Gemini Embedding 2

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Motor de búsqueda que encuentra **imágenes y audios a partir de texto** (y fotos a partir de otra foto) usando [Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2), el modelo de embeddings nativamente multimodal de Google, y [ChromaDB](https://www.trychroma.com/) como base de datos vectorial local.

```bash
$ uv run python main.py search "someone playing the piano" --kind audio
# → piano_melody.wav aparece primero, sin etiquetas ni transcripciones de por medio
```

Incluye una CLI, un cuaderno Jupyter paso a paso y pruebas que corren sin consumir la API.

## Qué demuestra

- **Búsqueda cruzada entre modalidades:** texto → imagen, texto → audio e imagen → imagen sobre el mismo índice, porque el modelo pone todo en un único espacio vectorial.
- **Uso correcto del modelo:** prefijos de tarea en el texto (este modelo no acepta `task_type`), dimensiones reducidas con MRL y la diferencia entre un vector *fusionado* y vectores *separados* (ver abajo).
- **ChromaDB bien configurado:** distancia coseno en lugar de la L2 por defecto, e IDs estables para que reindexar no duplique datos.
- **Funciona sin preparar nada:** `samples` genera imágenes y audios sintéticos, así que la demo corre sin descargar datasets.

## Inicio rápido

Requisitos: Python 3.12+, [uv](https://docs.astral.sh/uv/) y una API key gratuita de [Google AI Studio](https://aistudio.google.com/).

```bash
uv sync
cp .env.example .env               # pon tu GEMINI_API_KEY

uv run python main.py samples      # genera 6 imágenes, 2 audios y query.jpg
uv run python main.py index        # calcula los embeddings y los guarda en ChromaDB
uv run python main.py search "a red sports car on the street" --kind image --plot
```

### Comandos

| Comando | Qué hace |
| --- | --- |
| `samples` | Genera los medios sintéticos de prueba en `./media` (y `query.jpg` en la raíz). |
| `index` | Indexa todos los archivos soportados de `./media` (`--delay` controla la pausa entre llamadas). |
| `search "<texto>"` | Búsqueda por texto. `--kind image\|audio\|pdf\|video` filtra por tipo, `--no-prefix` desactiva el prefijo de tarea, `--plot` muestra las imágenes. |
| `search-image <ruta>` | Búsqueda visual: usa una imagen como consulta (p. ej. `query.jpg`). |
| `info` | Modelo, dimensiones y número de elementos indexados. |
| `clear` | Vacía la colección. |

También se puede usar `uv run gemini-search <comando>`.

### Configuración (`.env`)

| Variable | Default | Descripción |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Tu clave de Google AI Studio. |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Modelo de embeddings. |
| `EMBEDDING_DIMENSIONS` | `768` | Tamaño del vector: 768, 1536 o 3072. |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_db` | Dónde se guarda el índice. |

## Conceptos clave

### Un solo espacio para todo
Los sistemas multimodales clásicos combinan varios modelos (CLIP para imagen y texto, Whisper para audio) con espacios vectoriales que no siempre coinciden. Gemini Embedding 2 es **un único modelo** que recibe texto, imágenes (PNG/JPEG), audio (MP3/WAV), video (MP4/MOV) y PDFs, y los proyecta al mismo espacio. Por eso un texto puede encontrar directamente un audio.

### Similitud coseno
Dos vectores se comparan por el ángulo entre ellos:

$$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|}$$

ChromaDB usa distancia **L2** por defecto, así que la colección se crea con `{"hnsw:space": "cosine"}`. ChromaDB devuelve distancia coseno y la CLI la muestra como similitud: `1 − distancia`.

### Matryoshka (MRL): vectores más chicos
El modelo devuelve 3072 dimensiones, pero fue entrenado para concentrar la información en las primeras. Con `output_dimensionality=768` el vector se recorta y Google lo **normaliza automáticamente**. El índice ocupa un 75 % menos y la calidad de búsqueda casi no cambia.

### Prefijos de tarea
`gemini-embedding-2` **no acepta el campo `task_type`**: la tarea se escribe dentro del texto, de forma consistente en la indexación y en la consulta.

| Tarea | Consulta | Documento |
| --- | --- | --- |
| Búsqueda (asimétrica) | `task: search result \| query: {texto}` | `title: {título o none} \| text: {texto}` |
| Preguntas y respuestas | `task: question answering \| query: {texto}` | `title: … \| text: …` |
| Clasificación / clustering (simétrica) | `task: classification \| query: {texto}` | igual que la consulta |

En entradas mixtas (texto + imagen) Google **no recomienda** poner prefijo en la parte de texto. El comando `search --no-prefix` permite comparar resultados con y sin prefijo.

### La trampa: vector fusionado vs. vectores separados
- `contents=[texto, imagen]` → **un solo vector** que representa ambos juntos.
- `contents=[Content(parts=[texto]), Content(parts=[imagen])]` → **un vector por elemento**.

Para indexar un lote de archivos independientes hay que envolver cada uno en su propio `types.Content` (`embed_separate_batch`). El cuaderno muestra la diferencia con números.

## Cuaderno interactivo

```bash
uv run jupyter notebook notebooks/gemini_embedding_2_tutorial.ipynb
```

Recorre paso a paso: dimensiones con MRL, prefijos de consulta y documento, vector fusionado vs. separado, construcción del índice, comparación con y sin prefijo, búsqueda visual con `query.jpg` y búsqueda de audio con reproducción integrada.

## Estructura

```
├── main.py                       # entrada de la CLI
├── media/                        # medios de ejemplo (generados con `samples`)
├── query.jpg                     # imagen de consulta para la búsqueda visual
├── notebooks/                    # tutorial en Jupyter
├── src/gemini_embedding/
│   ├── config.py                 # configuración con pydantic-settings
│   ├── embeddings.py             # cliente de Gemini: prefijos, MRL, lotes
│   ├── vector_store.py           # ChromaDB con distancia coseno
│   ├── indexer.py                # indexación de un directorio
│   ├── search_engine.py          # búsqueda por texto y por imagen
│   ├── sample_media.py           # generador de imágenes y audios sintéticos
│   ├── visualizer.py             # tablas (Rich) y cuadrícula de imágenes (Matplotlib)
│   └── cli.py                    # comandos con Typer
└── tests/                        # pytest con el cliente de Gemini simulado
```

## Pruebas y calidad

```bash
uv run pytest -v          # sin red ni API key: el cliente de Gemini está simulado
uv run ruff check .
uv run ruff format --check .
```

## Limitaciones del modelo

- **Por petición:** hasta 8 192 tokens de texto (el exceso se recorta sin avisar), 6 imágenes, 180 s de audio, 120 s de video y 1 PDF de hasta 6 páginas.
- **El video ignora su audio:** si el sonido importa, hay que extraerlo y embeberlo aparte.
- **Espacios incompatibles:** los vectores de `gemini-embedding-001` y `gemini-embedding-2` no se pueden mezclar. Cambiar de modelo obliga a reindexar.
- **Servicio en la nube:** no es un modelo de pesos abiertos; los datos se envían a la API de Google.

## Créditos

Proyecto de estudio basado en el tutorial de DataCamp [*Gemini Embedding 2: A Guide With Python Examples*](https://www.datacamp.com/tutorial/gemini-embedding-2-guide), ampliado con una CLI, pruebas y un índice persistente.

Hecho por Juan José López.
