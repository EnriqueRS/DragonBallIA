# Dragon Ball Scholar 🐉

> Proyecto de **aprendizaje de ingeniería de IA** construyendo, pieza a pieza y
> a mano, un sistema RAG (Retrieval-Augmented Generation) experto en Dragon Ball.
>
> El objetivo no es el producto: es entender cada capa del circuito
> *texto → embedding → índice vectorial → retrieva → prompt → respuesta*.

---

## 🎯 Objetivo y método

Dominar de verdad los conceptos de RAG — embeddings, chunking, índices
vectoriales, retrieva, reranking — con la temática de Dragon Ball (dataset:
~1.571 páginas de personajes de la wiki en español).

**Método**: primero implementar cada pieza *a mano* (para saber qué hace por
dentro) y después migrar a frameworks (para apreciar qué nos ahorran). Cada
fase termina con un experimento medible, no con un "ya funciona".

---

## 📜 Historia: el proyecto anterior y por qué murió

Este repo ya fue un intento de lo mismo (código preservado en el commit
`7ce4a40` de git). Fracasó silenciosamente y la autopsia enseña más que el
código:

| Qué hacía | Por qué falló | Lección |
|---|---|---|
| Indexaba **cada página entera como 1 chunk** truncado a 2.500 caracteres | Un vector no puede representar bien 2.500 chars de temas mezclados: la similitud se diluye | **El chunking es la mitad del éxito de un RAG** (Fase 2) |
| Sin métricas de retrieva | Preguntabas por Monaka y recuperaba a Sidra... y nadie lo sabía hasta leer los logs a mano | **Sin golden set + recall@k no sabes que tu RAG está roto** (Fase 3) |
| Keys hardcodeadas en cada script | Al migrar el servidor, todas las keys murieron dispersas por el código | **Configuración centralizada en `.env`** (Fase 0) |
| Metadatos frágiles ("A veces Prisma devuelve strings") | Dependía de los caprichos del formato de respuesta de un servicio ajeno | **Infra propia y explícita** > cajas negras (Fase 0) |
| Prompt "si no sabes, di No hay datos" | Con contexto irrelevante, el modelo siempre respondía "no hay datos" | El prompt no arregla una mala retrieva |

Síntoma final (`response.txt`, hoy en git): tres preguntas seguidas →
`"I do not have enough information"` ×3.

---

## 🏗️ Arquitectura actual

```
                          ┌─────────────────────────────┐
                          │   NUC (192.168.100.73)      │
                          │                             │
 PREGUNTA ──► embedding ──┼──► Postgres 17 + pgvector   │
              (NAN API)   │    tabla chunks             │
                          │    índice HNSW (coseno)     │
                          └──────────┬──────────────────┘
                                     │ top-k por coseno
                                     ▼
                          prompt con contexto ──► LLM (glm5.3-flash
                                                   vía LiteLLM propio)
                                     │
                                     ▼
                                 RESPUESTA + citas
```

| Pieza | Elección | Por qué |
|---|---|---|
| Embeddings | `qwen3-embedding` @ api.nan.builders, **1024 dims** | Multilingüe fuerte; `dimensions` = truncado matryoshka |
| Rerank (Fase 6) | `rerank` (Qwen3-Reranker-8B) @ NAN | Segunda pasada de precisión, ya verificado con español |
| LLM | `nan/glm5.3-flash` @ litellm.custombots.es | Proxy LiteLLM self-hosted del nuc |
| Vector store | **Postgres + pgvector 0.8.0** en el nuc | Infra propia, SQL puro para aprender; BD `dragonball` dedicada con usuario propio |
| Idioma de todo | Español | El dataset lo es; models de embeddings tienen que lidiarlo |

Nota: en el servidor viejo existía un servicio de vector store en el puerto
8002 que quedó sin levantar tras una migración. En vez de resucitarlo,
construimos la pieza nosotros sobre pgvector — más didáctico y más estable.

---

## 📁 Estructura del repo

| Fichero | Fase | Qué es |
|---|---|---|
| `config.py` | 0 | Configuración central: lee `.env` y expone `EMBEDDINGS`, `LLM`, `PG`, `RERANK` |
| `.env` | 0 | Credenciales (fuera de git, **rotar al cerrar el proyecto**) |
| `smoke_test.py` | 0 | El circuito completo en miniatura: 3 fichas → pgvector → búsqueda → LLM |
| `embeddings.py` | 1 | Cliente de embeddings (individual + por lotes) y `similitud_coseno()` a mano |
| `explora_embeddings.py` | 1 | Seis experimentos guiados sobre embeddings (ver 🎓 Fase 1) |
| `dataset_dragonball/raw_texts/` | — | 1.571 ficheros `.txt` de personajes de la wiki |
| `dataset_dragonball/dragonball_finetuning.jsonl` | — | 6.893 pares instrucción/respuesta (para la Fase 6 opcional) |

---

## 🗺️ Plan de fases

- [x] **Fase 0 — Infraestructura**: BD pgvector + config central + smoke test verde
- [x] **Fase 1 — Embeddings a mano**: coseno a mano, mapa semántico, matryoshka, comparativa de modelos
- [ ] **Fase 2 — Chunking a mano**: estrategias (fijo / por secciones wiki) y sus efectos
- [ ] **Fase 3 — Búsqueda a mano**: indexar ~200 personajes, ranking con numpy vs SQL, **golden set + recall@5**
- [ ] **Fase 4 — RAG completo**: prompt con citas, iteración sobre fallos, playground
- [ ] **Fase 5 — Framework**: rehacer la pipeline con LangChain y comparar
- [ ] **Fase 6 — Extras**: escalado a 1.571, reranking, búsqueda híbrida, fine-tuning vs RAG

---

## 🚀 Cómo ejecutar

```bash
python3 smoke_test.py          # el circuito completo de punta a punta
python3 explora_embeddings.py  # los 6 experimentos de la Fase 1
```

Requisitos: `.env` con credenciales, `pip install requests numpy psycopg2-binary python-dotenv sentence-transformers`.

---

# 🎓 Didáctica

## Fase 0 — Infraestructura

### pgvector: un Postgres que entiende de vectores

Una columna `embedding VECTOR(1024)` y un operador mágico:

```sql
-- distancia coseno entre el vector guardado y la query:
SELECT personaje, 1 - (embedding <=> :query) AS similitud
FROM chunks
ORDER BY embedding <=> :query   -- ¡usa el índice HNSW!
LIMIT 5;
```

`<=>` devuelve **distancia** coseno (0 = idéntico, 2 = opuesto), por eso la
*similitud* es `1 - distancia`. El `ORDER BY` sobre el operador es lo que
activa el índice en lugar de un escaneo secuencial.

### HNSW en 30 segundos

Comparar la query contra TODOS los vectores es exacto pero lento (escaneo
secuencial). **HNSW** (Hierarchical Navigable Small World) construye un
grafo de vecindarios en varias capas: arriba, pocas "autopistas" gruesas; al
bajar, vecindarios cada vez más finos. La búsqueda salta de vecindario en
vecindario sin mirar todo. Es **aproximado** (puede saltarse el vecino
perfecto) pero ordena de magnitud más rápido. Límite duro: **solo indexa
hasta 2.000 dimensiones**.

### Embeddings matryoshka (por qué pedimos 1024 dims)

`qwen3-embedding` genera nativamente **4096** dimensiones, pero pgvector no
puede indexarlas. El modelo está entrenado con *Matryoshka Representation
Learning*: las primeras dimensiones del vector concentran lo esencial y las
últimas añaden detalle. Pedir `dimensions: 1024` a la API (o truncar el
vector tú mismo, ver experimento S5) conserva casi toda la semántica con
vectores 4× más pequeños e indexables con HNSW.

---

## Fase 1 — Embeddings

### ¿Qué es un embedding?

Un **vector de números reales** que representa el significado de un texto.
"Similar en significado" ⇒ "cerca en el espacio". No hay magia: es un modelo
neuronal entrenado para que textos parecidos produzcan vectores con ángulos
pequeños entre sí.

### La similitud coseno, a mano

$$\cos(\theta) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}$$

Solo mira el **ángulo** entre vectores, no su longitud: por eso compara bien
textos de distinta longitud. Implementada en `embeddings.py` con tres
operaciones numpy (`@`, `norm`, división) — eso es TODO lo que hay detrás.

### Los 6 experimentos y lo que enseñaron

Ejecuta `python3 explora_embeddings.py` para reproducir. Resultados reales
con `qwen3-embedding` (1024 dims):

**S1 · Geometría del coseno.** Consigo mismo → 1.000000; la misma frase dos
veces → 1.000000 (el modelo es determinista); una frase sin relación →
**0.527** (¡no cero!). 💡 *Todo texto comparte algo de fondo semántico con
otro: lo que importa es el ORDEN de las similitudes, no su valor absoluto.
Un "0.64" no significa "64% de parecido".*

**S2 · Mapa semántico.** Matriz completa en el output. Pareja más cercana:
`Son Goku ↔ Kakarotto (0.932)` — ni una letra en común. Más alejados:
`Vegeta ↔ tortilla (0.485)`. Curiosidad docente: **Freezer queda lejos de
todos (0.51–0.57)** — el modelo probablemente pondera también el *congelador*
ingles. Los embeddings llevan el ruido de la ambigüedad del mundo.

**S3 · Retrieva en miniatura — y su primer fallo.** Query: *«el príncipe de
los saiyans»* (sin nombres propios). Resultado:

| Sim | Término |
|---|---|
| 0.857 | **Kakarotto** ← ¡no es el príncipe! |
| 0.828 | Vegeta ← respuesta correcta |
| 0.825 | Son Goku |
| 0.514 | tortilla de patatas |

💡 *El ranking es semánticamente razonable (todo lo saiyan arriba, la
tortilla abajo) pero FALLA en el primer puesto: en textos tan cortos,
"Kakarotto" está tan cargado de saiyans que gana. Enseñanza doble: los
textos cortos dan embeddings ruidosos (Fase 2: chunks con contexto) y por
eso existen los rerankers (Fase 6): el modelo denso propone, el reranker
decide.*

**S4 · Aritmética de conceptos.** `rey − hombre + mujer ≈ ?` → `rey (0.84),
reina (0.79), mujer (0.76)`. 💡 *La analogía funciona a medias: el compuesto
queda pegado a su propio primer término. Ese truco era el de word2vec
(2013); los modelos de frases optimizan semejanza, no direcciones. Si
falla, no es un bug.*

**S5 · Matryoshka en la práctica.** Vector nativo 4096 truncado por
adelante; ranking para *«¿Quién es el príncipe de los saiyans?»* con docs
[Goku, Vegeta, tortilla]:

| dims | ranking | nota |
|---|---|---|
| 4096 | VEGE > GOKU > TORT | nativo |
| 1024 | VEGE > GOKU > TORT | lo que usamos |
| 128 | VEGE > GOKU > TORT | ¡igual! |
| 32 | VEGE > GOKU > TORT | ya se estrecha (0.785 vs 0.760) |

💡 *El ranking sobrevive hasta 32 dims, pero observa la CONVERGENCIA de
valores: a menos dimensiones, todo se parece más (goku 0.532→0.760, tort
0.200→0.308). Perdemos resolución fina. 1024 es el punto dulce para este
proyecto: indexable y sin pérdida perceptible.*

**S6 · Otro modelo = otro universo.** Mismos términos con el MiniLM local
(384 dims): `Kakarotto ↔ Son Goku` baja de **0.932 a 0.470** (¡MiniLM no sabe
que son la misma persona!) y `Goku ↔ tortilla` de 0.506 a 0.095 (separa mejor
lo absurdo). 💡 *Cada modelo define su propio espacio y sus números no son
comparables entre modelos. Regla de oro absoluta del proyecto: **nunca
mezclar embeddings de modelos distintos en la misma base vectorial**. Si
alguna vez cambias de modelo de embeddings, reindexa TODO.*

### Reglas de oro aprendidas (Fase 1)

1. El coseno mide ángulos: ignora longitudes, compara sentidos.
2. Los valores absolutos no significan nada; los órdenes, sí.
3. Textos muy cortos → embeddings ruidosos (lo arregla el chunking con contexto).
4. Cada modelo es un universo: no mezclar espacios, no comparar números entre modelos.
5. El modelo "sabe" del mundo (Kakarotto=Goku) pero solo lo que vio al entrenar.
6. Matryoshka: las primeras dims concentran lo esencial; truncar = trade-off tamaño/resolución.

---

## 🔐 Seguridad

- Credenciales en `.env`, excluido de git (`.gitignore`).
- BD con usuario dedicado `dragonball` (no el superusuario).
- **Pendiente al cerrar el proyecto**: rotar las API keys de NAN y LiteLLM
  (pasaron por el chat durante el arranque).
