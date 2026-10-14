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
`950f2c0` de git). Fracasó silenciosamente y la autopsia enseña más que el
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
                          │   NUC (Postgres en LAN)     │
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
| LLM | `nan/glm5.3-flash` vía proxy LiteLLM propio | Self-hosted en el homelab |
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
| `chunking.py` | 2 | Las 3 estrategias de troceado a mano + limpieza wiki + ficha infobox |
| `explora_chunking.py` | 2 | Comparativa de estrategias: estadísticas, dilución y mini-evaluación (🎓 Fase 2) |
| `dataset_dragonball/raw_texts/` | — | 1.571 ficheros `.txt` de personajes de la wiki |
| `dataset_dragonball/dragonball_finetuning.jsonl` | — | 6.893 pares instrucción/respuesta (para la Fase 6 opcional) |

---

## 🗺️ Plan de fases

- [x] **Fase 0 — Infraestructura**: BD pgvector + config central + smoke test verde
- [x] **Fase 1 — Embeddings a mano**: coseno a mano, mapa semántico, matryoshka, comparativa de modelos
- [x] **Fase 2 — Chunking a mano**: 3 estrategias comparadas + validación anti-corrupción del API
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

## Fase 2 — Chunking

### ¿Por qué partir el texto?

Un embedding representa UN significado en UN vector. Si metes una página
entera (70.000 caracteres en la de Goku) en un solo vector, ese vector no
representa ni la infancia de Goku ni su personalidad: representa
«mezcla de todo, dominado por lo primero que aparece». El chunking decide
qué unidades de significado se indexan y, con ello, qué puede recuperar tu
buscador. **Es la mitad del éxito de un RAG.**

### Las tres estrategias (implementadas a mano en `chunking.py`)

1. **Página entera truncada a 2.500** — la recreación fiel del bug que mató
   al proyecto viejo: 1 chunk por personaje, sin limpieza profunda.
2. **Fijo 1000/200** — ventanas de ~1.000 caracteres con solapamiento de
   200 (lo que hacía `RecursiveCharacterTextSplitter` de LangChain): parte
   bien los tamaños pero es ciega a la estructura.
3. **Secciones wiki** — consciente de la estructura: la ficha infobox se
   parsea a campos limpios (`familia: Bardock (Padre); Gine (Madre)…`), la
   intro y cada `== Sección ==` se trocean a ~1.200 caracteres con
   solapamiento por párrafos, y se descartan secciones de navegación
   (Galería, Referencias, Navegación). Todo con **inyección de sujeto**:
   cada chunk lleva el prefijo `Son Goku (Historia > Infancia): …` — un
   chunk suelto no sabe de quién habla.

### Resultados (8 personajes: Goku, Vegeta IV, Monaka, Paragus, Gine, Whis, Beerus, Freeza)

| Estrategia | nº chunks | media chars | % cortados a mitad de frase |
|---|---|---|---|
| 1. página entera | 8 | 2.500 (truncados) | 100% |
| 2. fijo 1000/200 | 226 | 832 | 48% |
| 3. secciones | 230 | 811 | 38% |

**La dilución (C3)** — media de similitud coseno entre chunks de personajes
DISTINTOS (cuanto más alta, más "igual entre sí" es todo el corpus):

| Estrategia | media | desviación |
|---|---|---|
| 1. página entera | **0.573** | **0.073** ← suelo más alto, menos contraste |
| 2. fijo | 0.484 | 0.094 |
| 3. secciones | **0.466** | 0.109 ← más espacio semántico, más señal |

**Mini-evaluación (C4)** — hit@3 con golden set de 8 preguntas:
8/8 con estrategia 1, 7/8 con las otras dos. Lectura honesta: con solo
8 candidatos, la estrategia 1 acierta «por eliminatoria» (3 plazas para 8
personajes) aunque su vector sea malo — **hit@3 con n=8 es una métrica
débil**. El veredicto real está en la dilución: con las 1.571 páginas del
dataset, los vectores-platillo de la estrategia 1 se amontonan en la misma
zona del espacio y el top-k se llena de personajes equivocados — el
mecanismo detrás de los scores planos (0.8550, 0.8538, 0.8560…) del
`response.txt` del proyecto viejo. La Fase 3 (200 personajes) hará visible
el efecto con n grande.

**Lección de evaluación**: la query *«¿quién destruyó el planeta Vegeta?»*
falla porque las páginas de Vegeta IV mencionan sin parar el planeta que da
nombre a su raza. El hit@personaje es una métrica burda — el chunk ganador
puede contener la respuesta sin ser del personaje esperado (y al revés).
Afinar el golden set es parte del trabajo; evaluar por RESPUESTA llega en
la Fase 4.

### ⚠️ El bug del API: corrupción silenciosa por lotes

Durante esta fase, `qwen3-embedding` vía NAN **falló de forma intermitente
con lotes**: devolvía n vectores por n textos — pero con índices repetidos
(`[0,1,2,3,3,4,5,6]`) y vectores duplicados (solo k únicos calculados),
**sin ningún código de error**. La primera vez se manifestó como un
misterioso «¿personalidad de Vegeta?» → Monaka: una query había recibido el
vector de otra. Validar a mano los embeddings de las queries reveló el
patrón (matriz de similitudes con filas idénticas).

La defensa, en `embeddings.py`:

```python
def _contrato_roto(data, n):        # valida CADA respuesta:
    # 1. nº vectores == nº textos
    # 2. índices == 0..n-1 sin repetir
    # 3. ningún par de vectores idénticos (dot > 0.9999)
```

Lote inválido → reintento (hasta 3) → degradación a llamadas individuales.
En la última ejecución cazó en vivo un lote corrupto y lo repitió con
éxito. Sin esa validación, chunks con el embedding de OTRO texto habrían
entrado en la base vectorial: **el fallo más peligroso no es el que lanza
error, es el que responde mal sin avisar**.

### Reglas de oro (Fase 2)

1. El chunking decide qué puede recuperar tu buscador: 1 chunk = 1 idea.
2. Inyecta el sujeto en cada chunk (nombre + sección): el chunk viaja solo.
3. La ficha infobox es oro estructurado: parsearla, no indexarla como prosa.
4. Limpia el marcado wiki antes de embeber; descarta secciones de navegación.
5. El solapamiento preserva contexto en el corte, no es gratis (duplica contenido).
6. Valida los contratos de tus dependencias externas, aunque "funcionen".

---

## 🔐 Seguridad

- Credenciales en `.env`, excluido de git (`.gitignore`).
- BD con usuario dedicado `dragonball` (no el superusuario).
- **Pendiente al cerrar el proyecto**: rotar las API keys de NAN y LiteLLM
  (pasaron por el chat durante el arranque).
