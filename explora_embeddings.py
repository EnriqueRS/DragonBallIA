"""Fase 1 — Exploraciones guiadas sobre embeddings de qwen3-embedding.

Cada sección responde una pregunta y enseña un concepto:

  S1  ¿Qué mide exactamente el coseno?           (geometría básica)
  S2  ¿Quién se parece a quién en Dragon Ball?   (mapa semántico)
  S3  ¿Y si busco con una descripción?           (retrieva en miniatura)
  S4  ¿Se puede hacer aritmética de conceptos?   (analogías)
  S5  ¿Por qué usamos 1024 dims y no 4096?       (embeddings matryoshka)
  S6  ¿Otro modelo de embeddings da lo mismo?    (cada modelo es un universo)

Ejecutar:  python3 explora_embeddings.py
"""
import numpy as np

from embeddings import generar_embeddings, generar_embedding, similitud_coseno

TERMINOS_DB = [
    "Son Goku",
    "Kakarotto",
    "Vegeta",
    "Freezer",
    "Beerus",
    "esfera del dragón",
    "Shen Long",
    "tortilla de patatas",
]
ETIQUETAS = ["GOKU", "KAKA", "VEGE", "FREE", "BEER", "ESFE", "SHEN", "TORT"]


def linea(char="─", n=64):
    print(char * n)


def seccion(titulo):
    print()
    linea("═")
    print(titulo)
    linea("═")


def s1_geometria_del_coseno():
    seccion("S1 · ¿Qué mide exactamente el coseno?")
    v_goku = generar_embedding("Son Goku es un saiyan")
    # Comprobaciones geométricas sin meter IA todavía:
    print(f"   consigo mismo        → {similitud_coseno(v_goku, v_goku):.6f}  (≈1: idéntico)")
    print(f"   simetría a·b = b·a   → {similitud_coseno(v_goku, [1, 2, 3] + [0] * 1021):.6f}")
    v2 = generar_embedding("Son Goku es un saiyan")  # mismo texto, segunda llamada
    print(f"   misma frase 2 veces  → {similitud_coseno(v_goku, v2):.6f}  (¿determinista el modelo?)")
    v3 = generar_embedding("Dos y dos son cuatro")
    print(f"   frase sin relación   → {similitud_coseno(v_goku, v3):.6f}  (baja pero no 0)")
    print("""
   💡 Lección: el coseno solo mira el ÁNGULO entre flechas, no su longitud.
      Un embedding nunca da 0 con otra frase: todo comparte algo de fondo
      semántico. Lo que importa es el ORDEN de las similitudes, no su valor
      absoluto (¡ojo con interpretar "0.64 de parecido" literalmente!).""")


def s2_mapa_semantico():
    seccion("S2 · ¿Quién se parece a quién en Dragon Ball?")
    embs = generar_embeddings(TERMINOS_DB)
    n = len(embs)
    print("      " + "  ".join(f"{e:>5}" for e in ETIQUETAS))
    for i in range(n):
        fila = []
        for j in range(n):
            fila.append("     ·" if j == i else f"{similitud_coseno(embs[i], embs[j]):5.3f}")
        print(f"{ETIQUETAS[i]:>4} " + "  ".join(fila))

    # Extraemos las conclusiones automáticamente para leerlas encima:
    sims = [(i, j, similitud_coseno(embs[i], embs[j]))
            for i in range(n) for j in range(i + 1, n)]
    i, j, s = max(sims, key=lambda t: t[2])
    print(f"\n   MÁS PAREJAS:  {TERMINOS_DB[i]!r} ↔ {TERMINOS_DB[j]!r}  ({s:.3f})")
    i, j, s = min(sims, key=lambda t: t[2])
    print(f"   MÁS ALEJADOS: {TERMINOS_DB[i]!r} ↔ {TERMINOS_DB[j]!r}  ({s:.3f})")
    print("""
   💡 Lección: los vectores capturan relaciones de significado, no palabras.
      "Kakarotto" y "Son Goku" no comparten ni una letra relevante, y sin
      embargo viven casi en el mismo punto del espacio.""")


def s3_retrieva_en_miniatura():
    seccion("S3 · Retrieva en miniatura: buscar por DESCRIPCIÓN")
    query = "el príncipe de los saiyans"
    embs = generar_embeddings(TERMINOS_DB)
    q = generar_embedding(query)
    ranking = sorted(
        zip(TERMINOS_DB, [similitud_coseno(q, e) for e in embs]),
        key=lambda t: -t[1],
    )
    print(f"   Query: «{query}»  (no contiene ningún nombre propio)\n")
    for term, s in ranking:
        barra = "█" * int((s - 0.2) * 80)
        print(f"   {s:.4f}  {term:<24} {barra}")
    print("""
   💡 Lección: esto ES un buscador vectorial. "Príncipe de los saiyans" es
      una descripción, y el embedding la lleva al vecindario de Vegeta.
      La Fase 3 será exactamente esto, pero con chunks de las wikis.""")


def s4_aritmetica_de_conceptos():
    seccion("S4 · Aritmética de conceptos (analogías)")
    candidatos = ["rey", "hombre", "mujer", "reina", "príncipe", "tortilla"]
    embs = {t: np.asarray(generar_embedding(t)) for t in candidatos}
    # La receta clásica de word2vec: rey − hombre + mujer ≈ ?
    objetivo = embs["rey"] - embs["hombre"] + embs["mujer"]
    ranking = sorted(
        ((c, similitud_coseno(objetivo, v)) for c, v in embs.items()),
        key=lambda t: -t[1],
    )
    print("   rey − hombre + mujer ≈ ¿?")
    for c, s in ranking:
        print(f"   {s:.4f}  {c}")
    print("""
   💡 Lección: los embeddings modernos capturan SEMEJANZA mejor que
      DIRECCIÓN. La aritmética de analogías era el truco estrella de
      word2vec (2013); con modelos de frases suele funcionar a medias.
      Cuando falle, no es un bug: es que el objetivo del modelo es otro.""")


def s5_matryoshka():
    seccion("S5 · Embeddings matryoshka: ¿por qué 1024 y no 4096 dims?")
    textos = [
        "Son Goku es un saiyan criado en la Tierra por el abuelo Gohan. "
        "Su nombre saiyan es Kakarotto.",
        "Vegeta es el príncipe de los saiyans, hijo del rey Vegeta, y rivals "
        "de Goku desde que llegó a la Tierra.",
        "La tortilla de patatas se hace con huevos, patatas y cebolla.",
    ]
    query = "¿Quién es el príncipe de los saiyans?"

    # Pedimos los vectores NATIVOS (4096) sin el parámetro dimensions:
    nativos = []
    for t in [*textos, query]:
        import requests
        from config import EMBEDDINGS
        r = requests.post(
            f"{EMBEDDINGS['base_url']}/embeddings",
            headers={"Authorization": f"Bearer {EMBEDDINGS['api_key']}"},
            json={"model": EMBEDDINGS["model"], "input": t},
        )
        r.raise_for_status()
        nativos.append(np.asarray(r.json()["data"][0]["embedding"]))
    docs, q = nativos[:3], nativos[3]

    print(f"   Dimensión nativa del modelo: {len(docs[0])}\n")
    print("   Truncamos el vector a las primeras N dimensiones y comparamos:")
    print("   (es lo que hace el parámetro `dimensions` de la API por dentro)")
    print("   → el coseno es invariante a la escala, no hace falta renormalizar\n")
    print("   dims | ranking de documentos para la query")
    for dims in [4096, 1024, 512, 256, 128, 64, 32]:
        sims = [similitud_coseno(q[:dims], d[:dims]) for d in docs]
        orden = sorted(range(3), key=lambda i: -sims[i])
        etiqueta = ["GOKU", "VEGE", "TORT"]
        print(f"   {dims:>4} | {' > '.join(etiqueta[i] for i in orden)}"
              f"    (vege={sims[1]:.3f}, goku={sims[0]:.3f}, tort={sims[2]:.3f})")
    print("""
   💡 Lección: las primeras dimensiones concentran lo esencial del
      significado (el modelo se entrena para ello: MRL, Matryoshka
      Representation Learning). Podemos perder resolución fina a cambio de
      vectores 4× más pequeños e indexables con HNSW (límite pgvector: 2000).""")


def s6_otro_modelo_otro_espacio():
    seccion("S6 · Otro modelo = otro universo (MiniLM local)")
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embs_local = modelo.encode(TERMINOS_DB, normalize_embeddings=True)

    kakarotto_local = embs_local[TERMINOS_DB.index("Kakarotto")]
    goku_local = embs_local[TERMINOS_DB.index("Son Goku")]
    tortilla_local = embs_local[TERMINOS_DB.index("tortilla de patatas")]

    # qwen3-embedding (los ya calculados en S2, los volvemos a pedir):
    embs_nan = generar_embeddings(TERMINOS_DB)
    kakarotto_nan = np.asarray(embs_nan[TERMINOS_DB.index("Kakarotto")])
    goku_nan = np.asarray(embs_nan[TERMINOS_DB.index("Son Goku")])
    tortilla_nan = np.asarray(embs_nan[TERMINOS_DB.index("tortilla de patatas")])

    print(f"   {'':<38}{'qwen3 (NAN)':>14}{'MiniLM (local)':>16}")
    print(f"   {'dims del espacio':<38}{'1024':>14}{embs_local.shape[1]:>16}")
    print(f"   {'Kakarotto ↔ Son Goku':<38}"
          f"{similitud_coseno(kakarotto_nan, goku_nan):>14.4f}"
          f"{float(kakarotto_local @ goku_local):>16.4f}")
    print(f"   {'Son Goku ↔ tortilla':<38}"
          f"{similitud_coseno(goku_nan, tortilla_nan):>14.4f}"
          f"{float(goku_local @ tortilla_local):>16.4f}")
    print("""
   💡 Lección: cada modelo define SU PROPIO espacio; los números NO son
      comparables entre modelos (¿0.79 de qwen "equivale" a 0.71 de MiniLM?
      Imposible saberlo: son escalas distintas). Y la consecuencia práctica
      más importante del proyecto: NUNCA mezcles embeddings de modelos
      distintos en la misma base vectorial.""")


if __name__ == "__main__":
    s1_geometria_del_coseno()
    s2_mapa_semantico()
    s3_retrieva_en_miniatura()
    s4_aritmetica_de_conceptos()
    s5_matryoshka()
    s6_otro_modelo_otro_espacio()
