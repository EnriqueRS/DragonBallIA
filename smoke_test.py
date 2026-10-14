"""Fase 0 — Smoke test: el circuito completo de un RAG en miniatura.

    texto → embedding (NAN) → pgvector → búsqueda por coseno → prompt → LLM

Cada función es una pieza del circuito. En las fases siguientes cada una
se convertirá en su propio módulo (embeddings.py, buscar.py, rag.py).
"""
import requests
import psycopg2

from config import EMBEDDINGS, LLM, PG

# Tres mini-fichas de conocimiento. Sustituyen al dataset real para probar
# el circuito sin depender del chunking (Fase 2) ni del scraping.
FRAGMENTOS = [
    (
        "Son Goku",
        "Historia",
        "Son Goku es un saiyan nacido en el planeta Vegeta y enviado a la Tierra "
        "como bebé para conquistarla. Su nombre saiyan es Kakarotto. Fue criado por "
        "el abuelo Gohan, quien le enseñó artes marciales.",
    ),
    (
        "Vegeta",
        "Historia",
        "Vegeta es el príncipe de los saiyans, hijo del rey Vegeta. Llegó a la "
        "Tierra para conseguir las esferas del dragón y acabó estableciéndose en "
        "el planeta tras ser derrotado por Goku.",
    ),
    (
        "Beerus",
        "Descripción",
        "Beerus es el Dios de la Destrucción del Universo 7. Su compañero y "
        "instructor es el ángel Whis. Duérmete durante siglos y al despertar "
        "busca al Super Saiyan God.",
    ),
]


def generar_embedding(texto: str) -> list[float]:
    """Texto → vector de EMBEDDINGS['dim'] dimensiones (API de embeddings de NAN)."""
    r = requests.post(
        f"{EMBEDDINGS['base_url']}/embeddings",
        headers={"Authorization": f"Bearer {EMBEDDINGS['api_key']}"},
        json={"model": EMBEDDINGS["model"], "input": texto, "dimensions": EMBEDDINGS["dim"]},
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def insertar_chunks(conn, fragmentos, embeddings):
    """Inserta (personaje, seccion, contenido, embedding) en la tabla chunks.

    pgvector acepta el vector como texto '[0.1,0.2,...]' con un cast ::vector.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE fuente = 'smoke_test'")
        for (personaje, seccion, contenido), emb in zip(fragmentos, embeddings):
            cur.execute(
                """INSERT INTO chunks (personaje, seccion, contenido, embedding, fuente)
                   VALUES (%s, %s, %s, %s::vector, %s)""",
                (personaje, seccion, contenido, "[" + ",".join(f"{x:.7g}" for x in emb) + "]", "smoke_test"),
            )
    conn.commit()
    print(f"   → {len(fragmentos)} filas insertadas\n")


def buscar(conn, query: str, k: int = 2):
    """Búsqueda semántica pura en SQL: ordena por distancia coseno <=>.

    `<=>` devuelve DISTANCIA coseno (0 = idéntico, 2 = opuesto), así que la
    SIMILITUD coseno es 1 - distancia. El ORDER BY usa el índice HNSW.
    """
    q_emb = "[" + ",".join(f"{x:.7g}" for x in generar_embedding(query)) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """SELECT personaje, seccion,
                      contenido,
                      1 - (embedding <=> %s::vector) AS similitud
               FROM chunks
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (q_emb, q_emb, k),
        )
        return cur.fetchall()


def responder_llm(prompt: str) -> str:
    """Envía el prompt ya construido al LLM (API OpenAI-compatible del proxy)."""
    r = requests.post(
        f"{LLM['base_url']}/chat/completions",
        headers={"Authorization": f"Bearer {LLM['api_key']}"},
        json={"model": LLM["model"], "messages": [{"role": "user", "content": prompt}], "temperature": 0},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    conn = psycopg2.connect(**PG)
    try:
        print("1️⃣  Embedding de los fragmentos de prueba...")
        embeddings = [generar_embedding(p[2]) for p in FRAGMENTOS]
        print(f"   → {len(embeddings)} vectores de {len(embeddings[0])} dimensiones\n")

        print("2️⃣  Insertando en pgvector...")
        insertar_chunks(conn, FRAGMENTOS, embeddings)

        pregunta = "¿Quién es Kakarotto y quién lo crio?"
        print(f"3️⃣  Búsqueda semántica para: «{pregunta}»")
        resultados = buscar(conn, pregunta, k=2)
        for personaje, seccion, contenido, sim in resultados:
            print(f"   [{sim:.4f}] {personaje} ({seccion})")
        print()

        contexto = "\n\n".join(
            f"[{p} - {s}] {c}" for p, s, c, _ in resultados
        )
        prompt = (
            "Eres un experto en Dragon Ball. Responde usando SOLO el contexto. "
            "Si la respuesta no está en el contexto, di 'No hay datos'.\n\n"
            f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"
        )

        print("4️⃣  Preguntando al LLM...")
        respuesta = responder_llm(prompt)
        print(f"\n   💬 {respuesta}")
    finally:
        conn.close()

    print("\n✅ Smoke test completo: embedding → pgvector → búsqueda → LLM")


if __name__ == "__main__":
    main()
