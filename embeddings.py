"""Fase 1 — Cliente de embeddings + la métrica estrella: similitud coseno.

Dos responsabilidades:
  1. hablar con la API de embeddings de NAN (individual y por lotes);
  2. medir "parecido" entre vectores con numpy, a mano, para entender
     qué hace por dentro cualquier librería de similitud.
"""
import requests
import numpy as np

from config import EMBEDDINGS

TAM_LOTE = 64  # cuántos textos por petición (la API acepta listas)


def generar_embeddings(textos: list[str]) -> list[list[float]]:
    """Lista de textos → lista de vectores (llamadas por lotes).

    Indexar miles de chunks llamando de uno en uno sería lento e ineficiente:
    el API de embeddings es estilo OpenAI y acepta `input` como lista.
    """
    resultados: list[list[float]] = []
    for i in range(0, len(textos), TAM_LOTE):
        lote = textos[i : i + TAM_LOTE]
        r = requests.post(
            f"{EMBEDDINGS['base_url']}/embeddings",
            headers={"Authorization": f"Bearer {EMBEDDINGS['api_key']}"},
            json={
                "model": EMBEDDINGS["model"],
                "input": lote,
                "dimensions": EMBEDDINGS["dim"],
            },
        )
        r.raise_for_status()
        # El API devuelve los vectores con su campo `index` (posición en el
        # lote), así que reordenamos por si acaso no vienen en orden.
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        resultados.extend(d["embedding"] for d in data)
    return resultados


def generar_embedding(texto: str) -> list[float]:
    """Texto → vector. Azúcar para el caso de uno solo."""
    return generar_embeddings([texto])[0]


def similitud_coseno(a, b) -> float:
    """Similitud coseno entre dos vectores, implementada a mano.

        cos(θ) = (a · b) / (‖a‖ · ‖b‖)

    Interpretación geométrica: NO mide distancia entre puntos, sino el ángulo
    entre flechas. Ignora la longitud del vector y solo mira la dirección:
      1 → misma dirección (mismo significado)
      0 → ortogonales (sin relación)
     −1 → opuestos

    ¿Por qué coseno y no distancia euclídea? Porque en RAG comparamos
    textos de longitud distinta: los embeddings largos tienen normas mayores
    y la distancia euclídea se dejaría engañar por el tamaño, no el sentido.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
