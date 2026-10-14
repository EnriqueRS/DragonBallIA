"""Fase 1 — Cliente de embeddings + la métrica estrella: similitud coseno.

Responsabilidades:
  1. hablar con la API de embeddings de NAN (individual y por lotes);
  2. medir "parecido" entre vectores con numpy, a mano, para entender
     qué hace por dentro cualquier librería de similitud.

⚠️  Lección de ingeniería (descubierta en Fase 2): la API de embeddings
    FALLA DE FORMA INTERMITENTE con lotes: a veces devuelve índices
    repetidos y vectores duplicados (n vectores por n textos, pero solo
    k únicos), sin ningún código de error. Corrupción silenciosa.
    Defensa: validar el contrato de cada respuesta (nº, índices, sin
    duplicados) y degradar a llamadas individuales si se viola.
"""
import sys

import requests
import numpy as np

from config import EMBEDDINGS

TAM_LOTE = 32          # cuántos textos por petición
INTENTOS_LOTE = 3      # reintentos si el lote viola el contrato
UMBRAL_DUPLICADO = 0.9999


def _post_embeddings(textos: list[str]):
    r = requests.post(
        f"{EMBEDDINGS['base_url']}/embeddings",
        headers={"Authorization": f"Bearer {EMBEDDINGS['api_key']}"},
        json={
            "model": EMBEDDINGS["model"],
            "input": textos,
            "dimensions": EMBEDDINGS["dim"],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"]


def _contrato_roto(data, n: int) -> str | None:
    """Valida el contrato de una respuesta por lotes.

    Devuelve None si es válida, o un texto describiendo la violación:
      1. nº de vectores == nº de textos;
      2. los `index` cubren 0..n-1 sin repetir (para poder reordenar);
      3. ningún par de vectores idénticos (la API a veces rellena con
         duplicados de otros textos del lote).
    """
    if len(data) != n:
        return f"devueltos {len(data)} vectores para {n} textos"
    if sorted(d["index"] for d in data) != list(range(n)):
        return f"índices corruptos: {[d['index'] for d in data]}"
    E = np.asarray([d["embedding"] for d in data], dtype=float)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    np.fill_diagonal(S, 0)
    if (S > UMBRAL_DUPLICADO).any():
        return "contiene vectores duplicados (corrupción del lote)"
    return None


def _embed_individual(texto: str) -> list[float]:
    for intento in range(INTENTOS_LOTE):
        data = _post_embeddings([texto])
        if (error := _contrato_roto(data, 1)) is None:
            return data[0]["embedding"]
        print(f"   ⚠️ llamada individual inválida ({error}), reintento {intento + 1}",
              file=sys.stderr)
    raise RuntimeError("La API de embeddings devuelve respuestas inválidas de forma persistente")


def generar_embeddings(textos: list[str]) -> list[list[float]]:
    """Lista de textos → lista de vectores (por lotes, con validación).

    Indexar miles de chunks llamando de uno en uno sería lento e ineficiente:
    el API acepta `input` como lista. Pero como el lote falla de forma
    intermitente, cada lote se valida y, si está corrupto, se reintenta y
    finalmente se degrada a llamadas individuales.
    """
    resultados: list[list[float]] = []
    for i in range(0, len(textos), TAM_LOTE):
        lote = textos[i : i + TAM_LOTE]
        for intento in range(INTENTOS_LOTE):
            data = _post_embeddings(lote)
            if (error := _contrato_roto(data, len(lote))) is None:
                break
            print(f"   ⚠️ lote de {len(lote)} inválido ({error}), reintento {intento + 1}",
                  file=sys.stderr)
        else:
            print(f"   ⚠️ lote corrupto tras {INTENTOS_LOTE} intentos → llamadas individuales",
                  file=sys.stderr)
            resultados.extend(_embed_individual(t) for t in lote)
            continue
        # El campo `index` marca la posición de cada texto en el lote.
        data = sorted(data, key=lambda d: d["index"])
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
