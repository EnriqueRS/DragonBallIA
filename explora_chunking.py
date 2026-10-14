"""Fase 2 — Experimentos comparando las 3 estrategias de chunking.

  C1  Estadísticas de chunks (nº, tamaños, % cortados a mitad de frase)
  C2  Goku en vivo: 3 chunks reales por estrategia, para inspección manual
  C3  La dilución: por qué el proyecto viejo recuperaba a Sidra si
      preguntabas por Monaka
  C4  Mini-evaluación: golden set de 8 preguntas → ¿acierta el personaje
      en el top-3 de cada estrategia?

Ejecutar:  python3 explora_chunking.py
"""
from pathlib import Path
from statistics import median

import numpy as np

from chunking import ESTRATEGIAS, Chunk, limpiar
from embeddings import generar_embeddings

RAW = Path("dataset_dragonball/raw_texts")

PERSONAJES = ["Son Goku", "Vegeta IV", "Monaka", "Paragus", "Gine",
              "Whis", "Beerus", "Freeza"]

# (pregunta, personaje que DEBE aparecer en el top-3 de chunks recuperados)
GOLDEN = [
    ("¿Quién es el padre de Broly?", "Paragus"),
    ("¿Tiene hijos Gine?", "Gine"),
    ("¿Cómo era la infancia de Son Goku?", "Son Goku"),
    ("¿Quién es Monaka?", "Monaka"),
    ("¿Qué personalidad tiene Vegeta?", "Vegeta IV"),
    ("¿Quién es Whis y qué papel tiene?", "Whis"),
    ("¿Quién destruyó el planeta Vegeta?", "Freeza"),
    ("¿Qué comida le gusta a Beerus?", "Beerus"),
]


def cargar(personaje: str) -> str:
    return (RAW / f"{personaje}.txt").read_text(encoding="utf-8", errors="ignore")


def chunks_de(nombre_estrategia: str) -> list[Chunk]:
    funcion = ESTRATEGIAS[nombre_estrategia]
    return [c for p in PERSONAJES for c in funcion(cargar(p), p)]


def pct_cortados(chunks: list[Chunk]) -> float:
    """% de chunks cuyo último carácter no es fin de frase natural."""
    fines = {".", "!", "?", "…", "»", '"', ")", "]"}
    cortados = [c for c in chunks if c.texto.rstrip()[-1] not in fines]
    return 100 * len(cortados) / max(len(chunks), 1)


def linea(char="─", n=66):
    print(char * n)


def seccion(titulo):
    print()
    linea("═")
    print(titulo)
    linea("═")


# ---------------------------------------------------------------- C1
def c1_estadisticas():
    seccion("C1 · Estadísticas de chunks por estrategia")
    print(f"   {'estrategia':<20}{'nº':>6}{'media':>8}{'p50':>7}{'p95':>7}{'máx':>7}{'%cortado':>10}")
    for nombre in ESTRATEGIAS:
        cs = chunks_de(nombre)
        tams = sorted(len(c.texto) for c in cs)
        print(f"   {nombre:<20}{len(cs):>6}"
              f"{int(np.mean(tams)):>8}{tams[len(tams)//2]:>7}"
              f"{tams[int(len(tams)*0.95)]:>7}{max(tams):>7}"
              f"{pct_cortados(cs):>9.0f}%")
    print("""
   💡 La estrategia 1 produce 1 chunk por personaje: TODO el significado
      comprimido en un solo vector y truncado a las primeras líneas.
      La 2 parte bien pero a ciegas. La 3 genera muchos chunks pequeños
      y semánticamente homogéneos.""")


# ---------------------------------------------------------------- C2
def c2_goku_en_vivo():
    seccion("C2 · Goku en vivo (inspección manual — léelos)")
    texto = cargar("Son Goku")
    print(f"   Página original: {len(texto):,} caracteres\n")
    for nombre in ESTRATEGIAS:
        cs = ESTRATEGIAS[nombre](texto, "Son Goku")
        print(f"   ── {nombre}: {len(cs)} chunks para Goku")
        for c in cs[:3]:
            preview = c.texto[:220].replace("\n", " ¶ ")
            print(f"      [{c.seccion}] «{preview}…»")
        print()


# ---------------------------------------------------------------- C3
def _matriz(M):
    N = M / np.linalg.norm(M, axis=1, keepdims=True)
    return N @ N.T


def c3_dilucion(cache):
    seccion("C3 · La dilución: la autopsia del proyecto viejo")
    print("   Media de similitud coseno entre chunks de personajes DISTINTOS:")
    for nombre in ESTRATEGIAS:
        cs = cache[nombre]
        M = np.asarray(cache["emb"][nombre])
        S = _matriz(M)
        perso = np.asarray([c.personaje for c in cs])
        distintos = S[perso[:, None] != perso[None, :]]
        print(f"   {nombre:<20} media={distintos.mean():.3f}  desv={distintos.std():.3f}")
    print("""
   💡 Lectura honesta: la estrategia 1 sube el SUELO de similitud
      (media 0.573 vs 0.466) y reduce la dispersión (desv 0.073 vs
      0.109): al ser todas las páginas «ficha infobox truncada», todos
      los personajes se parecen más entre sí y hay menos contraste.
      Con solo 8 personajes famosos la búsqueda aún se salva (Monaka
      gana 0.685 vs 0.432), pero con las 1.571 páginas del dataset ese
      suelo plano inunda el top-k — el mecanismo tras los scores planos
      (0.855/0.853/0.856...) del response.txt del proyecto viejo.
      La Fase 3, con 200 personajes, hará visible el efecto en hit@k.""")

    print("\n   Y la prueba de fuego — query «¿Quién es Monaka?» (top-5):")
    q = np.asarray(generar_embeddings(["¿Quién es Monaka?"])[0])
    for nombre in ESTRATEGIAS:
        cs = cache[nombre]
        sims = _matriz(np.vstack([q, cache["emb"][nombre]]))[0, 1:]
        top = sorted(zip(cs, sims), key=lambda t: -t[1])[:5]
        detalle = ", ".join(f"{c.personaje}({s:.3f})" for c, s in top)
        print(f"   {nombre:<20} {detalle}")
    print("""
   💡 Con 8 personajes la query aún encuentra a Monaka en las tres
      estrategias (0.685 vs 0.432 con la 1). La diferencia está en el
      ESCENARIO: la 1 solo tiene 8 vectores «todo el personaje
      comprimido» y su suelo de similitud es el más alto (ver C1);
      cuando el corpus crece a 1.571 páginas, esos vectores-platillo
      (ficha+intro+algo más) se acumulan en la misma zona del espacio y
      el top-k se llena de personajes que no son — los scores planos
      0.8550/0.8538/0.8560 del response.txt del proyecto viejo, con
      miles de candidatos y otro modelo de embeddings.""")


# ---------------------------------------------------------------- C4
def c4_golden(cache):
    seccion("C4 · Mini-evaluación: hit@3 sobre el golden set")
    preguntas = [p for p, _ in GOLDEN]
    q_emb = np.asarray(generar_embeddings(preguntas))

    aciertos = {n: 0 for n in ESTRATEGIAS}
    for i, (pregunta, esperado) in enumerate(GOLDEN):
        fila = []
        for nombre in ESTRATEGIAS:
            cs = cache[nombre]
            sims = _matriz(np.vstack([q_emb[i], cache["emb"][nombre]]))[0, 1:]
            top3 = sorted(zip(cs, sims), key=lambda t: -t[1])[:3]
            hit = any(c.personaje == esperado for c, _ in top3)
            aciertos[nombre] += hit
            ganador = top3[0][0]
            fila.append(f"{'✅' if hit else '❌'} 1º={ganador.personaje}({top3[0][1]:.2f})")
        print(f"   {pregunta[:42]:<44} " + "   ".join(fila))

    print(f"\n   {'HIT@3':<44}", end="")
    for nombre in ESTRATEGIAS:
        print(f"   {aciertos[nombre]}/{len(GOLDEN)}".center(16), end="")
    print("""
   💡 Lectura honesta de la tabla: con 8 candidatos y 3 plazas, la
      estrategia 1 puede acertar «por eliminatoria» aunque su vector sea
      malo (hit@3 con n=8 mide poco). El veredicto real está en C3: su
      suelo de similitud es el más alto y su dispersión la menor. Otra
      lección: «¿quién destruyó el planeta Vegeta?» falla porque las
      páginas de Vegeta IV mencionan constantemente el planeta que da
      nombre a su raza — el hit@personaje es una métrica burda (el chunk
      ganador hasta podría CONTENER la respuesta) y afinar el golden set
      es parte del trabajo. Fase 3: métricas con n grande y evaluación
      por respuesta, no por personaje.""")


# ---------------------------------------------------------------- main
def main():
    cache = {"emb": {}}
    for nombre in ESTRATEGIAS:
        cs = chunks_de(nombre)
        cache[nombre] = cs
        cache["emb"][nombre] = generar_embeddings([c.texto for c in cs])
        print(f"   [embed] {nombre}: {len(cs)} chunks")

    c1_estadisticas()
    c2_goku_en_vivo()
    c3_dilucion(cache)
    c4_golden(cache)


if __name__ == "__main__":
    main()
