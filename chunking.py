"""Fase 2 — Chunking a mano: partir el texto antes de embeber.

El chunking es LA decisión que mató al proyecto anterior: indexaba cada
página entera como 1 chunk truncado a 2.500 caracteres, y los vectores de
todo el dataset acababan pareciéndose entre sí (recuperaba a Sidra cuando
preguntabas por Monaka).

Aquí implementamos sin LangChain las tres estrategias que vamos a comparar:

  1. `chunk_pagina_entera`  — la recreación fiel del bug (1 chunk, truncado)
  2. `chunk_fijo`           — ventanas de N caracteres con solapamiento
  3. `chunk_secciones`      — consciente de la estructura wiki:
                              ficha infobox + intro + secciones ==...==

Ideas transversales:
  * `limpiar()`: quitar el ruido wiki (<ref>, plantillas, enlaces) — el
    embedding debe medir SIGNIFICADO, no marcado.
  * Inyección de sujeto: cada chunk lleva el nombre del personaje y su
    sección en el prefijo («Son Goku (Historia > Infancia): ...»). Un chunk
    suelto sin contexto no sabe de quién habla.
  * La ficha infobox se convierte en su propio chunk: los campos
    estructurados (familia, raza, transformaciones) responden preguntas
    exactas («¿tiene hijos Gine?») mejor que prosa.
"""
import re
from dataclasses import dataclass

MAX_CHARS_SECCION = 1200  # tamaño objetivo por chunk en la estrategia 3
OVERLAP = 150             # solapamiento (en caracteres) entre trozos

# Secciones puramente de navegación: no aportan significado, solo ruido.
SECCIONES_IGNORADAS = {"galería", "galeria", "referencias", "navegación",
                       "navegacion", "véase también", "vease tambien"}

# --- limpieza ------------------------------------------------------------

RE_REF = re.compile(r"<ref[^>]*/>|<ref.*?</ref>", re.DOTALL)
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
RE_LINK_EXT = re.compile(r"\[https?://\S+\s+([^\]]+)\]")   # [url texto] → texto
RE_LINK_EXT_SOLA = re.compile(r"\[https?://\S+\]")          # [url] → nada
RE_LINK_INT = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")  # [[a|b]] → b
RE_BR = re.compile(r"<br\s*/?>")
RE_TAG = re.compile(r"</?[a-zA-Z]+[^>]*>")


def limpiar(texto: str) -> str:
    """Quita el marcado wiki para que el embedding mida significado."""
    t = RE_REF.sub("", texto)
    t = RE_TEMPLATE.sub("", t)
    t = RE_TEMPLATE.sub("", t)  # segunda pasada: plantillas anidadas simples
    t = RE_LINK_EXT.sub(r"\1", t)
    t = RE_LINK_EXT_SOLA.sub("", t)
    t = RE_LINK_INT.sub(r"\1", t)
    t = RE_BR.sub("\n", t)
    t = RE_TAG.sub("", t)
    t = re.sub(r"\{+|\}+", "", t)   # llaves huérfanas de plantillas partidas
    t = re.sub(r"^\|\s*", "", t, flags=re.MULTILINE)  # tuberías residuales
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# --- tipos ----------------------------------------------------------------

@dataclass
class Chunk:
    texto: str        # texto final listo para embeber (con inyección de sujeto)
    personaje: str
    seccion: str      # ruta de secciones, p.ej. "Historia > Infancia"


# --- ficha infobox --------------------------------------------------------

def separar_ficha(texto: str) -> tuple[list[str], list[str]]:
    """Divide el fichero en (líneas de la ficha infobox, resto del texto).

    La ficha son las líneas `|campo = valor` del principio, cerradas por `}}`.
    """
    lineas = texto.split("\n")
    for i, linea in enumerate(lineas):
        if linea.strip() == "}}":
            return lineas[:i], lineas[i + 1:]
    return [], lineas  # fichero sin ficha infobox


CAMPOS_IGNORADOS = {"imagen", "pronunciación", "pronunciacion"}


def campos_ficha(lineas: list[str]) -> list[str]:
    """Convierte las líneas `|campo = valor` en pares limpios «campo: valor»."""
    crudos: dict[str, str] = {}
    campo_actual = None
    for linea in lineas:
        linea = linea.strip()
        if linea.startswith("|") and "=" in linea:
            campo, _, valor = linea[1:].partition("=")
            campo_actual = campo.strip().lower()
            crudos[campo_actual] = valor
        elif campo_actual and linea:  # continuación del valor anterior
            crudos[campo_actual] += " " + linea

    campos = []
    for campo, valor in crudos.items():
        if campo in CAMPOS_IGNORADOS:
            continue
        v = limpiar(valor)
        # los <li> ya son "; " y los <br /> saltos: compactamos separadores
        v = re.sub(r"\s*[;\n]\s*", "; ", v).strip("; ").strip()
        if v:  # descartamos campos vacíos («sexo = » etc.)
            campos.append(f"{campo}: {v}")
    return campos


# --- secciones wiki -------------------------------------------------------

RE_HEADER = re.compile(r"^\s*(={2,6})\s*(.*?)\s*\1\s*$")


def parsear_secciones(texto: str) -> list[tuple[str, str]]:
    """Convierte el cuerpo en [(ruta_de_secciones, contenido)].

    La ruta acumula la jerarquía: «Historia > Dragon Ball > Arco del Saiyan».
    El texto anterior a la primera cabecera se devuelve con ruta «Introducción».
    """
    pila: list[tuple[int, str]] = []
    resultado: list[tuple[str, list[str]]] = []
    ruta_actual, buf = "Introducción", []

    def cerrar():
        contenido = "\n".join(buf).strip()
        if contenido:
            resultado.append((ruta_actual, contenido))

    for linea in texto.split("\n"):
        m = RE_HEADER.match(linea)
        if m:
            cerrar()
            buf = []
            nivel, titulo = len(m.group(1)), m.group(2).strip()
            while pila and pila[-1][0] >= nivel:
                pila.pop()
            pila.append((nivel, titulo))
            ruta_actual = " > ".join(t for _, t in pila)
        else:
            buf.append(linea)
    cerrar()
    return [(ruta, contenido) for ruta, contenido in resultado]


# --- troceado genérico ----------------------------------------------------

def empaquetar_parrafos(texto: str, max_chars: int, overlap: int) -> list[str]:
    """Junta párrafos en trozos de ≈max_chars; el nuevo trozo arranca con el
    último párrafo del anterior (solapamiento de contexto, no de caracteres).
    """
    if len(texto) <= max_chars:
        return [texto]
    parrafos = [p.strip() for p in texto.split("\n") if p.strip()]
    trozos, actual, cola = [], "", ""
    for p in parrafos:
        if len(p) > max_chars:  # párrafo gigante: corte duro (caso raro)
            if actual:
                trozos.append(actual)
                actual, cola = "", ""
            trozos.extend(p[i:i + max_chars].strip() for i in range(0, len(p), max_chars))
            continue
        if actual and len(actual) + len(p) + 1 > max_chars:
            trozos.append(actual)
            actual = (cola + "\n" + p) if cola else p
        else:
            actual = f"{actual}\n{p}" if actual else p
        cola = p if len(p) <= overlap else ""
    if actual:
        trozos.append(actual)
    return trozos


# --- las tres estrategias ---------------------------------------------------

def chunk_pagina_entera(texto: str, personaje: str) -> list[Chunk]:
    """Estrategia 1 — recreación del bug del proyecto viejo.

    Un único chunk con la página truncada a 2.500 caracteres. Como las
    primeras líneas de TODAS las páginas son la ficha infobox, el vector de
    cada personaje acababa representando «una ficha con tablas», no al
    personaje. De ahí la dilución que verás en explora_chunking.py.
    """
    cuerpo = limpiar(texto)
    prefijo = f"Información sobre {personaje}: "
    return [Chunk(prefijo + cuerpo[: 2500 - len(prefijo)], personaje, "página entera")]


def chunk_fijo(texto: str, personaje: str, tam: int = 1000, overlap: int = 200) -> list[Chunk]:
    """Estrategia 2 — ventanas de `tam` caracteres con solapamiento.

    Es lo que hacía RecursiveCharacterTextSplitter de LangChain en el
    proyecto viejo (chunk_size=1000, chunk_overlap=200), pero a mano.
    Ciega a la estructura: corta donde toca, secciones partidas en dos.
    """
    cuerpo = limpiar(texto)
    trozos = empaquetar_parrafos(cuerpo, tam, overlap)
    return [Chunk(f"{personaje}: {t}", personaje, "texto fijo") for t in trozos]


def chunk_secciones(texto: str, personaje: str) -> list[Chunk]:
    """Estrategia 3 — consciente de la estructura wiki.

    Ficha → 1 chunk con los campos; intro y cada sección → chunks de
    ≈MAX_CHARS_SECCION con solapamiento; secciones de navegación fuera;
    todo con inyección de sujeto y ruta de sección.
    """
    chunks: list[Chunk] = []

    lineas_ficha, resto = separar_ficha(texto)
    campos = campos_ficha(lineas_ficha)
    if campos:
        ficha = f"{personaje} (Ficha): " + " | ".join(campos)
        for trozo in empaquetar_parrafos(ficha, MAX_CHARS_SECCION, 0):
            chunks.append(Chunk(trozo, personaje, "Ficha"))

    for ruta, contenido in parsear_secciones("\n".join(resto)):
        if ruta.split(" > ")[-1].lower() in SECCIONES_IGNORADAS:
            continue
        etiqueta = f"{personaje} ({ruta}): "
        for trozo in empaquetar_parrafos(limpiar(contenido), MAX_CHARS_SECCION, OVERLAP):
            chunks.append(Chunk(etiqueta + trozo, personaje, ruta))
    return chunks


ESTRATEGIAS = {
    "1. página entera": chunk_pagina_entera,
    "2. fijo 1000/200": chunk_fijo,
    "3. secciones": chunk_secciones,
}
