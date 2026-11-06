"""
Textos usados en la actividad. Todos son de dominio publico (Project Gutenberg).
El cuerpo de interes es Don Quijote de la Mancha; el resto sirve para comparar.
"""

import re
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CORPUS = RAIZ / "corpus"

# id de Gutenberg -> (nombre corto, descripcion)
TEXTOS = {
    "quijote": (2000, "Don Quijote de la Mancha (Cervantes, 1605/1615)"),
    "novelas": (61202, "Novelas ejemplares (Cervantes, 1613)"),
    "marianela": (17340, "Marianela (Perez Galdos, 1878)"),
    "platero": (9980, "Platero y yo (Juan Ramon Jimenez, 1914)"),
}


def descargar(nombre):
    """Baja el texto de Gutenberg una sola vez y lo deja en corpus/."""
    destino = CORPUS / f"{nombre}.txt"
    if destino.exists():
        return destino.read_text(encoding="utf-8")
    CORPUS.mkdir(exist_ok=True)
    gid = TEXTOS[nombre][0]
    url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    with urllib.request.urlopen(url, timeout=60) as r:
        crudo = r.read().decode("utf-8")
    texto = limpiar(crudo)
    destino.write_text(texto, encoding="utf-8")
    return texto


def limpiar(crudo):
    """Quita la licencia de Gutenberg y junta los renglones de cada parrafo."""
    ini = re.search(r"\*\*\* ?START OF.*?\*\*\*", crudo)
    fin = re.search(r"\*\*\* ?END OF", crudo)
    cuerpo = crudo[ini.end():fin.start()] if ini and fin else crudo
    cuerpo = cuerpo.replace("\r", "")
    parrafos = [" ".join(p.split()) for p in re.split(r"\n\s*\n", cuerpo)]
    return "\n".join(p for p in parrafos if p)


def cargar_todo():
    return {n: descargar(n) for n in TEXTOS}
