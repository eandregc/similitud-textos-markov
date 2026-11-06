"""
Modelo de la actividad: cadena de Markov (markovify), tf-idf y similitud coseno.
"""

import random
import re

import markovify
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def palabras(texto):
    return re.findall(r"[a-záéíóúüñ]+", texto.lower())


def dividir_mitades(texto):
    """Parte un texto en dos mitades por parrafos (A = primera, B = segunda)."""
    parrafos = texto.split("\n")
    m = len(parrafos) // 2
    return "\n".join(parrafos[:m]), "\n".join(parrafos[m:])


def entrenar_markov(texto, orden=2):
    """Cadena de Markov de palabras de orden `orden` (markovify.Text)."""
    return markovify.Text(texto, state_size=orden, well_formed=False)


def generar(modelo, n_palabras, semilla=0):
    """Genera frases sinteticas hasta juntar n_palabras palabras."""
    random.seed(semilla)
    frases, total = [], 0
    while total < n_palabras:
        f = modelo.make_sentence(tries=100)
        if f:
            frases.append(f)
            total += len(f.split())
    return " ".join(frases)


def mezclar_palabras(texto, semilla=0):
    """Linea base: las mismas palabras en orden totalmente aleatorio."""
    p = texto.split()
    random.Random(semilla).shuffle(p)
    return " ".join(p)


def trozos(texto, tam=1500):
    """Corta un texto en trozos de `tam` palabras (descarta el sobrante)."""
    p = texto.split()
    return [" ".join(p[i:i + tam]) for i in range(0, len(p) - tam + 1, tam)]


def similitud_global(docs, referencia, **kw):
    """tf-idf de documentos completos; coseno de cada uno contra la referencia."""
    nombres = list(docs)
    vec = TfidfVectorizer(**kw)
    X = vec.fit_transform([docs[n] for n in nombres])
    s = cosine_similarity(X[nombres.index(referencia)], X).ravel()
    return dict(zip(nombres, s))


def similitud_por_trozos(docs, referencia, tam=1500, **kw):
    """
    Corta cada texto en trozos y ajusta el tf-idf sobre todos (asi el idf si
    discrimina). Devuelve, por texto, el coseno de cada uno de sus trozos
    contra el centroide de los trozos de la referencia.
    """
    nombres = list(docs)
    por_doc = {n: trozos(docs[n], tam) for n in nombres}
    todos = [t for n in nombres for t in por_doc[n]]
    vec = TfidfVectorizer(**kw)
    X = vec.fit_transform(todos)
    idx, i = {}, 0
    for n in nombres:
        idx[n] = (i, i + len(por_doc[n]))
        i += len(por_doc[n])
    a, b = idx[referencia]
    centroide = np.asarray(X[a:b].mean(axis=0))
    return {n: cosine_similarity(X[i0:i1], centroide).ravel() for n, (i0, i1) in idx.items()}
