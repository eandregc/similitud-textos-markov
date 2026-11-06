"""
Similitud entre un texto original, uno sintetico (Markov) y otros textos.

Uso:
    python main.py                 # corrida completa (descarga el corpus la primera vez)
    python main.py --orden 3       # cadena de Markov de orden 3
    python main.py --palabras 30000 --semilla 7
"""

import argparse
import json
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data import CORPUS, RAIZ, TEXTOS, cargar_todo
from model import (dividir_mitades, entrenar_markov, generar, mezclar_palabras,
                   similitud_global, similitud_por_trozos)

REF = "Quijote A (original)"


def titulo(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orden", type=int, default=2, help="orden de la cadena de Markov")
    ap.add_argument("--palabras", type=int, default=20000, help="palabras del texto sintetico")
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()

    textos = cargar_todo()
    qa, qb = dividir_mitades(textos["quijote"])
    print(f"Quijote A: {len(qa.split()):,} palabras | Quijote B: {len(qb.split()):,} palabras")

    # --- Texto sintetico: cadena de Markov entrenada solo con Quijote A ---------
    titulo(f"Cadena de Markov de palabras, orden {a.orden}")
    sint = generar(entrenar_markov(qa, a.orden), a.palabras, a.semilla)
    (RAIZ / "sintetico").mkdir(exist_ok=True)
    (RAIZ / "sintetico" / "quijote_sintetico.txt").write_text(sint, encoding="utf-8")
    print(sint[:600] + " ...")

    # Sinteticos de otros autores y linea base de palabras revueltas (contraste)
    sint_gal = generar(entrenar_markov(textos["marianela"], a.orden), a.palabras, a.semilla)
    (RAIZ / "sintetico" / "marianela_sintetico.txt").write_text(sint_gal, encoding="utf-8")

    docs = {
        REF: qa,
        "Quijote B (misma obra)": qb,
        "Novelas ejemplares (Cervantes)": textos["novelas"],
        "Marianela (Galdos)": textos["marianela"],
        "Platero y yo (J. R. Jimenez)": textos["platero"],
        "Sintetico Markov de Quijote A": sint,
        "Sintetico Markov de Marianela": sint_gal,
        "Quijote A con palabras revueltas": mezclar_palabras(qa, a.semilla),
    }

    resultado = {}
    configs = {
        "palabras": dict(),
        "caracteres 3-5": dict(analyzer="char_wb", ngram_range=(3, 5), max_features=150000),
        "bigramas de palabras": dict(ngram_range=(2, 2), min_df=1),
    }

    # --- tf-idf global + coseno --------------------------------------------------
    for nombre, kw in configs.items():
        titulo(f"Coseno contra el original, tf-idf global ({nombre})")
        s = similitud_global(docs, REF, **kw)
        resultado[f"global_{nombre}"] = {k: round(float(v), 4) for k, v in s.items()}
        for k, v in sorted(s.items(), key=lambda x: -x[1]):
            print(f"{k:38}{v:8.4f}")

    # --- tf-idf por trozos: media y dispersion ----------------------------------
    titulo("Coseno por trozos de 1500 palabras contra el centroide del original")
    trz = similitud_por_trozos({k: v for k, v in docs.items()}, REF, tam=1500)
    resultado["trozos"] = {}
    for k, v in trz.items():
        if k == REF:
            continue
        resultado["trozos"][k] = {"media": round(float(v.mean()), 4),
                                  "desv": round(float(v.std()), 4), "n": int(len(v))}
        print(f"{k:38}{v.mean():8.4f} +- {v.std():.4f}  (n={len(v)})")

    # --- matriz de similitud entre todos los pares ------------------------------
    nombres = list(docs)
    X = TfidfVectorizer().fit_transform([docs[n] for n in nombres])
    M = cosine_similarity(X)

    # --- figuras ---------------------------------------------------------------
    fig_dir = RAIZ / "figuras"
    fig_dir.mkdir(exist_ok=True)
    corto = [n.replace(" (", "\n(") for n in nombres]

    fig, ax = plt.subplots(figsize=(8, 6.2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(nombres)), [str(i + 1) for i in range(len(nombres))])
    ax.set_yticks(range(len(nombres)), [f"{i + 1}. {n}" for i, n in enumerate(nombres)], fontsize=8)
    for i, j in combinations(range(len(nombres)), 2):
        pass
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.6 else "black")
    fig.colorbar(im, label="similitud coseno")
    fig.tight_layout()
    fig.savefig(fig_dir / "matriz_similitud.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ks = [k for k in nombres if k != REF]
    ax.boxplot([trz[k] for k in ks], orientation="horizontal", tick_labels=ks)
    ax.set_xlabel("similitud coseno de cada trozo contra el original")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fig_dir / "trozos_boxplot.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ancho = 0.27
    y = np.arange(len(ks))
    for i, nom in enumerate(configs):
        ax.barh(y + (i - 1) * ancho, [resultado[f"global_{nom}"][k] for k in ks], ancho, label=nom)
    ax.set_yticks(y, ks)
    ax.invert_yaxis()
    ax.set_xlabel("similitud coseno contra el original")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "global_representaciones.png", dpi=160)
    plt.close(fig)


    # --- efecto del orden de la cadena de Markov ---------------------------------
    titulo("Efecto del orden de la cadena (sintetico de Quijote A vs. original)")
    resultado["ordenes"] = {}
    print(f"{'orden':>6}{'coseno palabras':>18}{'coseno bigramas':>18}{'% frases copiadas':>20}")
    base = set(qa.replace("\n", " ").split(". "))
    for o in (1, 2, 3, 4):
        s_o = generar(entrenar_markov(qa, o), a.palabras, a.semilla)
        d2 = {REF: qa, "s": s_o}
        cp = similitud_global(d2, REF)["s"]
        cb = similitud_global(d2, REF, ngram_range=(2, 2))["s"]
        frases = [f.strip() for f in s_o.split(". ") if len(f.split()) > 5]
        copia = 100 * sum(f in base for f in frases) / max(len(frases), 1)
        resultado["ordenes"][o] = {"palabras": round(float(cp), 4), "bigramas": round(float(cb), 4),
                                   "copiadas_pct": round(copia, 1)}
        print(f"{o:>6}{cp:18.4f}{cb:18.4f}{copia:20.1f}")
    (RAIZ / "resultados.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print("\nListo: resultados.json, figuras/ y sintetico/ actualizados.")


if __name__ == "__main__":
    main()

