# Similitud de textos con cadenas de Markov, tf-idf y distancia del coseno

¿Qué tan parecido es un texto generado con una cadena de Markov a su original? ¿Y a otros textos del mismo autor o de otros? En este proyecto se toma **Don Quijote de la Mancha** como cuerpo de interés, se genera texto sintético con [Markovify](https://github.com/jsvine/markovify), se pasan todos los textos a vectores **tf-idf** y se mide la **similitud coseno**. El objetivo es ver cómo cambia la similitud según el texto con el que se compara y si una similitud alta prueba que dos textos son del mismo autor.

## Textos

Todos son de dominio público (Project Gutenberg) y se descargan solos la primera vez que se corre el programa a `corpus/`.

| Texto | Papel en el experimento |
|---|---|
| Quijote A (primera mitad de *Don Quijote*) | **Original / referencia**. Con él se entrena la cadena de Markov |
| Quijote B (segunda mitad) | Misma obra y mismo autor, pero no vista por el modelo |
| *Novelas ejemplares* (Cervantes) | Mismo autor, otra obra |
| *Marianela* (Pérez Galdós) | Otro autor del mismo idioma, siglo XIX |
| *Platero y yo* (J. R. Jiménez) | Otro autor y otro registro (prosa poética) |
| Sintético de Quijote A | Texto de Markov entrenado con Quijote A (`sintetico/quijote_sintetico.txt`) |
| Sintético de Marianela | Texto de Markov entrenado con *Marianela* (contraste) |
| Quijote A con palabras revueltas | Mismas palabras en orden aleatorio (línea base) |

## Método

1. **Cadena de Markov:** `markovify.Text` de palabras con `state_size = 2` (la siguiente palabra depende de las dos anteriores). Se generan 20 000 palabras.
2. **Representación:** tf-idf con tres variantes: palabras sueltas, n-gramas de caracteres (3 a 5) y bigramas de palabras.
3. **Similitud:** coseno entre el vector de cada texto y el del original, `cos(a, b) = a·b / (‖a‖ ‖b‖)`.
4. **Dos formas de calcularlo:**
   * *Global:* un vector por texto completo.
   * *Por trozos:* cada texto se corta en trozos de 1 500 palabras, el tf-idf se ajusta sobre todos los trozos (así el idf sí discrimina) y se compara cada trozo contra el centroide del original. Da media y desviación, no un solo número.
5. **Orden de la cadena:** se repite la generación con órdenes 1 a 4 para ver cuánto copia el modelo del original.

## Estructura

```
Similitud Textos Markov/
├── src/
│   ├── data.py     # textos, descarga y limpieza del corpus
│   ├── model.py    # Markov, tf-idf y similitud coseno
│   └── main.py     # experimento completo, figuras y resultados
├── corpus/         # textos originales (se descargan solos)
├── sintetico/      # textos generados con Markov
├── figuras/        # matriz de similitud, cajas por trozos, comparación de representaciones
├── resultados.json
├── requirements.txt
└── README.md
```

## Uso

```
pip install -r requirements.txt
cd src
python main.py                  # corrida completa
python main.py --orden 3        # cadena de Markov de orden 3
python main.py --palabras 30000 --semilla 7
```

## Resultados principales (orden 2, semilla 0)

| Texto | Coseno global (palabras) | Coseno global (bigramas) | Coseno por trozos (media ± desv.) |
|---|:---:|:---:|:---:|
| Quijote B (misma obra) | 0.992 | 0.882 | 0.740 ± 0.044 |
| Sintético de Quijote A | 0.997 | 0.801 | 0.840 ± 0.012 |
| Novelas ejemplares | 0.970 | 0.683 | 0.681 ± 0.045 |
| Marianela | 0.925 | 0.447 | 0.628 ± 0.039 |
| Sintético de Marianela | 0.926 | 0.394 | 0.691 ± 0.014 |
| Platero y yo | 0.484 | 0.157 | 0.241 ± 0.119 |
| Quijote A con palabras revueltas | 1.000 | 0.240 | 0.837 ± 0.013 |

La conclusión corta: la similitud depende mucho de **qué se mide** (palabras sueltas contra bigramas) y de **con qué se compara**; que sea alta no demuestra que dos textos sean del mismo autor. El análisis completo está en el documento Word de la actividad.
