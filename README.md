# Proyecto TD 2025/2026
## Detección de desinformación y análisis de polarización en redes sociales (RumourEval 2019)

Proyecto desarrollado para la asignatura **Tratamiento de Datos / Text Data**  
Máster en Ingeniería de Telecomunicación (UC3M), curso 2025–2026.

---

# 1. Introducción

Las redes sociales facilitan la difusión rápida de información, pero también de **rumores y desinformación**.  
En este proyecto abordamos la tarea de **stance detection** (postura) como aproximación al análisis de desinformación y polarización:

- Dado un **tuit fuente** (rumor) y una **respuesta**, predecir la **postura** de la respuesta.
- Comparar representaciones y modelos de PLN:
  - Modelos clásicos (KNN)
  - Redes neuronales (CNN 1D en PyTorch)
  - Transformer preentrenado con fine-tuning (DistilBERT)
- Extensión: baseline **léxica sencilla** (reglas por palabras clave) para medir un “suelo” interpretable.

---

# 2. Conjunto de datos

Dataset: **RumourEval 2019** (subconjunto en inglés).

Clases:

| Clase   | Descripción           |
|---------|-----------------------|
| support | Apoya el rumor        |
| deny    | Niega el rumor        |
| query   | Pregunta o duda       |
| comment | Comenta sin postura   |

## Limpieza de etiquetas
Se detectaron `NaN` en `label` en train antes de limpiar.

- NaN en `label` (antes de limpiar): train=2, val=0, test=0  
- NaN en `label` (después de limpiar): train=0, val=0, test=0  

## Distribución de clases (real)
**Train (4 877 muestras)**

| Clase   | Nº muestras | Proporción |
|---------|------------:|-----------:|
| comment | 3 495       | 0.717      |
| support | 642         | 0.132      |
| query   | 373         | 0.076      |
| deny    | 367         | 0.075      |

**Val (1 440 muestras)**

| Clase   | Nº muestras | Proporción |
|---------|------------:|-----------:|
| comment | 1 174       | 0.815      |
| query   | 114         | 0.079      |
| deny    | 79          | 0.055      |
| support | 73          | 0.051      |

**Test (1 675 muestras)**

| Clase   | Nº muestras | Proporción |
|---------|------------:|-----------:|
| comment | 1 405       | 0.839      |
| support | 104         | 0.062      |
| deny    | 100         | 0.060      |
| query   | 66          | 0.039      |

Archivos:

```text
data/datasets/
    rumoureval2019_train.csv
    rumoureval2019_val.csv
    rumoureval2019_test.csv
```

---

# 3. Objetivos

## 3.1 Objetivos básicos
- AEP del dataset (distribución, ejemplos, limpieza).
- Comparar representaciones:
  - TF–IDF
  - Word2Vec (media)
  - Sentence-BERT (embeddings contextuales)
- Entrenar y evaluar:
  - KNN (scikit-learn)
  - CNN 1D (PyTorch)
  - Transformer (DistilBERT) con fine-tuning
- Comparar contra baselines y discutir implicaciones.

## 3.2 Extensión implementada
- **Baseline léxica sencilla (reglas por palabras clave)** para aproximar postura de forma interpretable.
- Evaluación completa con *classification report*.

---

# 4. Estructura del repositorio

```text
Tratamiento-de-datos-2025-2026/
├── data/
│   └── datasets/
│       ├── rumoureval2019_train.csv
│       ├── rumoureval2019_val.csv
│       └── rumoureval2019_test.csv
├── notebooks/
│   └── Knn_3_modelos_vectorizacion.ipynb
├── scripts/
│   └── project_run.py
├── src/
│   └── project_run.py
└── README.md
```

---

# 5. Instalación y ejecución

## 5.1 Requisitos
- Python 3.10+
- Recomendado: GPU (CUDA) para CNN/Transformer (si está disponible)

## 5.2 Instalación

> Este repositorio no incluye `requirements.txt` para mantener la entrega mínima.  
> Si es necesario reproducir desde cero, un conjunto típico de dependencias es:

```bash
pip install numpy pandas scikit-learn gensim sentence-transformers datasets matplotlib
```

## 5.3 Ejecución

**Usando el notebook de Jupyter**

1. Lanzar Jupyter:
   ```bash
   jupyter notebook
   ```
2. Abrir `Knn_3_modelos_vectorizacion.ipynb`.
3. Ejecutar las celdas en orden (o **Kernel → Restart & Run All**).
4. El notebook imprime:
   - distribución de clases,
   - resultados por modelo,
   - *classification reports*,
   - resumen final de métricas.

**Usando los Scripts**
El projecto se puede ejecutar desde un Entorno de desarrollo integrado (IDE) , los usado han sido VSCode y pycharm.
Tendremo que tener un interprete python acorde con las especificaciones anteriormente mencionadas y instalar los paquetes (librerias necesarias).
Para ejecutar desde el IDE simplemente correrremos el unico script del proyecto presente en la carpeta *scripts* llamado *project_run.py*

> Adicionalmente si se quiere ejecutar en un CMD tendriamos que tener en el equipo las librerias necesarias anteriormente mencionadas
> Navegar hasta la carpeta *scripts* y abrir un CMD dentro de esta en el que ejecutaremos el siguiente comando *python project_run.py* 
> Probablemente haya problemas a la hora de la representacion de las graficas ya que la libreria **matplotlib** suele tener problemas al lanzar graficas desde el cmd.

---

# 6. Metodología

## 6.1 Preprocesamiento
- Limpieza básica (URLs, menciones, etc.).
- Construcción del texto combinado:

```text
source_text [SEP] reply_text
```

- Codificación de etiquetas con `LabelEncoder`:
`['comment', 'deny', 'query', 'support']`.

## 6.2 Representaciones
- **TF–IDF:** `TfidfVectorizer` (unigramas + bigramas), normalización.
- **Word2Vec:** gensim, embedding dim=100, documento = media de embeddings.
- **Sentence-BERT:** embeddings contextuales por lotes (sentence-transformers).

## 6.3 Modelos
- **KNN:** búsqueda de `k ∈ {1,3,5,7,9}` por validación.
- **CNN 1D (PyTorch):**
  - Conv1D + ReLU + Pooling + Dropout + FC
  - `CrossEntropyLoss` con **pesos por clase** (para mitigar desbalance)
- **Transformer (DistilBERT):**
  - `distilbert-base-uncased`
  - fine-tuning 3 épocas, `max_length=128`, entrenamiento en `cuda` si disponible
- **Baseline léxica (extensión):**
  - reglas por palabras clave para `query`, `deny`, `support` y fallback a `comment`.

---

# 7. Resultados (reales)


## 7.1 Baseline de mayoría (always `comment`)
- Accuracy en test: **0.8388**

> Es muy alta porque en test `comment` = 83.9% de las muestras.

---

## 7.2 KNN

### TF–IDF + KNN
- Mejor k en validación: **9** (acc val = 0.7903)
- Accuracy en test: **0.8299**

### Word2Vec + KNN
- Mejor k en validación: **9** (acc val = 0.8125)
- Accuracy en test: **0.8370**

### Sentence-BERT + KNN
- Mejor k en validación: **9** (acc val = 0.8035)
- Accuracy en test: **0.8304**

**Resumen KNN:**

| Representación | k óptimo | Acc. validación | Acc. test |
|---------------|---------:|----------------:|----------:|
| TF–IDF        | 9        | 0.7903          | 0.8299    |
| Word2Vec      | 9        | 0.8125          | 0.8370    |
| Sentence-BERT | 9        | 0.8035          | 0.8304    |

> Observación: en KNN, aunque la accuracy es alta, el *classification report* indica que el modelo tiende a predecir casi siempre `comment`, dejando f1≈0 en clases minoritarias.

---

## 7.3 CNN (PyTorch)

### TF–IDF + CNN
- Mejor acc validación: **0.8153**
- Accuracy test: **0.8299**

### Word2Vec + CNN (mejorada)
- Mejor acc validación: **0.7243**
- Accuracy test: **0.7003**
- Mejora cualitativa: ya aparecen predicciones en `deny` y `query`, aunque el rendimiento global baja.

### Sentence-BERT + CNN
- Mejor acc validación: **0.8125**
- Accuracy test: **0.7666**
- *Classification report* muestra f1 no nulo en minoritarias (aunque sigue siendo bajo).

**Resumen CNN:**

| Representación | Mejor acc. val | Acc. test |
|---------------|----------------:|----------:|
| TF–IDF + CNN  | 0.8153          | 0.8299   |
| Word2Vec + CNN (mejorada) | 0.7243 | 0.7003 |
| Sentence-BERT + CNN | 0.8125 | 0.7666 |

---

## 7.4 Transformer fine-tuned (DistilBERT)

- Mejor acc validación: **0.8264**
- Accuracy test: **0.8251**

*Classification report (TEST) - resumen:*
- `comment`: precision 0.8516, recall 0.9601, f1 0.9026
- `deny`: precision 0.2500, recall 0.0500, f1 0.0833
- `query`: precision 0.4286, recall 0.4091, f1 0.4186
- `support`: precision 0.1250, recall 0.0096, f1 0.0179

> Aunque la accuracy no supera a la baseline de mayoría, este modelo es el que mejor detecta `query` (recall ~0.47) y presenta un comportamiento más útil para estudiar polarización (clases minoritarias).

---

## 7.5 Extensión: baseline léxica sencilla

- Accuracy baseline léxica + URL (test): **0.5624**

*Classification report (TEST) - resumen:*
- `comment`: f1 0.7196
- `deny`: f1 0.1770
- `query`: f1 0.2024
- `support`: f1 0.1565

Interpretación:
- Mejora sustancial frente a la baseline léxica simple.
- Mantiene interpretabilidad y capta señales superficiales (`query` y `deny`), aunque con menor rendimiento que los modelos entrenados.

---

## 7.6 Comparativa global (accuracy)

| Modelo | Acc. test |
|-------|----------:|
| Baseline mayoría (`comment`) | **0.8388** |
| Word2Vec + KNN | 0.8370 |
| Sentence-BERT + KNN | 0.8304 |
| TF–IDF + KNN | 0.8299 |
| TF–IDF + CNN | 0.8299 |
| DistilBERT fine-tuned | 0.8251 |
| Sentence-BERT + CNN | 0.7666 |
| Word2Vec + CNN (mejorada) | 0.7003 |
| Baseline léxica + URL | 0.5624 |


>Acerca de discrepancias entre el codigo y el informe de resultados en funcion del hardware de la maquina los resultados pueden variar ya que se ha
>probado en entornos online como google colab como en varios entornos locales con diferentes specs hardware y los resultados variaban.
>Como criterio hemos presentado en este informe de resultados los mejores datos que se han obtenido.
---

# 8. Discusión

## 8.1 Por qué la accuracy “engaña”
Como `comment` es ~84% del test, un modelo que predice siempre `comment` ya logra **0.8388**.  
Por eso, modelos que “hacen algo” con clases minoritarias pueden bajar en accuracy, pero **ser más útiles**.

## 8.2 Interpretación de resultados
- **KNN**: alta accuracy, pero colapso hacia `comment` (minorías ignoradas).
- **CNN**: con Word2Vec (mejorada) aparecen minorías, pero se sacrifica accuracy; con SBERT la performance global no es buena en esta configuración.
- **Transformer**: mejor detección de `query` y algo de `deny`, aunque `support` sigue siendo el punto débil.

## 8.3 Polarización (conexión con postura)
Las posturas más informativas para polarización (`support` y `deny`) son minoritarias, y requieren:
- optimizar con métricas orientadas a minorías (macro-F1, recall por clase),
- estrategias de re-balanceo,
- y/o modelos más robustos.

---

# 9. Limitaciones y trabajo futuro

- Rebalanceo de clases (oversampling/undersampling, focal loss).
- Selección por **macro-F1** en lugar de accuracy.
- Ajuste de hiperparámetros y regularización.
- Explorar modelos más potentes (RoBERTa/DeBERTa) o enfoques multilingües.
- Incorporar información de conversación/estructura de red.

---

# 10. Ejecución rápida

```bash
jupyter notebook Knn_3_modelos_vectorizacion.ipynb
```

---

# 11. Créditos

Proyecto desarrollado por:

**Pablo Anegon Nuñez 100472728** 

**Mateo Tode Ceballos 100537530**

**Daniel Bahrami Planchuelo 100384032**
