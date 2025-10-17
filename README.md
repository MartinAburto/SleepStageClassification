# Sleep Stage Classification Project / Proyecto de Clasificación de Etapas del Sueño

**Course / Curso:** EL — Automatic Sleep Stage Classification using EEG (Sleep-EDFx)

---

## English

### Overview
This repository provides a **lightweight deep learning pipeline** for **automatic sleep stage classification** using EEG signals from **Sleep-EDFx**.  
The approach is inspired by the **LightSleepNet** architecture ([Zhou et al., EMBC 2021](https://doi.org/10.1109/EMBC46164.2021.9629878)), which leverages **spectrogram-based features** and a compact CNN design for rapid inference.  
Our implementation introduces **modifications and optimizations** to adapt the model for the Sleep-EDFx dataset and improve reproducibility.

### Goals
- Download and organize the Sleep-EDFx dataset locally (**not committed to Git**).
- Preprocess EEG signals and compute **time-frequency representations** (spectrograms).
- Train and evaluate a modified **LightSleepNet-inspired model** with subject-wise cross-validation.
- Report metrics (Accuracy, F1-macro, Cohen’s κ) and generate figures with consistent formatting.

### Repository Structure
```
SleepStageClassification/
├── data/        # Raw data (not tracked); only data/README.md is committed
├── notebooks/   # Jupyter notebooks (EDA, training, evaluation)
├── src/         # (optional) reusable modules: preprocessing, modeling
├── results/     # (optional) saved figures, metrics, reports
├── requirements.txt
├── .gitignore
└── README.md
```

### Setup (Python ≥3.10)
```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scriptsctivate

# 2) Install dependencies
pip install -U pip
pip install -r requirements.txt

# 3) (Optional) Enable Jupyter kernel
python -m ipykernel install --user --name sleepnet --display-name "SleepNet"
```

### Data: Sleep-EDFx
- Homepage: https://physionet.org/content/sleep-edfx/1.0.0/
- **Do not commit raw data.** Place downloaded EDF and hypnograms under `data/`.
- (Optional) Add a `data/README.md` with download instructions.

### Quickstart
```bash
# Launch notebooks
jupyter lab   # or: jupyter notebook
# Then open notebooks/your_notebook.ipynb
```

### References
- D. Zhou, Q. Xu, J. Wang, J. Zhang, G. Hu, L. Kettunen, Z. Chang, and F. Cong, “LightSleepNet: A Lightweight Deep Model for Rapid Sleep Stage Classification with Spectrograms,” in Proc. 43rd Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC), 2021, pp. 43–46, doi: 10.1109/EMBC46164.2021.9629878.
- PhysioNet Sleep-EDFx dataset (link above).

---

## Español

### Descripción
Este repositorio implementa una **arquitectura ligera basada en redes profundas** para la **clasificación automática de etapas del sueño** usando señales EEG del dataset **Sleep-EDFx**.  
El enfoque se inspira en **LightSleepNet** (Zhou et al., EMBC 2021), que utiliza **espectrogramas** y una CNN compacta para inferencia rápida.  
Nuestra versión incluye **modificaciones y ajustes** para mejorar la adaptabilidad y la reproducibilidad.

### Objetivos
- Descargar y organizar Sleep-EDFx localmente (**no subir datos crudos al repo**).
- Preprocesar señales EEG y calcular **representaciones tiempo–frecuencia** (espectrogramas).
- Entrenar y evaluar un modelo inspirado en **LightSleepNet** con validación sujeto-a-sujeto.
- Reportar métricas (Accuracy, F1-macro, κ) y generar figuras con formato consistente.

### Estructura del repositorio
(ver bloque anterior)

### Configuración (Python ≥3.10)
(ver bloque anterior)

### Datos: Sleep-EDFx
- Sitio: https://physionet.org/content/sleep-edfx/1.0.0/
- **No comprometer datos crudos en Git.** Colócalos bajo `data/`.

### Inicio rápido
(ver bloque anterior)

### Referencias
- Zhou et al., *LightSleepNet*, EMBC 2021.
- PhysioNet Sleep-EDFx.
