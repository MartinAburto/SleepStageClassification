# Sleep Stage Classification Project / Proyecto de Clasificación de Etapas del Sueño

**Course / Curso:** EL4106-1 — Automatic Sleep Stage Classification using EEG (Sleep-EDFx)

---

## English

### Overview

This repository provides three complementary pipelines for automatic sleep stage classification using signals from the Sleep-EDFx database:

1. **LightSpeedNet (LightSleepNet-inspired CNN)**  
   A lightweight deep learning model based on time-frequency representations (spectrograms) of EEG, adapted from LightSleepNet for efficient training and inference.

2. **Classic ML (Random Forest and SVM)**  
   A traditional machine learning pipeline that extracts hand-crafted features (time, frequency, and wavelet-based descriptors) primarily from EOG signals and trains ensemble and margin-based classifiers.

3. **CoSleepNet-style model (hybrid CNN-LSTM)**  
   A deep model that operates on raw or lightly preprocessed EEG/EOG windows, using a hybrid CNN-LSTM architecture similar to CoSleepNet to handle class imbalance and temporal dependencies.

Together, these three approaches enable comparison of lightweight CNNs, classic ML baselines, and hybrid deep architectures on the same dataset using a unified evaluation protocol.

### Goals

- Download and organize the Sleep-EDFx dataset locally (not committed to version control)
- Preprocess EEG/EOG signals to:
  - Compute spectrograms for LightSpeedNet
  - Extract hand-crafted features for Classic ML (RF/SVM)
  - Prepare raw 30-second windows (with or without DCT) for the CoSleepNet-style model
- Train and evaluate all three approaches using subject-wise cross-validation
- Report and compare metrics (Accuracy, Macro F1, Cohen's κ) with consistent visualization formatting

### Repository Structure
```
SleepStageClassification/
├── data/                     # Raw data (not tracked); only data/README.md is committed
├── notebooks/                # Jupyter notebooks for each approach
│   ├── LightSpeedNet.ipynb   # Lightweight CNN on spectrograms
│   ├── ClassicML.ipynb       # RF and SVM with hand-crafted features
│   └── CoSleepNet.ipynb      # Hybrid CNN-LSTM on raw EEG/EOG
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

### Setup

**Requirements:** Python 3.10 or higher
```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -U pip
pip install -r requirements.txt

# 3. (Optional) Enable Jupyter kernel
python -m ipykernel install --user --name sleepnet --display-name "SleepNet"
```

### Data: Sleep-EDFx

**Homepage:** [https://physionet.org/content/sleep-edfx/1.0.0/](https://physionet.org/content/sleep-edfx/1.0.0/)

**Important:** Raw data files should not be committed to version control. Place downloaded EDF files and hypnograms under the `data/` directory.

You may optionally add a `data/README.md` with download instructions, subject information, and preprocessing notes.

### Notebooks

#### `notebooks/LightSpeedNet.ipynb`
End-to-end pipeline for spectrogram generation, training, and evaluation of the LightSleepNet-inspired model.

#### `notebooks/ClassicML.ipynb`
Feature extraction (primarily from EOG) and training of Random Forest and SVM baselines following the methodology of Ghosh et al. (2020).

#### `notebooks/CoSleepNet.ipynb`
Implementation of a CoSleepNet-style hybrid CNN-LSTM that operates on multi-channel EEG/EOG windows, with experiments on different channel combinations and DCT preprocessing.

### Quick Start
```bash
# Launch Jupyter
jupyter lab   # or: jupyter notebook

# Open the desired notebook, e.g., notebooks/LightSpeedNet.ipynb
```

### References

**LightSleepNet / LightSpeedNet-like approach:**
- D. Zhou, Q. Xu, J. Wang, J. Zhang, G. Hu, L. Kettunen, Z. Chang, and F. Cong, "LightSleepNet: A Lightweight Deep Model for Rapid Sleep Stage Classification with Spectrograms," in *Proc. 43rd Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC)*, 2021, pp. 43–46, doi: [10.1109/EMBC46164.2021.9629878](https://doi.org/10.1109/EMBC46164.2021.9629878).

**Classic ML approach (RF/SVM with EOG features):**
- S. Ghosh, S. Saha, and R. K. Tripathy, "An automated system for sleep stage classification from EOG signals using wavelet-based features and ensemble learning," *Biomedical Signal Processing and Control*, vol. 62, 102074, Feb. 2020, doi: [10.1016/j.bspc.2020.102074](https://doi.org/10.1016/j.bspc.2020.102074).

**CoSleepNet-style hybrid CNN-LSTM:**
- E. Efe and S. Özşen, "CoSleepNet: Automated sleep staging using a hybrid CNN-LSTM network on imbalanced EEG-EOG datasets," *Biomedical Signal Processing and Control*, vol. 80, 104299, 2023.

**Dataset:**
- A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals," *Circulation*, vol. 101, no. 23, pp. e215–e220, 2000.

---

## Español

### Descripción

Este repositorio implementa tres enfoques complementarios para la clasificación automática de etapas del sueño usando señales del conjunto de datos Sleep-EDFx:

1. **LightSpeedNet (CNN ligera tipo LightSleepNet)**  
   Modelo profundo ligero basado en representaciones tiempo-frecuencia (espectrogramas) de EEG, inspirado en LightSleepNet, con una arquitectura compacta y eficiente.

2. **Classic ML (Random Forest y SVM)**  
   Pipeline de aprendizaje automático clásico que utiliza características diseñadas manualmente (descriptores de tiempo, frecuencia y wavelets) principalmente sobre señales EOG, siguiendo la metodología de Ghosh et al. (2020).

3. **Modelo tipo CoSleepNet (CNN-LSTM híbrida)**  
   Arquitectura profunda que opera sobre ventanas crudas o levemente preprocesadas de EEG/EOG, combinando convoluciones con capas LSTM para modelar dependencias temporales y manejar el desbalance entre clases.

Estos tres enfoques permiten comparar, bajo un mismo protocolo experimental, el desempeño de redes ligeras, métodos clásicos y modelos híbridos profundos.

### Objetivos

- Descargar y organizar el conjunto de datos Sleep-EDFx localmente (sin subirlo al control de versiones)
- Preprocesar señales EEG/EOG para:
  - Generar espectrogramas para LightSpeedNet
  - Extraer características para los modelos clásicos (RF/SVM)
  - Construir ventanas crudas de 30 segundos (con o sin DCT) para el modelo tipo CoSleepNet
- Entrenar y evaluar los tres enfoques con validación cruzada sujeto-a-sujeto
- Reportar y comparar métricas (Accuracy, Macro F1, κ de Cohen) con visualizaciones consistentes

### Estructura del repositorio
```
SleepStageClassification/
├── data/                     # Sleep-EDFx local (no se sube a Git)
├── notebooks/
│   ├── LightSpeedNet.ipynb   # Modelo ligero basado en espectrogramas
│   ├── ClassicML.ipynb       # Random Forest y SVM
│   └── CoSleepNet.ipynb      # Modelo híbrido CNN-LSTM
├── requirements.txt          # Dependencias de Python
├── .gitignore
└── README.md
```

### Configuración

**Requisitos:** Python 3.10 o superior
```bash
# 1. Crear y activar el entorno virtual
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Instalar las dependencias
pip install -U pip
pip install -r requirements.txt

# 3. (Opcional) Habilitar el kernel de Jupyter
python -m ipykernel install --user --name sleepnet --display-name "SleepNet"
```

### Datos: Sleep-EDFx

**Sitio web:** [https://physionet.org/content/sleep-edfx/1.0.0/](https://physionet.org/content/sleep-edfx/1.0.0/)

**Importante:** Los archivos de datos crudos no deben comprometerse en el control de versiones. Coloque los archivos EDF e hipnogramas descargados en el directorio `data/`.

Opcionalmente, puede agregar un archivo `data/README.md` con instrucciones de descarga, información de sujetos y notas de preprocesamiento.

### Notebooks

#### `notebooks/LightSpeedNet.ipynb`
Pipeline completo para generación de espectrogramas, entrenamiento y evaluación del modelo inspirado en LightSleepNet.

#### `notebooks/ClassicML.ipynb`
Extracción de características (principalmente de EOG) y entrenamiento de modelos base Random Forest y SVM siguiendo la metodología de Ghosh et al. (2020).

#### `notebooks/CoSleepNet.ipynb`
Implementación de un modelo híbrido CNN-LSTM tipo CoSleepNet que opera sobre ventanas multicanal de EEG/EOG, con experimentos en diferentes combinaciones de canales y preprocesamiento DCT.

### Inicio rápido
```bash
# Iniciar Jupyter
jupyter lab   # o: jupyter notebook

# Abrir el notebook deseado, por ejemplo, notebooks/LightSpeedNet.ipynb
```

### Referencias

**Enfoque LightSleepNet / LightSpeedNet:**
- D. Zhou, Q. Xu, J. Wang, J. Zhang, G. Hu, L. Kettunen, Z. Chang, and F. Cong, "LightSleepNet: A Lightweight Deep Model for Rapid Sleep Stage Classification with Spectrograms," in *Proc. 43rd Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC)*, 2021, pp. 43–46, doi: [10.1109/EMBC46164.2021.9629878](https://doi.org/10.1109/EMBC46164.2021.9629878).

**Enfoque ML clásico (RF/SVM con características EOG):**
- S. Ghosh, S. Saha, and R. K. Tripathy, "An automated system for sleep stage classification from EOG signals using wavelet-based features and ensemble learning," *Biomedical Signal Processing and Control*, vol. 62, 102074, Feb. 2020, doi: [10.1016/j.bspc.2020.102074](https://doi.org/10.1016/j.bspc.2020.102074).

**Modelo híbrido CNN-LSTM tipo CoSleepNet:**
- E. Efe and S. Özşen, "CoSleepNet: Automated sleep staging using a hybrid CNN-LSTM network on imbalanced EEG-EOG datasets," *Biomedical Signal Processing and Control*, vol. 80, 104299, 2023.

---

## License / Licencia

This project is licensed under the MIT License. See the LICENSE file for details.

Este proyecto está licenciado bajo la Licencia MIT. Consulte el archivo LICENSE para más detalles.
