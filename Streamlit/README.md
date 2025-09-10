🧠 NeuroTrader: Panel de Predicción y Sentimiento Financiero
NeuroTrader es una aplicación interactiva construida con Streamlit que visualiza las predicciones de varios modelos de machine learning entrenados para analizar el comportamiento del mercado de Bitcoin y su sentimiento asociado. Combina análisis técnico, inferencia de sentimiento con NLP y predicción de series temporales para ofrecer una visión integral de las señales del mercado.

📦 Modelos incluidos
Modelo	Descripción
RFC/SGD	Clasificadores Random Forest y SGD usando indicadores técnicos.
LSTM	Red neuronal secuencial entrenada con datos de precios históricos.
FinBERT (NLP)	Análisis de sentimiento en noticias financieras usando FinBERT.
XGBoost	Modelo de boosting entrenado con 14 características del mercado.
📅 Fecha actual de inferencia
Todos los modelos están evaluados con datos del 2 de septiembre de 2025.

🚀 Cómo ejecutar localmente desde la terminal (shell)
1. Clonar el repositorio
bash
git clone https://github.com/dftomatis/Neurotrader.git
cd Neurotrader
2. Crear un entorno virtual (opcional pero recomendado)
bash
python3 -m venv env
source env/bin/activate  # En Linux/macOS
env\Scripts\activate     # En Windows
3. Instalar las dependencias
bash
pip install --upgrade pip
pip install -r requirements.txt
4. Ejecutar la aplicación Streamlit
bash
streamlit run appneurotrader.py
Esto abrirá automáticamente la aplicación en tu navegador por defecto en http://localhost:8501.

bash
pip install --upgrade streamlit
📁 Estructura de archivos
Código
Neurotrader/
├── appneurotrader.py      # Aplicación principal de Streamlit
├── requirements.txt       # Lista de dependencias
├── results_rfc.json       # Salida del modelo RFC/SGD
├── results_lstm.json      # Salida del modelo LSTM
├── results_nlp.json       # Salida del modelo FinBERT
├── results_xgb.json       # Salida del modelo XGBoost
└── README.md              # Descripción del proyecto
