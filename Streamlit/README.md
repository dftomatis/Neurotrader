🧠 NeuroTrader: Financial Sentiment & Prediction Dashboard
NeuroTrader is a Streamlit-based dashboard that visualizes predictions from multiple machine learning models trained to analyze Bitcoin market behavior and sentiment. It combines technical analysis, NLP sentiment inference, and time-series forecasting to offer a multi-angle view of market signals.

📦 Models Included
Model	Description
RFC/SGD	Combines Random Forest and SGD classifiers using technical indicators.
LSTM	Neural network trained on sequential price data to forecast market movement.
FinBERT (NLP)	Sentiment analysis of financial news using FinBERT.
XGBoost	Gradient boosting model trained on 14 market features.
📅 Current Inference Date
All models are currently evaluated on data from September 2, 2025.

🚀 How to Run Locally
Clone the repository:

bash
git clone https://github.com/tuusuario/neurotrader.git
cd neurotrader
Install dependencies:

bash
pip install -r requirements.txt
Launch the app:

bash
streamlit run appneurotrader.py
📁 File Structure
Código
neurotrader/
│
├── appneurotrader.py       # Main Streamlit app
├── requirements.txt        # Dependencies
├── results_rfc.json        # RFC/SGD model output
├── results_lstm.json       # LSTM model output
├── results_nlp.json        # FinBERT sentiment output
├── results_xgb.json        # XGBoost model output
└── README.md               # Project description