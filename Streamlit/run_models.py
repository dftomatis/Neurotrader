import json
import sys
import random
import pandas as pd
import os

# 📅 Determina la tendencia del mercado según el día
def get_market_trend_from_date(date_str):
    day = int(date_str.split("-")[2])
    return "bullish" if day % 2 == 0 else "bearish"

# 🎯 Decide el sentimiento objetivo según la tendencia
def get_sentiment_by_trend(trend):
    if random.random() < 0.3:
        return "neutral"
    return "positive" if trend == "bullish" else "negative"

# 📰 Extrae una noticia real según el sentimiento, o usa neutral como respaldo
def get_news_by_sentiment(sentiment_target, path="bitcoin_news_etiquetado.csv", sample_size=1000):
    chunks = pd.read_csv(path, usecols=["article_text", "sentiment"], chunksize=10000)
    candidates = []

    for chunk in chunks:
        filtered = chunk[chunk["sentiment"] == sentiment_target]
        candidates.extend(filtered["article_text"].dropna().tolist())
        if len(candidates) >= sample_size:
            break

    # Si no se encuentra ninguna, buscar neutrales
    if not candidates:
        chunks = pd.read_csv(path, usecols=["article_text", "sentiment"], chunksize=10000)
        for chunk in chunks:
            neutral_news = chunk[chunk["sentiment"] == "neutral"]
            candidates.extend(neutral_news["article_text"].dropna().tolist())
            if len(candidates) >= sample_size:
                break

    return random.choice(candidates) if candidates else "No news available."

# 📈 Genera indicadores técnicos simulados
def generate_indicators():
    return {
        "RSI": round(random.uniform(20, 80), 2),
        "MACD": round(random.uniform(-2, 2), 2),
        "Volatility": round(random.uniform(0.5, 3.0), 2)
    }

# 🔄 Carga modelo base del 2 de septiembre
def load_reference_model(model_name):
    path = f"results_{model_name.lower()}_2025-09-02.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# 🔮 Genera resultados para modelos técnicos
def generate_technical_model(name, date_str):
    reference = load_reference_model(name)
    if reference:
        base_buy = reference["probabilities"]["buy"]
        variation = random.uniform(-0.05, 0.05)
        buy = round(min(max(base_buy + variation, 0.25), 0.57), 4)
    else:
        buy = round(random.uniform(0.25, 0.57), 4)

    hold = round(random.uniform(0.3, 0.7), 4)
    total = buy + hold
    if total > 1:
        buy = round(buy / total, 4)
        hold = round(hold / total, 4)

    return {
        "date": date_str,
        "model": name,
        "probabilities": {
            "buy": buy,
            "hold": hold
        },
        "indicadores": generate_indicators()
    }

# 🧠 Genera resultado para FinBERT (NLP)
def generate_nlp_model(date_str):
    trend = get_market_trend_from_date(date_str)
    sentiment_target = get_sentiment_by_trend(trend)
    news_text = get_news_by_sentiment(sentiment_target)

    buy = round(random.uniform(0.25, 0.57), 4)
    hold = round(random.uniform(0.3, 0.7), 4)
    total = buy + hold
    if total > 1:
        buy = round(buy / total, 4)
        hold = round(hold / total, 4)

    return {
        "date": date_str,
        "model": "NLP",
        "probabilities": {
            "buy": buy,
            "hold": hold
        },
        "noticia": news_text,
        "sentiment": sentiment_target
    }

# 🚀 Punto de entrada
def main():
    if len(sys.argv) < 2:
        print("⚠️ Debes proporcionar una fecha en formato YYYY-MM-DD")
        return

    date_str = sys.argv[1]
    models = ["rfc", "lstm", "xgb", "nlp"]

    for model in models:
        filename = f"results_{model}_{date_str}.json"

        # 🛡️ Proteger los resultados reales del 2 de septiembre
        if date_str == "2025-09-02" and os.path.exists(filename):
            print(f"🔒 Archivo real existente para {model} en {date_str}. No se sobrescribe.")
            continue

        # 🧠 Generar resultados
        if model == "nlp":
            output = generate_nlp_model(date_str)
        else:
            output = generate_technical_model(model.upper(), date_str)

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

    print(f"✅ Resultados procesados para {date_str}")

if __name__ == "__main__":
    main()
