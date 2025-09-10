import streamlit as st
import json
import os
import subprocess

st.set_page_config(page_title="🧠 NeuroTrader", layout="centered")

st.title("📈 NeuroTrader Dashboard")
st.write("Selecciona una fecha para visualizar las probabilidades de compra y análisis técnico.")

# 📅 Fecha seleccionada
fecha = st.date_input("Fecha de análisis", value=None)

if fecha:
    fecha_str = fecha.strftime("%Y-%m-%d")

    # 📁 Diccionario de modelos y sus archivos
    model_files = {
        "RFC/SGD": f"results_rfc_{fecha_str}.json",
        "LSTM": f"results_lstm_{fecha_str}.json",
        "FinBERT (NLP)": f"results_nlp_{fecha_str}.json",
        "XGBoost": f"results_xgb_{fecha_str}.json"
    }

    # 🔍 Verificar si todos los archivos existen
    missing_files = [f for f in model_files.values() if not os.path.exists(f)]

    # 🧠 Ejecutar modelos solo si faltan archivos
    if missing_files:
        try:
            subprocess.run(["python", "run_models.py", fecha_str], check=True)
        except Exception as e:
            st.error(f"❌ Error al ejecutar los modelos: {e}")
    else:
        st.success("✅ Resultados ya generados para esta fecha.")

        # 🎛️ Selector de modelo
    selected_model = st.sidebar.radio("Selecciona un modelo", list(model_files.keys()))
    file_path = model_files[selected_model]

    # 📊 Visualización de resultados
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            results = json.load(f)

        st.subheader(f"📊 Resultados del modelo: {selected_model}")
        st.markdown(f"**Fecha de inferencia:** `{results.get('date', fecha_str)}`")

        # 🔢 Mostrar solo el porcentaje de compra
        buy_prob = results.get("probabilities", {}).get("buy", None)

        # Fallback si las claves son class_1 (modelos reales)
        if buy_prob is None and "class_1" in results.get("probabilities", {}):
            buy_prob = results["probabilities"]["class_1"]

        if buy_prob is not None:
            buy_percent = round(buy_prob * 100, 2)
            st.markdown(f"**Probabilidad de compra:** `{buy_percent}%`")

            # 💬 Sugerencia basada en umbral
            if buy_percent < 35:
                st.markdown("💬 **Sugerencia:** Venta recomendada")
            elif 36 <= buy_percent <= 44:
                st.markdown("💬 **Sugerencia:** Mantener posición")
            elif buy_percent > 47:
                st.markdown("💬 **Sugerencia:** Posible oportunidad de compra")
            else:
                st.markdown("💬 **Sugerencia:** Zona ambigua, revisar otros indicadores")
        else:
            st.warning("⚠️ No se encontró probabilidad de compra.")

        # 📈 Indicadores técnicos (si existen)
        if "indicadores" in results:
            st.markdown("📊 **Indicadores técnicos:**")
            for k, v in results["indicadores"].items():
                st.write(f"- {k}: {v}")

        # 📰 Noticia y sentimiento (solo FinBERT)
        if selected_model == "FinBERT (NLP)":
            if "noticia" in results:
                st.markdown("📰 **Noticia analizada:**")
                st.write(results["noticia"])
            if "sentiment" in results:
                st.markdown(f"🔍 **Sentimiento detectado:** `{results['sentiment'].capitalize()}`")
    else:
        st.warning(f"⚠️ No se encontró el archivo para {selected_model} en la fecha {fecha_str}.")
else:
    st.info("Selecciona una fecha para comenzar.")
