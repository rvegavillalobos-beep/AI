import streamlit as st
import time
import os
import pandas as pd

from neat_agent import run_neat, render_genome

st.set_page_config(page_title="Flappy Bird NEAT", layout="centered")
st.title("🐦 Flappy Bird entrenado con NEAT")

st.markdown("""
Este agente aprende a jugar Flappy Bird usando **algoritmos genéticos (NEAT)**.
Cada generación es una población de pájaros; los que sobreviven más tiempo
y pasan más tuberías tienen mayor probabilidad de "reproducirse" en la
siguiente generación.
""")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")

if "fitness_history" not in st.session_state:
    st.session_state.fitness_history = []
if "winner" not in st.session_state:
    st.session_state.winner = None
if "config" not in st.session_state:
    st.session_state.config = None

generations = st.slider("Número de generaciones a entrenar", 5, 100, 20)

if st.button("🚀 Entrenar"):
    st.session_state.fitness_history = []
    progress_bar = st.progress(0)
    chart_placeholder = st.empty()

    def callback(fitness):
        st.session_state.fitness_history.append(fitness)
        progress_bar.progress(min(len(st.session_state.fitness_history) / generations, 1.0))
        chart_placeholder.line_chart(pd.DataFrame({"mejor fitness": st.session_state.fitness_history}))

    with st.spinner("Entrenando población..."):
        winner, stats, config = run_neat(CONFIG_PATH, generations, progress_callback=callback)

    st.session_state.winner = winner
    st.session_state.config = config
    st.success("¡Entrenamiento completo!")

if st.session_state.winner is not None:
    st.subheader("🏆 Mejor genoma entrenado")
    if st.button("▶️ Ver al mejor pájaro jugar"):
        frames = render_genome(st.session_state.winner, st.session_state.config)
        placeholder = st.empty()
        for frame in frames:
            placeholder.image(frame)
            time.sleep(0.03)
        st.info(f"Sobrevivió {len(frames)} frames y pasó tuberías con éxito.")
