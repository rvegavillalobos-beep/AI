import streamlit as st
import time
import os
import io
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
    gen_label = st.empty()

    def callback(fitness):
        st.session_state.fitness_history.append(fitness)
        current_gen = len(st.session_state.fitness_history)
        progress_bar.progress(min(current_gen / generations, 1.0))
        gen_label.text(f"Generación {current_gen}/{generations} — mejor fitness: {fitness:.2f}")
        chart_placeholder.line_chart(pd.DataFrame({"mejor fitness": st.session_state.fitness_history}))
        time.sleep(0.15)  # pausa artificial solo para que se vea la animación

    with st.spinner("Entrenando población..."):
        winner, stats, config = run_neat(CONFIG_PATH, generations, progress_callback=callback)

    st.session_state.winner = winner
    st.session_state.config = config
    st.success("¡Entrenamiento completo!")


def frames_to_gif_bytes(frames, duration_ms=30, resize_to=(300, 450)):
    """Convierte una lista de imágenes PIL en un GIF animado en memoria,
    para mostrarlo de una sola vez en vez de frame por frame."""
    resized = [f.resize(resize_to) for f in frames]
    buf = io.BytesIO()
    resized[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=resized[1:],
        duration=duration_ms,
        loop=0,
    )
    buf.seek(0)
    return buf


if st.session_state.winner is not None:
    st.subheader("🏆 Mejor genoma entrenado")
    if st.button("▶️ Ver al mejor pájaro jugar"):
        with st.spinner("Generando animación..."):
            frames = render_genome(st.session_state.winner, st.session_state.config)
            gif_bytes = frames_to_gif_bytes(frames, duration_ms=30)
        st.image(gif_bytes)
        st.info(f"Sobrevivió {len(frames)} frames.")
