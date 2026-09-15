import streamlit as st
import time
import os
import io
import pandas as pd

from neat_agent import run_neat, render_genome, draw_network

st.set_page_config(page_title="Flappy Bird NEAT", layout="centered")
st.title("🐦 Flappy Bird entrenado con NEAT")

st.markdown("""
Este agente aprende a jugar Flappy Bird usando **algoritmos genéticos (NEAT)**.
Cada generación es una población de pájaros; los que sobreviven más tiempo
y pasan más tuberías tienen mayor probabilidad de "reproducirse" en la
siguiente generación.

Para que puedas ver **qué pasa por dentro**, la app guarda automáticamente
5 "fotos" del entrenamiento (generación 1, 25%, 50%, 75% y la última),
mostrando cómo vuela toda la población y cómo cambia el "cerebro"
(red neuronal) del mejor pájaro en cada una de ellas.
""")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")

for key, default in [
    ("fitness_history", []),
    ("avg_fitness_history", []),
    ("species_history", []),
    ("winner", None),
    ("config", None),
    ("snapshot_store", None),
    ("snapshot_generations", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

generations = st.slider("Número de generaciones a entrenar", 5, 100, 20)


def frames_to_gif_bytes(frames, duration_ms=60, resize_to=(250, 375)):
    if not frames:
        return None
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


if st.button("🚀 Entrenar"):
    st.session_state.fitness_history = []
    st.session_state.avg_fitness_history = []
    st.session_state.species_history = []
    progress_bar = st.progress(0)
    gen_label = st.empty()
    chart_placeholder = st.empty()

    def callback(best_fitness, avg_fitness, num_species):
        st.session_state.fitness_history.append(best_fitness)
        st.session_state.avg_fitness_history.append(avg_fitness)
        st.session_state.species_history.append(num_species)
        current_gen = len(st.session_state.fitness_history)
        progress_bar.progress(min(current_gen / generations, 1.0))
        gen_label.text(
            f"Generación {current_gen}/{generations} — "
            f"mejor: {best_fitness:.2f} | promedio: {avg_fitness:.2f} | "
            f"especies: {num_species}"
        )
        chart_placeholder.line_chart(pd.DataFrame({
            "mejor fitness": st.session_state.fitness_history,
            "fitness promedio": st.session_state.avg_fitness_history,
        }))
        time.sleep(0.15)

    with st.spinner("Entrenando población y capturando generaciones clave..."):
        winner, stats, config, snapshot_store, snapshot_generations = run_neat(
            CONFIG_PATH, generations, progress_callback=callback
        )

    st.session_state.winner = winner
    st.session_state.config = config
    st.session_state.snapshot_store = snapshot_store
    st.session_state.snapshot_generations = snapshot_generations
    st.success("¡Entrenamiento completo!")

    st.subheader("🧬 Número de especies por generación")
    st.line_chart(pd.DataFrame({"especies": st.session_state.species_history}))


# ---------- Galería de generaciones clave ----------
if st.session_state.snapshot_store:
    st.header("📸 Galería de generaciones")
    st.markdown("Explora cómo evolucionó la población y su 'cerebro' a lo largo del entrenamiento.")

    tab_labels = [f"Gen {g}" for g in st.session_state.snapshot_generations]
    tabs = st.tabs(tab_labels)

    for tab, gen_num in zip(tabs, st.session_state.snapshot_generations):
        with tab:
            snap = st.session_state.snapshot_store.get(gen_num)
            if snap is None:
                st.warning("No se capturó esta generación.")
                continue

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🎬 Población volando**")
                gif_bytes = frames_to_gif_bytes(snap["frames"])
                if gif_bytes:
                    st.image(gif_bytes)
                st.caption(
                    f"Sobrevivieron {snap['num_alive_end']} de "
                    f"{snap['population_size']} pájaros hasta el final del clip."
                )

            with col2:
                st.markdown("**🧠 Red neuronal del mejor pájaro**")
                network_img = draw_network(snap["best_genome"], st.session_state.config)
                st.image(network_img)

            st.markdown(
                f"**Mejor fitness:** {snap['best_fitness']:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Fitness promedio:** {snap['avg_fitness']:.2f}"
            )


# ---------- Ganador final jugando partida completa ----------
if st.session_state.winner is not None:
    st.header("🏆 Ganador final — partida completa")
    if st.button("▶️ Ver al mejor pájaro jugar"):
        with st.spinner("Generando animación..."):
            frames = render_genome(st.session_state.winner, st.session_state.config)
            gif_bytes = frames_to_gif_bytes(frames, duration_ms=30, resize_to=(300, 450))
        st.image(gif_bytes)
        st.info(f"Sobrevivió {len(frames)} frames.")
