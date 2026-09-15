import streamlit as st
import time
import os
import io
import pandas as pd

from neat_agent import (
    run_neat, render_genome, draw_network,
    create_population, make_eval_function,
)

st.set_page_config(page_title="Flappy Bird NEAT", layout="centered")
st.title("🐦 Flappy Bird entrenado con NEAT")

st.markdown("""
Este agente aprende a jugar Flappy Bird usando **algoritmos genéticos (NEAT)**.
Elige uno de los dos modos de entrenamiento abajo.
""")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")

defaults = {
    "fitness_history": [],
    "avg_fitness_history": [],
    "species_history": [],
    "winner": None,
    "config": None,
    "snapshot_store": None,
    "snapshot_generations": None,
    "cont_population": None,
    "cont_config": None,
    "cont_gen_counter": {"gen": 0},
    "cont_snapshot_store": {},
    "cont_fitness_history": [],
    "cont_avg_fitness_history": [],
    "cont_species_history": [],
    "cont_active": False,
}
for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


def frames_to_gif_bytes(frames, duration_ms=60, resize_to=(250, 375)):
    if not frames:
        return None
    resized = [f.resize(resize_to) for f in frames]
    buf = io.BytesIO()
    resized[0].save(
        buf, format="GIF", save_all=True,
        append_images=resized[1:], duration=duration_ms, loop=0,
    )
    buf.seek(0)
    return buf


def render_gallery(snapshot_store, config, key_prefix=""):
    sorted_gens = sorted(snapshot_store.keys())
    tab_labels = [f"Gen {g}" for g in sorted_gens]
    tabs = st.tabs(tab_labels)
    for tab, gen_num in zip(tabs, sorted_gens):
        with tab:
            snap = snapshot_store[gen_num]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🎬 Población volando**")
                gif_bytes = frames_to_gif_bytes(snap["frames"])
                if gif_bytes:
                    st.image(gif_bytes)
                st.caption(
                    f"Sobrevivieron {snap['num_alive_end']} de "
                    f"{snap['population_size']} pájaros."
                )
            with col2:
                st.markdown("**🧠 Red neuronal del mejor pájaro**")
                network_img = draw_network(snap["best_genome"], config)
                st.image(network_img)
            st.markdown(
                f"**Mejor fitness:** {snap['best_fitness']:.2f} &nbsp;|&nbsp; "
                f"**Promedio:** {snap['avg_fitness']:.2f}"
            )


# ============================================================
# MODO 1: Entrenar un número fijo de generaciones
# ============================================================
st.header("🎯 Modo 1: Entrenar un número fijo de generaciones")

generations = st.slider("Número de generaciones a entrenar", 5, 300, 20)

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
            f"mejor: {best_fitness:.2f} | promedio: {avg_fitness:.2f} | especies: {num_species}"
        )
        chart_placeholder.line_chart(pd.DataFrame({
            "mejor fitness": st.session_state.fitness_history,
            "fitness promedio": st.session_state.avg_fitness_history,
        }))
        time.sleep(0.1)

    with st.spinner("Entrenando población y capturando generaciones clave..."):
        winner, stats, config, snapshot_store, snapshot_generations = run_neat(
            CONFIG_PATH, generations, progress_callback=callback
        )

    st.session_state.winner = winner
    st.session_state.config = config
    st.session_state.snapshot_store = snapshot_store
    st.session_state.snapshot_generations = snapshot_generations
    st.success("¡Entrenamiento completo!")

if st.session_state.species_history:
    st.subheader("🧬 Número de especies por generación (Modo 1)")
    st.line_chart(pd.DataFrame({"especies": st.session_state.species_history}))

if st.session_state.snapshot_store:
    st.subheader("📸 Galería de generaciones (Modo 1)")
    render_gallery(st.session_state.snapshot_store, st.session_state.config)

if st.session_state.winner is not None:
    st.subheader("🏆 Ganador final (Modo 1) — partida completa")
    if st.button("▶️ Ver al mejor pájaro jugar (Modo 1)"):
        with st.spinner("Generando animación..."):
            frames = render_genome(st.session_state.winner, st.session_state.config)
            gif_bytes = frames_to_gif_bytes(frames, duration_ms=30, resize_to=(300, 450))
        st.image(gif_bytes)
        st.info(f"Sobrevivió {len(frames)} frames.")


st.divider()

# ============================================================
# MODO 2: Entrenamiento continuo (Play / Stop)
# ============================================================
st.header("♾️ Modo 2: Entrenamiento continuo (tú decides cuándo parar)")
st.markdown("""
Corre generación por generación de forma indefinida. Presiona **▶️ Iniciar**
para comenzar (o continuar) y **⏹ Detener** para pausar y revisar resultados
sin que la pantalla siga cambiando. Mientras está activo, la página se
actualiza sola constantemente — es normal.
""")

snapshot_interval = st.number_input(
    "Capturar galería cada cuántas generaciones", min_value=1, max_value=50, value=5
)

col_start, col_stop, col_reset = st.columns(3)
with col_start:
    start_clicked = st.button("▶️ Iniciar / Continuar")
with col_stop:
    stop_clicked = st.button("⏹ Detener")
with col_reset:
    reset_clicked = st.button("🔄 Reiniciar todo")

if reset_clicked:
    st.session_state.cont_population = None
    st.session_state.cont_config = None
    st.session_state.cont_gen_counter = {"gen": 0}
    st.session_state.cont_snapshot_store = {}
    st.session_state.cont_fitness_history = []
    st.session_state.cont_avg_fitness_history = []
    st.session_state.cont_species_history = []
    st.session_state.cont_active = False
    st.rerun()

if start_clicked:
    st.session_state.cont_active = True
    if st.session_state.cont_population is None:
        fitness_lists = {
            "best": st.session_state.cont_fitness_history,
            "avg": st.session_state.cont_avg_fitness_history,
            "species": st.session_state.cont_species_history,
        }
        population, config, stats = create_population(CONFIG_PATH, fitness_lists=fitness_lists)
        st.session_state.cont_population = population
        st.session_state.cont_config = config

if stop_clicked:
    st.session_state.cont_active = False

status_placeholder = st.empty()
cont_chart_placeholder = st.empty()

if st.session_state.cont_fitness_history:
    status_placeholder.text(
        f"Generación actual: {st.session_state.cont_gen_counter['gen']} — "
        f"mejor: {st.session_state.cont_fitness_history[-1]:.2f} | "
        f"promedio: {st.session_state.cont_avg_fitness_history[-1]:.2f} | "
        f"especies: {st.session_state.cont_species_history[-1]}"
    )
    cont_chart_placeholder.line_chart(pd.DataFrame({
        "mejor fitness": st.session_state.cont_fitness_history,
        "fitness promedio": st.session_state.cont_avg_fitness_history,
    }))

if st.session_state.cont_active and st.session_state.cont_population is not None:
    population = st.session_state.cont_population

    eval_function = make_eval_function(
        should_capture_fn=lambda g: (g == 1) or (g % int(snapshot_interval) == 0),
        snapshot_store=st.session_state.cont_snapshot_store,
        gen_counter=st.session_state.cont_gen_counter,
        max_snapshots=10,
    )

    with st.spinner(f"Entrenando generación {st.session_state.cont_gen_counter['gen'] + 1}..."):
        population.run(eval_function, 1)

    time.sleep(0.2)
    st.rerun()

if not st.session_state.cont_active and st.session_state.cont_population is not None:
    best = st.session_state.cont_population.best_genome
    if best is not None:
        st.subheader("🏆 Mejor pájaro encontrado hasta ahora (Modo 2)")
        if st.button("▶️ Ver al mejor pájaro jugar (Modo 2)"):
            with st.spinner("Generando animación..."):
                frames = render_genome(best, st.session_state.cont_config)
                gif_bytes = frames_to_gif_bytes(frames, duration_ms=30, resize_to=(300, 450))
            st.image(gif_bytes)
            st.info(f"Sobrevivió {len(frames)} frames.")

if st.session_state.cont_snapshot_store:
    st.subheader("📸 Galería de generaciones (Modo 2 — últimas 10 capturadas)")
    render_gallery(st.session_state.cont_snapshot_store, st.session_state.cont_config)
