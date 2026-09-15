import os
import tempfile
import neat
from PIL import Image, ImageDraw
from flappy_env import Game, Bird, BIRD_X, WIDTH, HEIGHT, PIPE_GAP

CONFIG_TEXT = """[NEAT]
fitness_criterion       = max
fitness_threshold       = 1000
pop_size                = 50
reset_on_extinction     = True
no_fitness_termination = False

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.5
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full
node_add_prob           = 0.2
node_delete_prob        = 0.2
num_hidden              = 0
num_inputs              = 5
num_outputs             = 1
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation        = 20
species_elitism       = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""

BIRD_COLORS = [
    (255, 215, 0), (255, 99, 71), (60, 179, 113), (30, 144, 255),
    (238, 130, 238), (255, 165, 0), (0, 206, 209), (218, 112, 214),
]


def get_config_path():
    tmp_dir = tempfile.gettempdir()
    path = os.path.join(tmp_dir, "config-feedforward-generated.txt")
    f = open(path, "w", encoding="utf-8")
    f.write(CONFIG_TEXT)
    f.close()
    return path


def _draw_frame_single(bird, game):
    img = Image.new("RGB", (WIDTH, HEIGHT), (135, 206, 235))
    draw = ImageDraw.Draw(img)
    for pipe in game.pipes:
        draw.rectangle([pipe.x, 0, pipe.x + 60, pipe.gap_y], fill=(34, 139, 34))
        draw.rectangle([pipe.x, pipe.gap_y + PIPE_GAP, pipe.x + 60, HEIGHT], fill=(34, 139, 34))
    draw.ellipse([BIRD_X - 12, bird.y - 12, BIRD_X + 12, bird.y + 12], fill=(255, 215, 0))
    return img


def _draw_frame_multi(birds, game):
    img = Image.new("RGB", (WIDTH, HEIGHT), (135, 206, 235))
    draw = ImageDraw.Draw(img)
    for pipe in game.pipes:
        draw.rectangle([pipe.x, 0, pipe.x + 60, pipe.gap_y], fill=(34, 139, 34))
        draw.rectangle([pipe.x, pipe.gap_y + PIPE_GAP, pipe.x + 60, HEIGHT], fill=(34, 139, 34))
    alive_birds = [b for b in birds if b.alive]
    for i, bird in enumerate(alive_birds):
        color = BIRD_COLORS[i % len(BIRD_COLORS)]
        draw.ellipse([BIRD_X - 8, bird.y - 8, BIRD_X + 8, bird.y + 8], fill=color)
    return img


def draw_network(genome, config, size=(480, 380)):
    """Dibuja la topología de la red: entradas a la izquierda, ocultos en
    el medio, salida a la derecha. Verde = peso positivo, rojo = negativo."""
    input_keys = list(config.genome_config.input_keys)
    output_keys = list(config.genome_config.output_keys)
    node_ids = set(genome.nodes.keys())
    hidden_keys = [n for n in node_ids if n not in output_keys]

    W, H = size
    margin = 50
    positions = {}

    def place_column(keys, x):
        n = len(keys)
        if n == 0:
            return
        for idx, k in enumerate(keys):
            y = margin + idx * (H - 2 * margin) / max(1, n - 1) if n > 1 else H / 2
            positions[k] = (x, y)

    place_column(input_keys, margin)
    place_column(hidden_keys, W / 2)
    place_column(output_keys, W - margin)

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for (in_key, out_key), conn in genome.connections.items():
        if not conn.enabled:
            continue
        if in_key not in positions or out_key not in positions:
            continue
        x1, y1 = positions[in_key]
        x2, y2 = positions[out_key]
        weight = conn.weight
        color = (34, 139, 34) if weight > 0 else (220, 20, 60)
        width = max(1, min(6, int(abs(weight))))
        draw.line([x1, y1, x2, y2], fill=color, width=width)

    input_names = ["bird_y", "bird_vel", "pipe_dx", "gap_top", "gap_bottom"]
    for idx, k in enumerate(input_keys):
        x, y = positions[k]
        draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill=(70, 130, 180))
        name = input_names[idx] if idx < len(input_names) else str(k)
        draw.text((x - 35, y + 12), name, fill=(0, 0, 0))

    for k in hidden_keys:
        x, y = positions[k]
        draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill=(255, 165, 0))

    for k in output_keys:
        x, y = positions[k]
        draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill=(220, 20, 60))
        draw.text((x - 12, y + 12), "jump", fill=(0, 0, 0))

    return img


def make_eval_function(snapshot_generations, snapshot_store,
                        max_frames=3000, frame_skip=4, max_capture_frames=300):
    """Crea la función de evaluación que NEAT llama cada generación.
    Si la generación actual está en snapshot_generations, además de
    evaluar el fitness, va grabando frames de TODA la población para
    poder mostrarlos después como GIF."""
    counter = {"gen": 0}

    def eval_genomes(genomes, config):
        counter["gen"] += 1
        current_gen = counter["gen"]
        capture = current_gen in snapshot_generations

        birds, nets, ge = [], [], []
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            birds.append(Bird())
            genome.fitness = 0
            ge.append(genome)

        game = Game()
        frames = [] if capture else None

        while game.any_alive(birds) and game.frame < max_frames:
            next_pipe = game.get_next_pipe()

            for i, bird in enumerate(birds):
                if not bird.alive:
                    continue
                output = nets[i].activate((
                    bird.y,
                    bird.vel,
                    next_pipe.x - BIRD_X,
                    next_pipe.gap_y,
                    next_pipe.gap_y + PIPE_GAP - bird.y,
                ))
                if output[0] > 0.5:
                    bird.jump()

            game.update(birds)

            if capture and len(frames) < max_capture_frames and game.frame % frame_skip == 0:
                frames.append(_draw_frame_multi(birds, game))

        for i, bird in enumerate(birds):
            ge[i].fitness = bird.frames_survived * 0.1 + bird.score * 5

        if capture:
            best_genome = max(ge, key=lambda g: g.fitness)
            snapshot_store[current_gen] = {
                "frames": frames,
                "best_genome": best_genome,
                "best_fitness": best_genome.fitness,
                "avg_fitness": sum(g.fitness for g in ge) / len(ge),
                "num_alive_end": sum(1 for b in birds if b.alive),
                "population_size": len(birds),
            }

    return eval_genomes


def run_neat(config_path, generations=20, progress_callback=None, snapshot_generations=None):
    real_config_path = get_config_path()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        real_config_path,
    )

    if snapshot_generations is None:
        snapshot_generations = sorted(set([
            1,
            max(1, generations // 4),
            max(1, generations // 2),
            max(1, (generations * 3) // 4),
            generations,
        ]))

    snapshot_store = {}
    eval_function = make_eval_function(set(snapshot_generations), snapshot_store)

    population = neat.Population(config)
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    if progress_callback:
        class StreamlitReporter(neat.reporting.BaseReporter):
            def post_evaluate(self, config, population, species, best_genome):
                fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
                avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0
                num_species = len(species.species)
                progress_callback(best_genome.fitness, avg_fitness, num_species)

        population.add_reporter(StreamlitReporter())

    winner = population.run(eval_function, generations)
    return winner, stats, config, snapshot_store, snapshot_generations


def render_genome(genome, config, max_frames=1500):
    """Reproduce la partida completa de un solo genoma (el ganador final)."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    bird = Bird()
    game = Game()
    frames = []

    while bird.alive and game.frame < max_frames:
        next_pipe = game.get_next_pipe()
        output = net.activate((
            bird.y,
            bird.vel,
            next_pipe.x - BIRD_X,
            next_pipe.gap_y,
            next_pipe.gap_y + PIPE_GAP - bird.y,
        ))
        if output[0] > 0.5:
            bird.jump()

        game.update([bird])
        frames.append(_draw_frame_single(bird, game))

    return frames
