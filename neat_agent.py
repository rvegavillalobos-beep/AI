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


def get_config_path():
    tmp_dir = tempfile.gettempdir()
    path = os.path.join(tmp_dir, "config-feedforward-generated.txt")
    f = open(path, "w", encoding="utf-8")
    f.write(CONFIG_TEXT)
    f.close()
    return path


def eval_genomes(genomes, config):
    birds, nets, ge = [], [], []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird())
        genome.fitness = 0
        ge.append(genome)

    game = Game()
    max_frames = 1000

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

    for i, bird in enumerate(birds):
        ge[i].fitness = bird.frames_survived * 0.1 + bird.score * 5


def run_neat(config_path, generations=20, progress_callback=None):
    real_config_path = get_config_path()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        real_config_path,
    )

    population = neat.Population(config)
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    if progress_callback:
        class StreamlitReporter(neat.reporting.BaseReporter):
            def post_evaluate(self, config, population, species, best_genome):
                progress_callback(best_genome.fitness)

        population.add_reporter(StreamlitReporter())

    winner = population.run(eval_genomes, generations)
    return winner, stats, config


def render_genome(genome, config, max_frames=600):
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
        frames.append(_draw_frame(bird, game))

    return frames


def _draw_frame(bird, game):
    img = Image.new("RGB", (WIDTH, HEIGHT), (135, 206, 235))
    draw = ImageDraw.Draw(img)

    for pipe in game.pipes:
        draw.rectangle([pipe.x, 0, pipe.x + 60, pipe.gap_y], fill=(34, 139, 34))
        draw.rectangle([pipe.x, pipe.gap_y + PIPE_GAP, pipe.x + 60, HEIGHT], fill=(34, 139, 34))

    draw.ellipse(
        [BIRD_X - 12, bird.y - 12, BIRD_X + 12, bird.y + 12],
        fill=(255, 215, 0),
    )
    return img
