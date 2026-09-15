import neat
from PIL import Image, ImageDraw
from flappy_env import Game, Bird, BIRD_X, WIDTH, HEIGHT, PIPE_GAP


def eval_genomes(genomes, config):
    birds, nets, ge = [], [], []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird())
        genome.fitness = 0
        ge.append(genome)

    game = Game()
    max_frames = 1000  # límite de seguridad

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
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
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
