import random

WIDTH, HEIGHT = 400, 600
GRAVITY = 0.5
JUMP_STRENGTH = -8
PIPE_WIDTH = 60
PIPE_GAP = 100
PIPE_SPEED = 4
BIRD_X = 60
BIRD_RADIUS = 12


class Bird:
    def __init__(self):
        self.y = HEIGHT // 2
        self.vel = 0
        self.alive = True
        self.score = 0
        self.frames_survived = 0

    def jump(self):
        self.vel = JUMP_STRENGTH

    def update(self):
        self.vel += GRAVITY
        self.y += self.vel
        self.frames_survived += 1
        if self.y < 0 or self.y > HEIGHT:
            self.alive = False


class Pipe:
    def __init__(self, x):
        self.x = x
        self.gap_y = random.randint(80, HEIGHT - 80 - PIPE_GAP)
        self.passed_by = set()

    def update(self):
        self.x -= PIPE_SPEED

    def collide(self, bird):
        if BIRD_X + BIRD_RADIUS > self.x and BIRD_X - BIRD_RADIUS < self.x + PIPE_WIDTH:
            if bird.y - BIRD_RADIUS < self.gap_y or bird.y + BIRD_RADIUS > self.gap_y + PIPE_GAP:
                return True
        return False


class Game:
    def __init__(self):
        self.pipes = [Pipe(WIDTH + 100)]
        self.frame = 0

    def get_next_pipe(self):
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > BIRD_X:
                return pipe
        return self.pipes[-1]

    def update(self, birds):
        self.frame += 1
        next_pipe = self.get_next_pipe()

        for bird in birds:
            if not bird.alive:
                continue
            bird.update()
            if next_pipe.collide(bird):
                bird.alive = False
            if next_pipe.x + PIPE_WIDTH < BIRD_X and id(bird) not in next_pipe.passed_by:
                next_pipe.passed_by.add(id(bird))
                bird.score += 1

        for pipe in self.pipes:
            pipe.update()

        if self.pipes[0].x + PIPE_WIDTH < 0:
            self.pipes.pop(0)

        if self.pipes[-1].x < WIDTH - 200:
            self.pipes.append(Pipe(WIDTH + 50))

    def any_alive(self, birds):
        return any(b.alive for b in birds)
