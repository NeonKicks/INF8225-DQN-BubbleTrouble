from threading import Timer
import json

from bubbles import *
from player import *


class Game:
    def __init__(self, level=1):
        self.balls = []
        self.hexagons = []
        self.player = Player()
        self.level = level
        self.game_over = False
        self.level_completed = False
        self.time_left = 0
        self.timers = None
        self.reward = 0

    def load_level(self, level=1):
        self.__init__(level)
        self.player.set_position(WINDOWWIDTH / 2)
        self.player.is_alive = True

        with open(APP_PATH + 'levels.json', 'r') as levels_file:
            levels = json.load(levels_file)
            level = levels[str(self.level)]
            self.time_left = level['time']
            for ball in level['balls']:
                x, y = ball['x'], ball['y']
                size = ball['size']
                speed = ball['speed']
                self.balls.append(Ball(x, y, size, speed))
            for hexagon in level['hexagons']:
                x, y = hexagon['x'], hexagon['y']
                size = hexagon['size']
                speed = hexagon['speed']
                self.hexagons.append(Hexagon(x, y, size, speed))
        self._timer()

    def _check_for_collisions(self):
        self._check_for_bubble_collision(self.balls, True)
        self._check_for_bubble_collision(self.hexagons, False)

    def _check_for_bubble_collision(self, bubbles, is_ball):
        for bubble_index, bubble in enumerate(bubbles):
            if pygame.sprite.collide_rect(bubble, self.player.weapon) \
                    and self.player.weapon.is_active:
                self.player.weapon.is_active = False
                self.reward += REWARD_DESTROY
                if is_ball:
                    self._split_ball(bubble_index)
                else:
                    self._split_hexagon(bubble_index)
                return True
            if pygame.sprite.collide_mask(bubble, self.player):
                self.player.is_alive = False
                self._decrease_lives()
                return True
        return False

    def _decrease_lives(self):
        self.player.lives -= 1
        if self.player.lives:
            self.player.is_alive = False
        else:
            self.game_over = True
            self.reward += REWARD_DEATH

    def restart(self):
        self.load_level(self.level)

    def _split_ball(self, ball_index):
        ball = self.balls[ball_index]
        if ball.size > 1:
            self.balls.append(Ball(
                ball.rect.left - ball.size**2,
                ball.rect.top - 10, ball.size - 1, [-3, -5])
            )
            self.balls.append(
                Ball(ball.rect.left + ball.size**2,
                     ball.rect.top - 10, ball.size - 1, [3, -5])
            )
        del self.balls[ball_index]

    def _split_hexagon(self, hex_index):
        hexagon = self.hexagons[hex_index]
        if hexagon.size > 1:
            self.hexagons.append(
                Hexagon(hexagon.rect.left, hexagon.rect.centery,
                        hexagon.size - 1, [-3, -5]))
            self.hexagons.append(
                Hexagon(hexagon.rect.right, hexagon.rect.centery,
                        hexagon.size - 1, [3, -5]))
        del self.hexagons[hex_index]

    def update(self, restart=True):
        if restart:
            if self.level_completed or self.game_over:
                self.restart()
                return
        self.reward += REWARD_STEP
        self._check_for_collisions()
        for ball in self.balls:
            ball.update()
        for hexagon in self.hexagons:
            hexagon.update()
        self.player.update()
        if not self.balls and not self.hexagons:
            print(len(self.balls))
            self.level_completed = True
            self.reward += REWARD_WIN

    def _timer(self):
        self._stop_timers()
        self.timers = [Timer(t, self._tick_second, []) for t in range(1, self.time_left)]
        for timer in self.timers:
            timer.start()

    def _tick_second(self):
        self.time_left -= 1
        if self.time_left == 0:
            self._decrease_lives()

    def _stop_timers(self):
        if self.timers is not None:
            for timer in self.timers:
                if timer.is_alive():
                    timer.cancel()

    def exit_game(self):
        self._stop_timers()

    def state(self):
        return None
