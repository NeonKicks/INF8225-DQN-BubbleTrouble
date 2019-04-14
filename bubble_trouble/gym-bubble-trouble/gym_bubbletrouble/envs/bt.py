from pygame.locals import *
from game import *


pygame.init()
surface = pygame.Surface((WINDOWWIDTH, WINDOWHEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont('monospace', 30)
game = Game()
exit_game = False
screen = None


def setup():
    pygame.display.set_caption('Bubble Trouble')
    pygame.mouse.set_visible(True)
    global screen
    screen = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))


def get_numpy_array():
    return pygame.surfarray.array3d(surface)


def start_level(level):
    print('Starting game')
    game.load_level(level)
    pygame.mouse.set_visible(False)

    # Global game loop
    global exit_game
    while True:
        game.update()
        draw_world()
        handle_game_event()
        if exit_game:
            break
        render()
        clock.tick(FPS)
    print('Exiting game')


def quit_game():
    global exit_game
    exit_game = True
    pygame.display.quit()
    pygame.quit()
    game.exit_game()


def draw_ball(ball):
    surface.blit(ball.image, ball.rect)


def draw_hex(hexagon):
    surface.blit(hexagon.image, hexagon.rect)


def draw_player(player):
    surface.blit(player.image, player.rect)


def draw_weapon(weapon):
    surface.blit(weapon.image, weapon.rect)


def draw_timer():
    timer = font.render(str(game.time_left), 1, RED)
    rect = timer.get_rect()
    rect.bottomleft = 10, WINDOWHEIGHT - 10
    surface.blit(timer, rect)


def draw_player_lives(player):
    player_image = pygame.transform.scale(player.image, (20, 20))
    rect = player_image.get_rect()
    for life_num in range(player.lives):
        surface.blit(
            player_image,
            (WINDOWWIDTH - (life_num + 1) * 20 - rect.width, 10)
        )


def draw_world():
    surface.fill(WHITE)
    for hexagon in game.hexagons:
        draw_hex(hexagon)
    for ball in game.balls:
        draw_ball(ball)
    if game.player.weapon.is_active:
        draw_weapon(game.player.weapon)
    draw_player(game.player)
    draw_player_lives(game.player)
    draw_timer()


def render():
    screen.blit(surface, (0, 0))
    pygame.display.update()

def handle_game_event():
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_LEFT:
                game.player.moving_left = True
            elif event.key == K_RIGHT:
                game.player.moving_right = True
            elif event.key == K_SPACE and not game.player.weapon.is_active:
                game.player.shoot()
            elif event.key == K_ESCAPE:
                quit_game()
        if event.type == KEYUP:
            if event.key == K_LEFT:
                game.player.moving_left = False
            elif event.key == K_RIGHT:
                game.player.moving_right = False
        if event.type == QUIT:
            quit_game()


if __name__ == '__main__':
    setup()
    start_level(1)
