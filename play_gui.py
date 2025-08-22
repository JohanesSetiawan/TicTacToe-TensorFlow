import os
import sys
import random
import pygame
import numpy as np
import tensorflow as tf

from utils.mcts import MCTS
from model.models import ResNet
from utils.game import TicTacToe
from config import *

# --- Pygame Configuration ---
pygame.init()
WIDTH, HEIGHT = 300, 400
LINE_WIDTH = 5
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = 300 // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 10
CROSS_WIDTH = 15
SPACE = SQUARE_SIZE // 4

# Colors
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (84, 84, 84)
TEXT_COLOR = (255, 255, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe vs AI')
screen.fill(BG_COLOR)

# Fonts
FONT = pygame.font.Font(None, 40)
SMALL_FONT = pygame.font.Font(None, 20)

# --- Device Setup ---
DEVICE = "/CPU:0"
print(f"Inference will use: {DEVICE}")

# --- MODEL_PATH ---
LATEST_MODEL_PATH = LATEST_MODEL_PATH


def draw_lines():  # --- Pygame Helper Functions ---
    pygame.draw.rect(screen, BG_COLOR, (0, 0, 300, 300)
                     )  # Clear board only
    # Horizontal lines
    pygame.draw.line(screen, LINE_COLOR, (0, SQUARE_SIZE),
                     (300, SQUARE_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (0, 2 * SQUARE_SIZE),
                     (300, 2 * SQUARE_SIZE), LINE_WIDTH)
    # Vertical lines
    pygame.draw.line(screen, LINE_COLOR, (SQUARE_SIZE, 0),
                     (SQUARE_SIZE, 300), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (2 * SQUARE_SIZE, 0),
                     (2 * SQUARE_SIZE, 300), LINE_WIDTH)


def draw_figures(board):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 1:  # 'X'
                pygame.draw.line(screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE),
                                 (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), CROSS_WIDTH)
                pygame.draw.line(screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE -
                                 SPACE), (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE), CROSS_WIDTH)
            elif board[row][col] == -1:  # 'O'
                pygame.draw.circle(screen, CIRCLE_COLOR, (int(col * SQUARE_SIZE + SQUARE_SIZE // 2), int(
                    row * SQUARE_SIZE + SQUARE_SIZE // 2)), CIRCLE_RADIUS, CIRCLE_WIDTH)


def draw_status(message):
    pygame.draw.rect(screen, BG_COLOR, (0, 300, 300, 100)
                     )  # Clear status area
    text = SMALL_FONT.render(message, True, TEXT_COLOR)
    text_rect = text.get_rect(center=(WIDTH / 2, 350))
    screen.blit(text, text_rect)
    pygame.display.update()


def selection_screen():
    """Display role selection screen with timer."""
    x_rect = pygame.Rect(50, 150, 100, 50)
    o_rect = pygame.Rect(150, 150, 100, 50)
    start_ticks = pygame.time.get_ticks()

    while True:
        screen.fill(BG_COLOR)

        # Display text
        title_text = FONT.render("Choose Your Symbol", True, TEXT_COLOR)
        screen.blit(title_text, title_text.get_rect(center=(WIDTH / 2, 50)))

        # X and O buttons
        pygame.draw.rect(screen, LINE_COLOR, x_rect)
        pygame.draw.rect(screen, LINE_COLOR, o_rect)
        x_text = FONT.render("X", True, CROSS_COLOR)
        o_text = FONT.render("O", True, CIRCLE_COLOR)
        screen.blit(x_text, x_text.get_rect(center=x_rect.center))
        screen.blit(o_text, o_text.get_rect(center=o_rect.center))

        # Timer
        seconds_left = 10 - (pygame.time.get_ticks() - start_ticks) // 1000
        if seconds_left < 0:
            seconds_left = 0
        timer_text = FONT.render(f"Time: {seconds_left}", True, TEXT_COLOR)
        screen.blit(timer_text, timer_text.get_rect(center=(WIDTH / 2, 250)))

        pygame.display.update()

        # Handle Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if x_rect.collidepoint(event.pos):
                    # human_player=1 (X), starting_player=1 (Human)
                    return 1, 1
                if o_rect.collidepoint(event.pos):
                    # human_player=-1 (O), starting_player=-1 (Human)
                    return -1, -1

        # Timeout logic
        if seconds_left <= 0:
            print("Time's up! Role randomized. AI goes first.")
            ai_player = random.choice([1, -1])
            human_player = -ai_player
            return human_player, ai_player  # AI starts


def main():
    # Initialize model and MCTS once
    game = TicTacToe()
    model = ResNet(game, num_resBlocks=RESBLOCKS, num_hidden=HIDDEN_UNITS)

    print("Building model with dummy input...")
    dummy_input = np.expand_dims(
        game.get_encoded_state(game.get_initial_state()), axis=0)
    _ = model(tf.convert_to_tensor(dummy_input, dtype=tf.float32))
    print("✅ Model built successfully.")

    latest_model_path = LATEST_MODEL_PATH
    if not os.path.exists(latest_model_path):
        print(f"ERROR: Model file not found at {latest_model_path}")
        return
    model.load_weights(latest_model_path)
    mcts = MCTS(model, game, args)

    while True:  # Loop to restart game
        human_player, player = selection_screen()
        state = game.get_initial_state()
        game_over = False

        draw_lines()
        pygame.display.update()

        # Main Game Loop
        while not game_over:
            human_symbol = 'X' if human_player == 1 else 'O'

            if player == human_player:
                draw_status(f"Your turn ({human_symbol}). Click a square.")
            else:
                draw_status(
                    f"AI's turn ({'O' if human_symbol == 'X' else 'X'}).")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and player == human_player:
                    mouseX, mouseY = event.pos
                    if mouseY < 300:  # Ensure click is on board area
                        clicked_row = mouseY // SQUARE_SIZE
                        clicked_col = mouseX // SQUARE_SIZE

                        if state[clicked_row, clicked_col] == 0:
                            move = clicked_row * 3 + clicked_col
                            state = game.get_next_state(state, move, player)
                            draw_figures(state)
                            _, is_terminal = game.get_value_and_terminated(
                                state, move)
                            player = game.get_opponent(player)

                            if is_terminal:
                                game_over = True

            if player != human_player and not game_over:
                draw_status(
                    f"AI's turn ({'O' if human_symbol == 'X' else 'X'}). AI is thinking...")

                neutral_state = game.change_perspective(state, player)
                with tf.device(DEVICE):
                    action_probs = mcts.search(neutral_state)

                action = np.argmax(action_probs)

                chosen_move_prob = action_probs[action]
                ai_symbol = 'O' if human_player == 1 else 'X'

                # --- AI THINKING EFFECT ---
                print(
                    f"AI confidence for placing '{ai_symbol}' at square {action + 1}: {chosen_move_prob:.2%}")
                print(f"AI chooses square {action + 1}")
                pygame.time.wait(1000)  # 1 second delay
                # -------------------------

                state = game.get_next_state(state, action, player)
                value, is_terminal = game.get_value_and_terminated(
                    state, action)
                draw_figures(state)
                player = game.get_opponent(player)

                if is_terminal:
                    game_over = True

            pygame.display.update()

        # Game Over Screen
        _, is_terminal = game.get_value_and_terminated(
            state, None)  # Check final result
        winner_player = -player  # Last player to move is the winner

        if value == 1:
            winner_symbol = 'X' if winner_player == 1 else 'O'
            draw_status(
                f"Winner is {winner_symbol}! Press 'R' to play again.")
        else:
            draw_status("Game is a draw! Press 'R' to play again.")

        # Loop waiting for restart
        wait_for_restart = True
        while wait_for_restart:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        wait_for_restart = False


if __name__ == "__main__":
    main()
