import tensorflow as tf
from utils.mctsParallel import MCTSParallel

import random
import numpy as np
from config import SAVE_DIR
from tqdm import trange


def augment_data(states, policy_targets, game):
    augmented_states = []
    augmented_policies = []
    for state, policy in zip(states, policy_targets):
        policy_2d = policy.reshape(game.row_count, game.column_count)
        for i in range(4):  # 4 rotations
            # Rotate state and policy
            # State (3, 3, 3) -> (channel, row, col)
            rotated_state = np.rot90(state, i, axes=(1, 2))
            rotated_policy = np.rot90(policy_2d, i)

            # Flip rotated state and policy
            flipped_state = np.flip(rotated_state, axis=2)
            flipped_policy = np.flip(rotated_policy, axis=1)

            augmented_states.append(rotated_state)
            augmented_policies.append(rotated_policy.flatten())
            augmented_states.append(flipped_state)
            augmented_policies.append(flipped_policy.flatten())

    return np.array(augmented_states), np.array(augmented_policies)


class SelfPlayGame:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(model, game, args)

    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SelfPlayGame(self.game)
                   for _ in range(self.args['num_parallel_games'])]

        while len(spGames) > 0:
            states = np.stack([g.state for g in spGames])
            neutral_states = self.game.change_perspective(states, player)

            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames))[::-1]:
                g = spGames[i]

                action_probs = np.zeros(self.game.action_size)
                for child in g.root.children:
                    action_probs[child.action_taken] = child.visit_count

                sum_action_probs = np.sum(action_probs)
                if sum_action_probs > 0:
                    action_probs /= sum_action_probs

                g.memory.append((g.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (
                    1 / self.args['temperature'])

                sum_temp_probs = np.sum(temperature_action_probs)
                if sum_temp_probs > 0:
                    # If there are probabilities, normalize and choose based on them
                    temperature_action_probs /= sum_temp_probs
                    action = np.random.choice(
                        self.game.action_size, p=temperature_action_probs)
                else:
                    # Fallback: If all probabilities are zero, choose a valid move randomly
                    print(
                        f"WARNING: Game {i} MCTS returned all-zero probabilities. Choosing a random valid move.")
                    valid_moves = self.game.get_valid_moves(g.state)
                    action = np.random.choice(np.where(valid_moves == 1)[0])

                g.state = self.game.get_next_state(g.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(
                    g.state, action)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in g.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(
                            value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]

            player = self.game.get_opponent(player)

        return return_memory

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx + self.args['batch_size']]
            states, policy_targets, value_targets = zip(*sample)

            states = np.array(states)
            policy_targets = np.array(policy_targets)
            value_targets = np.array(value_targets).reshape(-1, 1)

            if self.args['augment']:
                states, policy_targets = augment_data(
                    states, policy_targets, self.game)
                # Since data becomes 8x more, we need to duplicate value targets too
                value_targets = np.repeat(value_targets, 8, axis=0)

            with tf.GradientTape() as tape:
                out_policy, out_value = self.model(states, training=True)

                policy_loss = tf.keras.losses.CategoricalCrossentropy(
                    from_logits=True)(policy_targets, out_policy)
                value_loss = tf.keras.losses.MeanSquaredError()(value_targets, out_value)
                loss = policy_loss + value_loss

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()

            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            # How to save model weights in Keras
            self.model.save_weights(
                f"{SAVE_DIR}model_{iteration}.weights.h5")
