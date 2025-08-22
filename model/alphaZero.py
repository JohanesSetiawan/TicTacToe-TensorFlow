import tensorflow as tf
from utils.mcts import MCTS

import random
import numpy as np
from tqdm import trange
from config import SAVE_DIR


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

            # Flip the already rotated state and policy
            flipped_state = np.flip(rotated_state, axis=2)
            flipped_policy = np.flip(rotated_policy, axis=1)

            augmented_states.append(rotated_state)
            augmented_policies.append(rotated_policy.flatten())
            augmented_states.append(flipped_state)
            augmented_policies.append(flipped_policy.flatten())

    return np.array(augmented_states), np.array(augmented_policies)


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(model, game, args)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (
                1 / self.args['temperature'])

            sum_probs = np.sum(temperature_action_probs)
            if sum_probs > 0:
                # If there are probabilities, normalize and choose based on them
                temperature_action_probs /= sum_probs
                action = np.random.choice(
                    self.game.action_size, p=temperature_action_probs)
            else:
                # Fallback: If all probabilities are zero, choose a random valid move
                print(
                    "WARNING: MCTS returned all-zero probabilities. Choosing a random valid move.")
                valid_moves = self.game.get_valid_moves(neutral_state)
                action = np.random.choice(np.where(valid_moves == 1)[0])

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(
                state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(
                        value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx + self.args['batch_size']]
            states, policy_targets, value_targets = zip(*sample)

            # Convert data from NumPy list to TensorFlow tensor format
            states = np.array(states)
            policy_targets = np.array(policy_targets)
            value_targets = np.array(value_targets).reshape(-1, 1)

            if self.args['augment']:
                states, policy_targets = augment_data(
                    states, policy_targets, self.game)
                # Since data becomes 8x more, we need to duplicate value targets as well
                value_targets = np.repeat(value_targets, 8, axis=0)

            # In TensorFlow, we use `tf.GradientTape` to "record" operations
            # on forward pass so we can compute gradients automatically.
            with tf.GradientTape() as tape:
                # 1. Forward Pass
                # Call model with `training=True` to activate layers like BatchNormalization
                out_policy, out_value = self.model(states, training=True)

                # 2. Calculate Loss
                # Policy loss: CrossEntropy. `from_logits=True` because our model output hasn't passed through softmax.
                policy_loss = tf.keras.losses.CategoricalCrossentropy(
                    from_logits=True)(policy_targets, out_policy)

                # Value loss: Mean Squared Error
                value_loss = tf.keras.losses.MeanSquaredError()(value_targets, out_value)

                # Total loss
                loss = policy_loss + value_loss

            # 3. Calculate Gradients
            # Calculate gradients of loss with respect to all trainable variables in the model
            grads = tape.gradient(loss, self.model.trainable_variables)

            # 4. Apply Gradients (Update Weights)
            # Apply the calculated gradients to the model using optimizer
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            # How to save model weights in Keras
            self.model.save_weights(
                f"{SAVE_DIR}model_{iteration}.weights.h5")
