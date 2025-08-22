import tensorflow as tf

import math
import numpy as np


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            # This formula is the same as the original, calculating Q-value from current player's perspective
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(
                    child_state, player=-1)

                child = Node(self.game, self.args,
                             child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args

    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)

        # --- Conversion Block 1: First Model Call ---
        encoded_state = self.game.get_encoded_state(state)
        tensor_state = tf.convert_to_tensor(encoded_state, dtype=tf.float32)

        # Equivalent to .unsqueeze(0)
        tensor_state = tf.expand_dims(tensor_state, 0)

        # Call model with `training=False` for inference/evaluation mode
        policy, _ = self.model(tensor_state, training=False)

        policy = tf.nn.softmax(policy, axis=1)  # Equivalent to torch.softmax
        policy = tf.squeeze(policy, axis=0)    # Equivalent to .squeeze(0)
        policy = policy.numpy()                # Equivalent to .cpu().numpy()

        # Dirichlet noise and valid moves logic remains the same
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for _ in range(self.args['num_mcts_searches']):
            node = root

            while node.is_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(
                node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                # --- Conversion Block 2: Model Call Inside Loop ---
                encoded_node_state = self.game.get_encoded_state(node.state)
                tensor_node_state = tf.convert_to_tensor(
                    encoded_node_state, dtype=tf.float32)
                tensor_node_state = tf.expand_dims(tensor_node_state, 0)

                policy, value = self.model(tensor_node_state, training=False)

                policy = tf.nn.softmax(policy, axis=1).numpy().squeeze(0)
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                # Equivalent to value.item() for tensor (1,1)
                value = value.numpy()[0][0]

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
