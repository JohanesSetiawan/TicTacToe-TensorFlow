import tensorflow as tf
from utils.mcts import Node

import numpy as np


class MCTSParallel:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args

    def search(self, states, spGames):
        # --- Conversion Block 1: Model Call for Root Expansion ---
        tensor_states = tf.convert_to_tensor(
            self.game.get_encoded_state(states), dtype=tf.float32)

        # Call model with batch `states`. `training=False` for inference mode.
        policy, _ = self.model(tensor_states, training=False)

        # Direct .numpy() since it's already batched
        policy = tf.nn.softmax(policy, axis=1).numpy()

        # Dirichlet noise and valid moves logic remains the same, applied to batch
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])

        valid_moves = self.game.get_valid_moves(states)
        policy *= valid_moves
        policy /= np.sum(policy, axis=1, keepdims=True)

        # Expand root node for each game in parallel games
        for i, g in enumerate(spGames):
            g.root = Node(self.game, self.args, states[i], visit_count=1)
            g.root.expand(policy[i])

        for _ in range(self.args['num_mcts_searches']):
            # Select node for each game
            for g in spGames:
                g.node = None
                node = g.root

                while node.is_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(
                    node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)
                else:
                    g.node = node

            # --- Conversion Block 2: Model Call for Node Expansion ---
            # Collect all nodes that can be expanded from all games
            expandable_spGames_indices = [
                i for i, g in enumerate(spGames) if g.node is not None]

            if len(expandable_spGames_indices) > 0:
                # Create new batch from node states to be expanded
                states_to_expand = np.stack(
                    [spGames[i].node.state for i in expandable_spGames_indices])

                tensor_states_to_expand = tf.convert_to_tensor(
                    self.game.get_encoded_state(states_to_expand), dtype=tf.float32
                )

                # Call model with this new batch. `training=False`.
                policy, value = self.model(
                    tensor_states_to_expand, training=False)

                policy = tf.nn.softmax(policy, axis=1).numpy()
                value = value.numpy()  # Result is already numpy array (batch_size, 1)

            # Perform expand and backpropagate for each node that was evaluated
            for i, game_idx in enumerate(expandable_spGames_indices):
                node = spGames[game_idx].node

                # Get policy and value corresponding to this node
                node_policy = policy[i]
                node_value = value[i]

                valid_moves = self.game.get_valid_moves(node.state)
                node_policy *= valid_moves
                node_policy /= np.sum(node_policy)

                node.expand(node_policy)
                node.backpropagate(node_value)
