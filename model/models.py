import tensorflow as tf
from keras import layers, Model


class ResBlock(layers.Layer):
    def __init__(self, num_hidden):
        super(ResBlock, self).__init__()
        # Use convolutional with 3x3 kernel and same padding
        self.conv1 = layers.Conv2D(num_hidden, kernel_size=3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(num_hidden, kernel_size=3, padding='same')
        self.bn2 = layers.BatchNormalization()

    def call(self, x, training=None):  # `call` is the method equivalent to `forward` in PyTorch
        residual = x

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x, training=training)  # <-- `training` flag is propagated
        x = tf.nn.relu(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x, training=training)  # <-- `training` flag is propagated

        # Adding residual connection (skip connection)
        x = layers.add([x, residual])
        x = tf.nn.relu(x)
        return x


class ResNet(Model):
    def __init__(self, game, num_resBlocks, num_hidden):
        super(ResNet, self).__init__()

        # Getting board dimensions from `game` object to avoid hardcoding
        input_shape = (game.row_count, game.column_count, 3)

        self.startBlock = tf.keras.Sequential([
            # Input shape only needs to be defined in the first layer
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(num_hidden, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        # Backbone contains a stack of ResBlocks
        self.backBone = [ResBlock(num_hidden) for _ in range(num_resBlocks)]

        # Policy Head: Predicting action probabilities
        self.policyHead = tf.keras.Sequential([
            layers.Conv2D(32, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Flatten(),

            # Output total neurons same as `action_size` from game
            layers.Dense(game.action_size)
        ])

        # Value Head: Predicting game outcome (-1, 0, or 1)
        self.valueHead = tf.keras.Sequential([
            layers.Conv2D(3, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Flatten(),

            # The output is a single neuron with tanh activation (range -1 to 1)
            layers.Dense(1, activation='tanh')
        ])

    def call(self, x, training=None):
        x = self.startBlock(x, training=training)

        # Passing input through all ResBlocks in the backbone
        for resBlock in self.backBone:
            x = resBlock(x, training=training)

        # Calculating output from policy and value head
        policy = self.policyHead(x, training=training)
        value = self.valueHead(x, training=training)

        return policy, value
