import tensorflow as tf
from keras.optimizers import Adam

from config import *
from model.models import ResNet
from utils.game import TicTacToe
from model.alphaZeroParallel import AlphaZeroParallel

import os
import random
import numpy as np

tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        DEVICE = "/GPU:0"
        print(f"✅ GPU found, using: {DEVICE}")
        print(f"Number of GPUs available: {len(gpus)}")
    except RuntimeError as e:
        print(e)
        DEVICE = "/CPU:0"
        print(f"⚠️ Error while setting up GPU, switching to: {DEVICE}")
else:
    DEVICE = "/CPU:0"
    print(f"WARNING: NO GPU DETECTED! Using CPU: {DEVICE}")

if __name__ == '__main__':
    # Initialize TicTacToe
    game = TicTacToe()
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model = ResNet(game, num_resBlocks=RESBLOCKS, num_hidden=HIDDEN_UNITS)

    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    print(f"✅ Ensuring save directory exists at: {save_dir}")

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    if LOAD:
        try:
            latest_checkpoint = tf.train.latest_checkpoint(SAVE_DIR)
            if latest_checkpoint:
                print(f"Trying to load checkpoint from: {latest_checkpoint}")
                checkpoint.restore(latest_checkpoint).expect_partial()
                print("✅ Checkpoint loaded successfully.")
            else:
                print("⚠️ No checkpoint found to load.")
        except Exception as e:
            print(f"❌ Error while loading checkpoint: {e}")

    # Initialize AlphaZero
    alphaZero = AlphaZeroParallel(model, optimizer, game, args)

    # Running the entire process within the context of the device we've defined
    with tf.device(DEVICE):
        alphaZero.learn()
