args = {
    'num_iterations': 10,
    'num_selfPlay_iterations': 350,
    'num_parallel_games': 100,
    'num_mcts_searches': 100,
    'num_epochs': 5,
    'batch_size': 64,
    'temperature': 1.25,
    'C': 3,
    'augment': True,
    'dirichlet_alpha': 0.3,
    'dirichlet_epsilon': 0.125,
}

SAVE_DIR = "./ckpt/"

# Set the path to the latest model weights, for example: .\models\model_<numberIterations>.weights.h5
LATEST_MODEL_PATH = SAVE_DIR + "model_9.weights.h5"

LOAD = False  # Set to True if you want to load a model checkpoint from the previous training
LEARNING_RATE = 0.001
RESBLOCKS = 4
HIDDEN_UNITS = 64
