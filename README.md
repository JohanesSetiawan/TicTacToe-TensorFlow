# AlphaZero for Tic-Tac-Toe (TensorFlow Edition)

Hey there! Welcome to my little project where I taught an AI to master the classic game of Tic-Tac-Toe. This isn't your standard minimax algorithm; this AI learned how to play from scratch, just by playing against itself thousands of times. It's a fun, interactive way to see the power of reinforcement learning in action.

## The Backstory

I've always been blown away by what DeepMind accomplished with AlphaGo and AlphaZero. The idea of an AI achieving superhuman ability in complex games without any human-provided strategy is just wild.

I stumbled upon an awesome PyTorch implementation of AlphaZero by **Robert Foerster** and thought it was a fantastic learning resource. Since my go-to framework is TensorFlow, I set myself a challenge: could I re-write the entire project in TensorFlow and get a feel for how all the components connect? This repo is the result of that journey.

## How It Works (The Gist)

The AI doesn't have any hardcoded rules like "take the center square if it's open." Instead, it figures out the best moves on its own. Guts of the process are:

1.  **Self-Play:** The AI plays thousands of games against itself to gather data. Every move, win, loss, and draw is recorded as an "experience."
2.  **Learning:** A **Residual Neural Network (ResNet)** then trains on this data. It learns to do two things:
      * **Policy:** Predict the probability of winning from each possible next move.
      * **Value:** Guess the eventual outcome of the game (win, lose, draw) from the current board state.
3.  **Thinking:** When it's the AI's turn to play, it uses a **Monte Carlo Tree Search (MCTS)** algorithm. This MCTS acts as the "thinking" part, simulating dozens of future game paths. The neural network's "intuition" guides this search, telling it which paths are more promising to explore.

The cycle then repeats: the smarter AI plays more games, generates better data, and trains to become even smarter.

## The Good & The Could-Be-Better

  * **The Good:**

      * The AI gets surprisingly strong\! After just a few iterations, it plays perfectly and is impossible to beat.
      * The Pygame GUI is a fun, visual way to interact with and test the trained model.
      * The code is well-structured, separating the game logic, model architecture, and training process.

  * **The Could-Be-Better:**

      * The GUI is pretty basic, but it gets the job done for playing the game.
      * While the hyperparameters in `config.py` are tuneable, finding the "perfect" set for a new game would require some experimentation.

-----

## How to Run

It's pretty straightforward. Just clone the repo and get your environment set up.

#### 1\. Install library
```bash
pip install --upgrade tensorflow-cpu tqdm pygame
```

#### 2\. Training the AI

If you want to train your own model from scratch, just run:

```bash
python main.py
```

This will start the self-play and training loop, saving model weights in the `.\ckpt\` directory after each iteration.

#### 3\. Playing Against the AI

To play a game against the pre-trained model, run:

```bash
python play_gui.py
```

⚠️ **Heads Up\!** This uses **Pygame**, which requires a desktop environment with a display. You **cannot** run this GUI version in a pure command-line environment like a standard Google Colab cell.

-----

## Tinkering with the AI's Brain (`config.py`)

Want to experiment? The `config.py` file is where you can tweak all the hyperparameters for training and inference.

```python
# config.py
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

SAVE_DIR = "./Models/"

# Set the path to the model weights you want to load for inference
LATEST_MODEL_PATH = SAVE_DIR + "TicTacToe/model_9.weights.h5"

LOAD = False  # Set to True if you want to resume training from a checkpoint
LEARNING_RATE = 0.001
RESBLOCKS = 4
HIDDEN_UNITS = 64
```

🚨 **CRITICAL WARNING:** The parameters `RESBLOCKS` and `HIDDEN_UNITS` define the **architecture** of the neural network. If you change these values, you **cannot** load the pre-trained weights, as the "shape" of the model will be different. This will cause a complex error. If you change the architecture, you **must** train a new model from scratch.

-----

## Pre-Trained Model

I've included the weights for a fully trained model that plays optimally. The configuration used to train this model is listed below. **To use these weights, your `config.py` must have the same `RESBLOCKS` and `HIDDEN_UNITS` values.**

| NAME\_WEIGHTS        | RESBLOCKS | HIDDEN\_UNITS | ARGS                                                                                     |
| :------------------- | :-------: | :-----------: | :--------------------------------------------------------------------------------------- |
| `model_9.weights.h5` |     4     |      64       | `num_iterations: 10`, `num_mcts_searches: 100`, etc. (see `config.py` for the full list) |

-----

## Acknowledgements & Shout-Outs

This project wouldn't have been possible without standing on the shoulders of giants.

  * A huge shout-out to **Robert Foerster** for his clean and understandable PyTorch implementation of AlphaZero, which served as the foundation for this project. You can find his original work here: [foersterrobert/AlphaZero](https://github.com/foersterrobert/AlphaZero).
  * Big thanks to **@google** for developing the Gemini AI model. I had Gemini 2.5 Pro as a pair-programming partner throughout the entire PyTorch-to-TensorFlow conversion, debugging, and enhancement process.