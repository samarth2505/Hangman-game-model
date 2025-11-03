# Intelligent Hangman Agent (HMM + DQN)

This project is an intelligent Hangman-solving agent built for the **UE23CS352A: Machine Learning Hackathon**. It uses a hybrid approach combining a **Hidden Markov Model (HMM)** for language "intuition" and a **Deep Q-Network (DQN)** for strategic decision-making to win games with maximum efficiency.

## 🎯 The Challenge

The goal is to create an agent that can effectively guess letters to solve Hangman puzzles. The agent is evaluated by playing 2,000 games against a hidden test set.

The final score is calculated using a formula that heavily penalizes mistakes:

## 🧠 Core Architecture & Methodology

This solution is a hybrid AI model, broken into two main phases as required by the challenge.

### Phase 1: The "Oracle" — Hidden Markov Model (HMM)

The agent's "intuition" comes from a generative HMM trained on the `corpus.txt` file. Its job is to provide a probability distribution (a "hint") over the 26 letters based on the current game state.

* **Training (Baum-Welch):** The `DiscreteHMM` class uses the **Baum-Welch (Expectation-Maximization)** algorithm to train the HMM. This is an unsupervised method that finds "hidden states" that best represent the language's contextual patterns (e.g., a state for vowels, a state for consonants after 'S', etc.).
* **Per-Length Modeling:** To handle words of different lengths (as suggested in the hints), the code trains **separate, specialized HMMs for each word length** (e.g., one model for all 5-letter words, one for 6-letter words, etc.).
* **Hyperparameter Tuning (K-Sweep):** Before training, the code sweeps through different numbers of hidden states (`K_options = [8, 16, 32]`) and picks the best `K` for each word length based on its validation log-likelihood.
* **Output (`get_hmm_prior`):** This is the final function from Phase 1. It filters the corpus for words matching the current pattern (e.g., `_ A _ _ L E`), weights them by their HMM probability, and calculates the final probability for each un-guessed letter. This 26-element vector is the HMM's "hint".

### Phase 2: The "Brain" — Deep Q-Network (DQN)

The agent's "brain" is a **Deep Q-Network (DQN)**, a Reinforcement Learning (RL) agent tasked with using the HMM's hint to make the optimal strategic decision.

* **Environment (`HangmanEnv`):** A custom environment was built to simulate the game. Its **reward function** is the most critical part, as it's directly aligned with the hackathon's scoring formula to force the agent to learn the correct strategy:
    * **Wrong Guess:** `reward = -5.0`
    * **Repeated Guess:** `reward = -2.0`
    * **Win Game:** `reward = +20.0` (large bonus for success)
    * **Lose Game:** `reward = -10.0` (large penalty for failure)

* **Hybrid State (`build_input_vector`):** The DQN is fed a state vector that combines all relevant information:
    1.  The current masked word (flattened).
    2.  A binary vector of already-guessed letters.
    3.  The number of lives left.
    4.  The 26-probability **"hint" vector** from the HMM.

* **Training:** The `DQNAgent` is trained using standard, effective RL methodologies:
    * **Experience Replay:** Stores past moves in a `ReplayBuffer` to break correlations and stabilize learning.
    * **Target Network:** Uses a separate `target_net` for stable Q-value estimation.
    * **$\epsilon$-greedy Policy:** Balances **exploration** (random guesses to find new strategies) with **exploitation** (using the network's best guess).

## ⚙️ Requirements

This project is built as a single Jupyter/Kaggle notebook.
* `python 3.x`
* `numpy`
* `torch` (for the DQN)
* `google.colab` (or standard `pathlib` for file management)

## 🚀 How to Run

The notebook (`ml_hackathon (1).ipynb`) is designed to be run from top to bottom.

1.  **Cell 1-12: HMM Training**
    * Run these cells first. This will load `corpus.txt`, perform the K-sweep, train the HMMs for each word length, and define the `get_hmm_prior` function. This is the "oracle."

2.  **Cell 18-24: DQN Training**
    * Run the environment definition (`HangmanEnv`), the agent classes (`DQNNet`, `DQNAgent`), and the main `train_dqn` loop.
    * This is the longest step and will take considerable time (depending on the number of `episodes`).
    * This will save the trained "brain" to a file named `dqn_hangman.pt`.

3.  **Cell 26: Final Evaluation**
    * This final cell loads the `test.txt` file and the saved `dqn_hangman.pt` model.
    * It runs the agent in pure **exploitation mode (epsilon = 0)** for 2,000 games to get the official score.

## 📈 Results

The training script (cell 24) will output the agent's progress and save the best model. The final evaluation script (cell 26) will output the official score breakdown:
