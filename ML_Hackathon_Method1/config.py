# config.py
# Holds all global constants and configuration variables for the project.

import string

# --- General ---
RSEED = 42
LIVES = 6 # From PDF
MAX_LEN_CAP = 17 # 99th percentile from your data exploration (cell 5)

# --- HMM Config ---
MIN_EXAMPLES_PER_LENGTH = 300
K_OPTIONS = [8, 16, 32] # HMM hidden states to try
HMM_N_ITER = 80
HMM_TEST_SIZE = 0.15
HMM_TOP_K = 2000 # Number of HMM candidates to consider in get_hmm_prior

# --- DQN Config ---
# Training
DQN_EPISODES = 50000 # Increase this from 3000 for better results
DQN_EVAL_EVERY = 500
DQN_BATCH_SIZE = 128
DQN_LR = 1e-3
DQN_GAMMA = 0.98
DQN_EPS_START = 1.0
DQN_EPS_FINAL = 0.05
DQN_EPS_DECAY = 20000 # Fixed decay steps

# Model
DQN_HIDDEN_UNITS = 256
DQN_BUFFER_CAPACITY = 100000
DQN_TARGET_UPDATE_EPISODES = 200 # How often to sync target network

# --- File Paths ---
CORPUS_PATH = "/kaggle/input/hackathon-files/corpus.txt" # Adjust as needed
TEST_PATH = "/kaggle/input/hackathon-files/test.txt"     # Adjust as needed

HMM_DATA_PATH = "hmm_data.pkl"         # Where to save the trained HMMs
DQN_MODEL_PATH = "dqn_hangman.pt"    # Where to save the trained DQN

# --- Alphabet ---
ALPHABET = sorted(list(string.ascii_lowercase))
LETTER2IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX2LETTER = {i: c for c, i in LETTER2IDX.items()}
VOCAB_SIZE = len(ALPHABET) # Should be 26