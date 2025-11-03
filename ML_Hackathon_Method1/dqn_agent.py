# dqn_agent.py
# Defines the Hangman environment, the DQN model, and the agent logic.
# Also includes the HMM 'get_hmm_prior' function which acts as the oracle.

import numpy as np
import random
import pickle
from collections import deque, namedtuple, Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import configuration and utilities
import config
import utils

# --- Load HMM Data ---
# Load the trained HMMs and log-probabilities once when this module is imported.
# This avoids slow file I/O on every single game step.
try:
    with open(config.HMM_DATA_PATH, 'rb') as f:
        hmm_data = pickle.load(f)
    HMM_MODELS = hmm_data['models']
    LOGPROB_BY_KEY = hmm_data['logprob_by_key']
    print(f"Successfully loaded HMM data from {config.HMM_DATA_PATH}")
except FileNotFoundError:
    print(f"Error: HMM data file not found at {config.HMM_DATA_PATH}")
    print("Please run train_hmm.py first.")
    HMM_MODELS = {}
    LOGPROB_BY_KEY = {}
except Exception as e:
    print(f"Error loading HMM data: {e}")
    HMM_MODELS = {}
    LOGPROB_BY_KEY = {}

# --- HMM Oracle Function (from cell 11) ---

def get_key_for_length(L):
    """Finds the appropriate HMM model key (L# or B_bucket) for a given length."""
    key_exact = f"L{L}"
    if key_exact in HMM_MODELS:
        return key_exact
    
    # Fallback to buckets
    if L <= 4:
        key = "B_short"
    elif L <= 8:
        key = "B_medium"
    else:
        key = "B_long"
        
    # Final fallback if a bucket is empty
    if key not in HMM_MODELS:
        return list(HMM_MODELS.keys())[0]
    return key

def get_hmm_prior(masked, guessed_letters=None, top_k=config.HMM_TOP_K):
    """
    Calculates the probability distribution over un-guessed letters using the HMMs.
    """
    if guessed_letters is None:
        guessed_letters = set()
    
    guessed_wrong_set = set()
    guessed_correct_set = set()
    
    if isinstance(guessed_letters, dict):
        for ch, present in guessed_letters.items():
            if not present:
                guessed_wrong_set.add(ch)
            else:
                guessed_correct_set.add(ch)
    elif isinstance(guessed_letters, set):
        # If it's a set, we can't know which were wrong, 
        # but we can know which are not in the current mask.
        for ch in guessed_letters:
            if ch not in masked:
                guessed_wrong_set.add(ch)

    L = len(masked)
    key = get_key_for_length(L)
    candidates = []
    entries = LOGPROB_BY_KEY.get(key, [])
    
    # Iterate HMM log-probabilities
    for w, ll in entries:
        if utils.pattern_matches(w, masked, guessed_wrong_set):
            candidates.append((w, ll))
            if top_k and len(candidates) >= top_k:
                break
                
    # Fallback to simple letter frequency if no HMM candidates
    if len(candidates) == 0:
        freqs = Counter()
        total = 0
        for w, _ in entries:
            if len(w) != L: continue
            if guessed_wrong_set and any(ch in w for ch in guessed_wrong_set):
                continue
            for i, ch in enumerate(w):
                if masked[i] == '_' and ch not in guessed_letters:
                    freqs[ch] += 1
                    total += 1
        if total == 0: # Absolute fallback
            return {c: 1.0 / config.VOCAB_SIZE for c in config.ALPHABET}
        return {c: (freqs[c] / total) for c in config.ALPHABET}

    # Compute weights from log-likelihoods
    lls = np.array([ll for _, ll in candidates])
    maxll = np.max(lls)
    weights = np.exp(lls - maxll)
    post = weights / np.sum(weights) # Normalize to probabilities
    
    # Marginalize to get letter probabilities
    letter_mass = Counter()
    for (w, _), p in zip(candidates, post):
        for i, ch in enumerate(w):
            if masked[i] == '_' and ch not in guessed_letters:
                letter_mass[ch] += p
                
    total_mass = sum(letter_mass.values())
    if total_mass == 0: # Fallback if all letters in blanks are already guessed
        return {c: 1.0 / config.VOCAB_SIZE for c in config.ALPHABET}
        
    probs = {c: (letter_mass[c] / total_mass) for c in config.ALPHABET}
    return probs

# --- Hangman Environment (from cell 19, with CORRECTED rewards) ---
class HangmanEnv:
    def __init__(self, word_list, max_wrong=config.LIVES, max_len=config.MAX_LEN_CAP):
        self.word_list = word_list
        self.max_wrong = max_wrong
        self.max_len = max_len
        self.V = config.VOCAB_SIZE
        self.word = ""
        self.mask = []
        self.guessed = set()
        self.wrong = 0
        self.done = False
        self.repeated_guesses = 0
        
    def reset(self, word=None):
        if word is None:
            self.word = random.choice(self.word_list)
        else:
            self.word = word
            
        self.mask = ['_'] * len(self.word)
        self.guessed = set()
        self.wrong = 0
        self.done = False
        self.repeated_guesses = 0
        return self.get_state()

    def get_state(self):
        """Returns the current state dictionary."""
        masked = ''.join(self.mask)
        lives_left = self.max_wrong - self.wrong
        
        # Build guessed_dict for HMM prior
        guessed_dict = {}
        for ch in config.ALPHABET:
            if ch in self.guessed:
                guessed_dict[ch] = (ch in self.word)
                
        hmm_prior_dict = get_hmm_prior(masked, guessed_letters=guessed_dict)
        
        return {
            'masked': masked, 
            'guessed_set': set(self.guessed), 
            'lives_left': lives_left, 
            'hmm_prior': hmm_prior_dict
        }

    def step(self, action_letter):
        """
        Takes an action (letter) and returns (state, reward, done, info).
        Uses the PDF-aligned reward system.
        """
        if self.done:
            raise RuntimeError("step() called on finished env")
        
        info = {'result': 'running'}
        reward = 0.0

        if action_letter in self.guessed:
            self.repeated_guesses += 1
            reward = -2.0  # PDF Penalty: (Total Repeated Guesses * 2)
            info['repeated'] = True
            return self.get_state(), reward, self.done, info

        self.guessed.add(action_letter)
        
        if action_letter in self.word:
            # Correct guess
            for i, ch in enumerate(self.word):
                if ch == action_letter:
                    self.mask[i] = ch
            reward = 1.0  # Small positive shaping reward for progress
        else:
            # Wrong guess
            self.wrong += 1
            reward = -5.0  # PDF Penalty: (Total Wrong Guesses * 5)
        
        # Check for game end
        if '_' not in self.mask:
            self.done = True
            reward = 20.0  # Large scaled bonus for winning (Success Rate * 2000)
            info['result'] = 'win'
        elif self.wrong >= self.max_wrong:
            self.done = True
            reward = -10.0 # Extra penalty for losing
            info['result'] = 'loss'
            
        return self.get_state(), reward, self.done, info

# --- State Vector Builder (from cell 21) ---
def build_input_vector(state):
    """Converts the state dictionary into a flat numpy vector for the DQN."""
    masked = state['masked']
    guessed_set = state['guessed_set']
    hmm_prior = state['hmm_prior']
    lives = state['lives_left']
    
    max_len = config.MAX_LEN_CAP
    Vloc = config.VOCAB_SIZE
    
    # 1. Masked word (one-hot encoded, flattened)
    masked_oh = np.zeros((max_len, Vloc), dtype=np.float32)
    for i in range(max_len):
        if i < len(masked):
            ch = masked[i]
            if ch != '_':
                idx = config.LETTER2IDX[ch]
                masked_oh[i, idx] = 1.0
    masked_flat = masked_oh.flatten()
    
    # 2. Guessed letters (binary)
    guessed_bin = np.zeros(Vloc, dtype=np.float32)
    for ch in guessed_set:
        if ch in config.LETTER2IDX:
            guessed_bin[config.LETTER2IDX[ch]] = 1.0
            
    # 3. HMM prior (probabilities)
    hmm_vec = np.array([hmm_prior.get(config.IDX2LETTER[i], 0.0) for i in range(Vloc)], dtype=np.float32)
    
    # 4. Lives (normalized)
    lives_norm = np.array([lives / config.LIVES], dtype=np.float32)
    
    # Concatenate all features
    inp = np.concatenate([masked_flat, guessed_bin, hmm_vec, lives_norm])
    return inp

# --- DQN Model & Agent (from cells 20 & 22) ---

class DQNNet(nn.Module):
    """The neural network architecture for the DQN."""
    def __init__(self, input_dim, output_dim, hidden=config.DQN_HIDDEN_UNITS):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """A simple replay buffer for storing experiences."""
    def __init__(self, capacity=config.DQN_BUFFER_CAPACITY):
        self.buf = deque(maxlen=capacity)
        
    def push(self, *args):
        self.buf.append(Transition(*args))
        
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))
        
    def __len__(self):
        return len(self.buf)

class DQNAgent:
    """The agent that manages the DQN, target network, and optimization."""
    def __init__(self, input_dim, action_dim, device):
        self.action_dim = action_dim
        self.device = device
        self.gamma = config.DQN_GAMMA
        
        self.online = DQNNet(input_dim, action_dim).to(device)
        self.target = DQNNet(input_dim, action_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        
        self.opt = optim.Adam(self.online.parameters(), lr=config.DQN_LR)
        self.replay = ReplayBuffer()
        self.steps = 0

    def select_action(self, state_vec, available_mask, eps):
        """Selects an action using an epsilon-greedy policy."""
        if random.random() < eps:
            allowed_idxs = [i for i, b in enumerate(available_mask) if b]
            if not allowed_idxs:
                # Fallback: if no actions are available (shouldn't happen in hangman)
                return random.randrange(self.action_dim)
            return random.choice(allowed_idxs)
        
        # Exploitation
        self.online.eval()
        with torch.no_grad():
            x = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
            q_values = self.online(x).cpu().numpy().flatten()
            
            # Mask out unavailable actions
            q_masked = np.full_like(q_values, -np.inf)
            for i, is_available in enumerate(available_mask):
                if is_available:
                    q_masked[i] = q_values[i]
                    
            act = int(np.argmax(q_masked))
        self.online.train()
        return act

    def push_transition(self, *args):
        self.replay.push(*args)

    def update(self, batch_size=config.DQN_BATCH_SIZE):
        """Performs one step of optimization on the online network."""
        if len(self.replay) < batch_size:
            return 0.0
            
        trans = self.replay.sample(batch_size)
        
        state_b = torch.from_numpy(np.vstack(trans.state)).float().to(self.device)
        next_state_b = torch.from_numpy(np.vstack(trans.next_state)).float().to(self.device)
        action_b = torch.tensor(trans.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_b = torch.tensor(trans.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        done_b = torch.tensor(trans.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 1. Get Q(s, a)
        q_vals = self.online(state_b).gather(1, action_b)
        
        # 2. Get max Q(s', a')
        with torch.no_grad():
            next_q = self.target(next_state_b).max(1)[0].unsqueeze(1)
            target_q = reward_b + (1.0 - done_b) * self.gamma * next_q
            
        # 3. Calculate loss
        loss = F.mse_loss(q_vals, target_q)
        
        # 4. Optimize
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1
        return loss.item()

    def sync_target(self):
        """Copies weights from the online network to the target network."""
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path):
        """Saves the online network's state dictionary."""
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        """Loads weights into the online network and syncs the target."""
        self.online.load_state_dict(torch.load(path, map_location=self.device))
        self.sync_target()
        self.online.eval()