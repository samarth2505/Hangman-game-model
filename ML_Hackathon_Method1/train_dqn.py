# train_dqn.py
# Runnable script to train the DQN agent.

import torch
import numpy as np
import random
import math
import time
from collections import Counter

import config
import utils
from dqn_agent import HangmanEnv, DQNAgent, build_input_vector

def evaluate_agent(agent, test_words, max_games=2000):
    """
    Runs the agent in pure-exploitation mode (eps=0.0) 
    to get final performance metrics.
    """
    games = min(len(test_words), max_games)
    if games == 0:
        return {'games': 0, 'successes': 0, 'total_wrong': 0, 'total_repeated': 0}
        
    successes = 0
    total_wrong = 0
    total_repeated = 0
    
    # Use the HangmanEnv from dqn_agent.py
    env = HangmanEnv(test_words, max_wrong=config.LIVES) 

    for i, w in enumerate(test_words[:games]):
        state = env.reset(word=w) 
        state_vec = build_input_vector(state)
        done = False
        steps = 0
        
        while not done:
            available_mask = np.array([ch not in state['guessed_set'] for ch in config.ALPHABET])
            act_idx = agent.select_action(state_vec, available_mask, eps=0.0) 
            act_letter = config.IDX2LETTER[act_idx]
            
            next_state, r, done, info = env.step(act_letter)
            
            state = next_state
            state_vec = build_input_vector(state)
            steps += 1
            if steps > config.MAX_LEN_CAP + 20: # Safety break
                break
                
        if 'result' in info and info['result'] == 'win':
            successes += 1
        total_wrong += env.wrong
        total_repeated += env.repeated_guesses
        
    return {'games': games, 'successes': successes, 'total_wrong': total_wrong, 'total_repeated': total_repeated}


def main():
    print("--- Starting DQN Training (Phase 2) ---")
    
    # Set seeds
    random.seed(config.RSEED)
    np.random.seed(config.RSEED)
    torch.manual_seed(config.RSEED)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load corpus for training
    all_words = utils.load_words(config.CORPUS_PATH)
    all_words = [w for w in all_words if all(ch in config.LETTER2IDX for ch in w) and len(w) <= config.MAX_LEN_CAP]
    random.shuffle(all_words)
    
    # Create train/test split
    test_size = min(2000, int(0.15 * len(all_words)))
    test_set = all_words[:test_size]
    train_pool = all_words[test_size:]
    print(f"Train pool size: {len(train_pool)}, Test size: {len(test_set)}")

    # Initialize Environment and Agent
    # Calculate input dim: (max_len * V) + V (guessed) + V (hmm) + 1 (lives)
    input_dim = (config.MAX_LEN_CAP * config.VOCAB_SIZE) + config.VOCAB_SIZE + config.VOCAB_SIZE + 1
    action_dim = config.VOCAB_SIZE
    print(f"Input dim: {input_dim}, Action dim: {action_dim}")
    
    agent = DQNAgent(input_dim, action_dim, device)
    env = HangmanEnv(train_pool, max_wrong=config.LIVES)
    
    # Training loop
    losses = []
    eval_history = []
    best_score = -float('inf')
    start_time = time.time()

    print(f"Starting training for {config.DQN_EPISODES} episodes...")
    for ep in range(1, config.DQN_EPISODES + 1):
        state = env.reset()
        state_vec = build_input_vector(state)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Epsilon-greedy action selection
            eps = config.DQN_EPS_FINAL + (config.DQN_EPS_START - config.DQN_EPS_FINAL) * \
                  math.exp(-1.0 * agent.steps / config.DQN_EPS_DECAY)
            
            available_mask = np.array([ch not in state['guessed_set'] for ch in config.ALPHABET])
            act_idx = agent.select_action(state_vec, available_mask, eps)
            act_letter = config.IDX2LETTER[act_idx]
            
            # Step the environment
            next_state, r, done, info = env.step(act_letter)
            next_state_vec = build_input_vector(next_state)
            
            # Store in replay buffer
            agent.push_transition(state_vec, act_idx, next_state_vec, r, float(done))
            
            # Update agent
            loss = agent.update()
            if loss:
                losses.append(loss)
                
            state_vec = next_state_vec
            state = next_state
            total_reward += r
            steps += 1
            if steps > config.MAX_LEN_CAP + 10: # Safety break
                break

        # Sync target network
        if ep % config.DQN_TARGET_UPDATE_EPISODES == 0:
            agent.sync_target()
            
        # Evaluate
        if ep % config.DQN_EVAL_EVERY == 0 or ep == config.DQN_EPISODES:
            eval_metrics = evaluate_agent(agent, test_set, max_games=len(test_set))
            
            success_rate = eval_metrics['successes'] / eval_metrics['games']
            total_wrong = eval_metrics['total_wrong']
            total_repeated = eval_metrics['total_repeated']
            
            # Calculate final score based on PDF
            final_score = (success_rate * 2000.0) - (total_wrong * 5.0) - (total_repeated * 2.0)
            eval_history.append((ep, final_score))
            
            print(f"[EP {ep}/{config.DQN_EPISODES}] Success: {success_rate:.3f}, "
                  f"Wrong: {total_wrong}, Repeated: {total_repeated}, "
                  f"Final Score: {final_score:.1f}")
            
            if final_score > best_score:
                best_score = final_score
                agent.save(config.DQN_MODEL_PATH)
                print(f"Saved new best model to {config.DQN_MODEL_PATH}")

    end_time = time.time()
    print(f"\n--- Training Finished ---")
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best validation score: {best_score:.2f}")

if __name__ == "__main__":
    main()