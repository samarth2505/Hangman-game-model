# evaluate.py
# Runnable script to load the trained DQN agent and evaluate its
# final score on the official test set.

import torch
import numpy as np
import random
import time
from collections import Counter

import config
import utils
from dqn_agent import HangmanEnv, DQNAgent, build_input_vector

def main():
    print("--- Starting Final Evaluation (Phase 3) ---")
    
    # Set seeds for reproducibility of test set sampling
    random.seed(config.RSEED)
    np.random.seed(config.RSEED)
    
    # --- 1. Load Test Data ---
    print(f"Loading test corpus from {config.TEST_PATH}...")
    test_words = utils.load_words(config.TEST_PATH)
    test_words = [w for w in test_words if all(ch in config.LETTER2IDX for ch in w) and len(w) <= config.MAX_LEN_CAP]
    
    num_test_games = 2000
    if len(test_words) < num_test_games:
        print(f"Warning: Only {len(test_words)} unique test words. Reusing words.")
        eval_word_list = [test_words[i % len(test_words)] for i in range(num_test_games)]
    else:
        # Get a consistent random sample of 2000 words
        eval_word_list = random.sample(test_words, num_test_games)
        
    print(f"Loaded {len(test_words)} valid test words. Evaluating on {num_test_games} games.")

    # --- 2. Initialize Environment & Agent ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = (config.MAX_LEN_CAP * config.VOCAB_SIZE) + config.VOCAB_SIZE + config.VOCAB_SIZE + 1
    action_dim = config.VOCAB_SIZE
    
    agent = DQNAgent(input_dim, action_dim, device)
    
    try:
        agent.load(config.DQN_MODEL_PATH)
        print(f"Successfully loaded trained model from {config.DQN_MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {config.DQN_MODEL_PATH}.")
        return
    except Exception as e:
        print(f"An error occurred loading the model: {e}")
        return

    agent.online.eval() # Set to evaluation mode

    # Create an environment
    env = HangmanEnv(eval_word_list, max_wrong=config.LIVES)

    # --- 3. Run the Evaluation Loop ---
    total_wins = 0
    total_wrong_guesses = 0
    total_repeated_guesses = 0
    start_time = time.time()

    for i_game in range(num_test_games):
        game_wrong_this_turn = 0
        game_repeated_this_turn = 0

        # Reset environment with the specific test word
        state = env.reset(word=eval_word_list[i_game])
        state_vec = build_input_vector(state)
        done = False
        steps = 0

        while not done:
            # --- Make a Decision (Exploitation Only) ---
            available_mask = np.array([ch not in state['guessed_set'] for ch in config.ALPHABET])
            act_idx = agent.select_action(state_vec, available_mask, eps=0.0)
            act_letter = config.IDX2LETTER[act_idx]
            
            next_state, reward, done, info = env.step(act_letter)
            
            # --- Tally Statistics Based on Reward ---
            if reward == -2.0:
                game_repeated_this_turn += 1
            elif reward == -5.0 or reward == -10.0: # Wrong guess
                game_wrong_this_turn += 1
                
            state = next_state
            state_vec = build_input_vector(state)
            steps += 1
            if steps > config.MAX_LEN_CAP + 20: # Safety break
                break

        # --- Game Over: Tally Final Results ---
        if 'result' in info and info['result'] == 'win':
            total_wins += 1
            
        total_wrong_guesses += game_wrong_this_turn
        total_repeated_guesses += game_repeated_this_turn
        
        if (i_game + 1) % (num_test_games // 10) == 0:
            print(f"  ...completed game {i_game + 1}/{num_test_games}")

    end_time = time.time()
    print(f"--- Evaluation Finished in {end_time - start_time:.2f} seconds ---")

    # --- 4. Calculate Final Score (Using PDF Formula) ---
    success_rate = (total_wins / num_test_games)
    score_from_wins = success_rate * 2000.0
    penalty_from_wrong = total_wrong_guesses * 5.0
    penalty_from_repeated = total_repeated_guesses * 2.0

    final_score = score_from_wins - penalty_from_wrong - penalty_from_repeated

    # --- 5. Display Final Report ---
    print("\n" + "="*30)
    print(" 📊 FINAL HACKATHON SCORE 📊")
    print("="*30)
    print(f"Total Games Played: {num_test_games}")
    print(f"Total Wins:         {total_wins}")
    print(f"Success Rate:       {success_rate * 100:.2f}%")
    print(f"Avg. Wrong Guesses: {total_wrong_guesses / num_test_games:.2f}")
    print(f"Avg. Repeated Guesses: {total_repeated_guesses / num_test_games:.2f}")
    print("\n--- Scoring Breakdown ---")
    print(f"Score from Wins:    + {score_from_wins:.2f}")
    print(f"Penalty (Wrong):    - {penalty_from_wrong}")
    print(f"Penalty (Repeated): - {penalty_from_repeated}")
    print("------------------------------")
    print(f"🏆 FINAL SCORE: {final_score:.2f}")
    print("="*30)

if __name__ == "__main__":
    main()