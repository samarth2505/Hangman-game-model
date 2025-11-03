# train_hmm.py
# Runnable script to train all HMMs and save them to a pickle file.

import pickle
from collections import defaultdict
import numpy as np

import config
import utils
import hmm_model

def main():
    print("--- Starting HMM Training (Phase 1) ---")
    
    # Load and process corpus
    words = utils.load_words(config.CORPUS_PATH)
    
    # --- Group words by length (from cell 5) ---
    words_by_len = defaultdict(list)
    for w in words:
        L = len(w)
        if L > config.MAX_LEN_CAP:
            continue
        words_by_len[L].append(w)

    lengths_to_train = [
        L for L, c in words_by_len.items() 
        if len(c) >= config.MIN_EXAMPLES_PER_LENGTH
    ]
    lengths_to_train = sorted(lengths_to_train)
    print(f"Exact lengths to train: {lengths_to_train}")

    buckets = {'short': [], 'medium': [], 'long': []}
    for L, wlist in words_by_len.items():
        if L in lengths_to_train:
            continue
        if L <= 4:
            buckets['short'].extend(wlist)
        elif L <= 8:
            buckets['medium'].extend(wlist)
        else:
            buckets['long'].extend(wlist)
    
    for k in buckets:
        print(f"Bucket {k}: {len(buckets[k])} words")

    # --- Train HMMs (from cell 10) ---
    models = {}           # key -> model object and meta
    logprob_by_key = {}   # key -> list of (word, logprob) precomputed

    for L in lengths_to_train:
        wlist = words_by_len[L]
        print(f"\n== Training for exact length {L}, words={len(wlist)} ==")
        best_model, best_K, stats = hmm_model.train_hmm_with_K_sweep(
            wlist, 
            K_options=config.K_OPTIONS, 
            n_iter=config.HMM_N_ITER, 
            test_size=config.HMM_TEST_SIZE
        )
        key = f"L{L}"
        models[key] = {
            'model': best_model, 
            'type': 'length', 
            'length': L, 
            'best_K': best_K, 
            'n_words': len(wlist)
        }
        
        # Precompute log-likelihoods (for get_hmm_prior)
        entries = []
        for w in wlist:
            try:
                ll = float(best_model.score(utils.word_to_obs(w)))
            except:
                ll = -1e9
            entries.append((w, ll))
        entries.sort(key=lambda x: x[1], reverse=True)
        logprob_by_key[key] = entries
        
    for bname, wlist in buckets.items():
        if len(wlist) == 0:
            continue
        print(f"\n== Training for bucket '{bname}', words={len(wlist)} ==")
        best_model, best_K, stats = hmm_model.train_hmm_with_K_sweep(
            wlist, 
            K_options=config.K_OPTIONS, 
            n_iter=config.HMM_N_ITER, 
            test_size=config.HMM_TEST_SIZE
        )
        key = f"B_{bname}"
        models[key] = {
            'model': best_model, 
            'type': 'bucket', 
            'bucket': bname, 
            'best_K': best_K, 
            'n_words': len(wlist)
        }
        
        # Precompute log-likelihoods
        entries = []
        for w in wlist:
            try:
                ll = float(best_model.score(utils.word_to_obs(w)))
            except:
                ll = -1e9
            entries.append((w, ll))
        entries.sort(key=lambda x: x[1], reverse=True)
        logprob_by_key[key] = entries

    # --- Save the trained data ---
    hmm_data = {
        'models': models,
        'logprob_by_key': logprob_by_key
    }
    
    with open(config.HMM_DATA_PATH, 'wb') as f:
        pickle.dump(hmm_data, f)
        
    print(f"\n--- HMM Training Complete ---")
    print(f"Saved HMM models and log-probabilities to {config.HMM_DATA_PATH}")

if __name__ == "__main__":
    main()