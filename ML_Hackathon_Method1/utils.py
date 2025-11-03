# utils.py
# Helper functions for data loading, cleaning, and pattern matching.

import numpy as np
import config

def load_words(path):
    """Loads a list of words from a text file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = [ln.strip() for ln in f if ln.strip()]
        # Filter empty and non-alphabetic words
        words = [w.lower() for w in raw if w and w.isalpha()]
        
        # Deduplicate while preserving order
        seen = set()
        cleaned_unique = []
        for w in words:
            if w not in seen:
                seen.add(w)
                cleaned_unique.append(w)
        
        print(f"Loaded {len(cleaned_unique)} unique, clean words from {path}")
        return cleaned_unique
    except FileNotFoundError:
        print(f"Error: Word file not found at {path}")
        return []

def word_to_obs(word):
    """Converts a string word to a numpy array of letter indices."""
    return np.array([config.LETTER2IDX[c] for c in word], dtype=np.int64)

def obs_to_word(obs):
    """Converts a numpy array of indices back to a string word."""
    return ''.join(config.IDX2LETTER[int(i)] for i in obs)

def pattern_matches(word, masked, guessed_wrong_set=None):
    """Checks if a word fits a masked pattern, given wrong guesses."""
    if len(word) != len(masked):
        return False
    
    # Check if word matches the revealed letters
    for wc, mc in zip(word, masked):
        if mc != '_' and mc != wc:
            return False
            
    # Check if word contains any letters guessed wrong
    if guessed_wrong_set:
        # Optimization: create set of word letters
        word_letters = set(word)
        for ch in guessed_wrong_set:
            if ch in word_letters:
                return False
    
    return True