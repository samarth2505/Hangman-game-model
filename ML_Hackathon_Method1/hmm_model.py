# hmm_model.py
# Contains the DiscreteHMM class and its training function.

import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import config
import utils

# --- Log-space math helpers (from cell 7) ---
def logsumexp(a, axis=None):
    a = np.asarray(a)
    a_max = np.max(a, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return (a_max + s).squeeze(axis)

# --- DiscreteHMM Class (from cell 7) ---
class DiscreteHMM:
    def __init__(self, n_states, n_symbols, random_state=42, eps=1e-12):
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.random_state = random_state
        self.eps = eps
        rng = np.random.RandomState(random_state)
        self.startprob_ = rng.rand(n_states)
        self.startprob_ /= self.startprob_.sum()
        self.transmat_ = rng.rand(n_states, n_states)
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)
        self.emissionprob_ = rng.rand(n_states, n_symbols)
        self.emissionprob_ /= self.emissionprob_.sum(axis=1, keepdims=True)
        # log forms will be computed when needed
        self._to_log()

    def _to_log(self):
        self.log_startprob_ = np.log(self.startprob_ + self.eps)
        self.log_transmat_ = np.log(self.transmat_ + self.eps)
        self.log_emissionprob_ = np.log(self.emissionprob_ + self.eps)

    def _forward_log(self, obs):
        T = len(obs)
        N = self.n_states
        logalpha = np.full((T, N), -np.inf)
        logalpha[0] = self.log_startprob_ + self.log_emissionprob_[:, obs[0]]
        for t in range(1, T):
            prev = logalpha[t-1]
            temp = prev[:, None] + self.log_transmat_
            logalpha[t] = logsumexp(temp, axis=0) + self.log_emissionprob_[:, obs[t]]
        loglik = logsumexp(logalpha[-1], axis=0)
        return logalpha, loglik

    def _backward_log(self, obs):
        T = len(obs)
        N = self.n_states
        logbeta = np.full((T, N), -np.inf)
        logbeta[-1] = 0.0
        for t in range(T-2, -1, -1):
            temp = self.log_transmat_ + (self.log_emissionprob_[:, obs[t+1]] + logbeta[t+1])[None, :]
            logbeta[t] = logsumexp(temp, axis=1)
        return logbeta

    def score(self, obs):
        self._to_log()
        _, ll = self._forward_log(obs)
        return float(ll)

    def fit(self, sequences, n_iter=50, tol=1e-4, verbose=False):
        # sequences: list of 1D int arrays
        N = self.n_states
        M = self.n_symbols
        prev_ll = None
        for it in range(n_iter):
            start_counts = np.zeros(N)
            trans_counts = np.zeros((N, N))
            emit_counts = np.zeros((N, M))
            total_ll = 0.0
            self._to_log()
            for obs in sequences:
                T = len(obs)
                if T == 0:
                    continue
                logalpha, loglik = self._forward_log(obs)
                logbeta = self._backward_log(obs)
                total_ll += loglik
                loggamma = logalpha + logbeta - loglik
                gamma = np.exp(loggamma)
                start_counts += gamma[0]
                for t in range(T):
                    emit_counts[:, obs[t]] += gamma[t]
                if T > 1:
                    for t in range(T-1):
                        temp = (logalpha[t][:, None] +
                                self.log_transmat_ +
                                (self.log_emissionprob_[:, obs[t+1]] + logbeta[t+1])[None, :])
                        logxi = temp - loglik
                        xi = np.exp(logxi)
                        trans_counts += xi
            # M-step with tiny smoothing
            self.startprob_ = start_counts + 1e-8
            self.startprob_ /= self.startprob_.sum()
            row_sums = trans_counts.sum(axis=1, keepdims=True)
            row_sums[row_sums==0] = 1.0
            self.transmat_ = (trans_counts + 1e-8) / row_sums
            e_row_sums = emit_counts.sum(axis=1, keepdims=True)
            e_row_sums[e_row_sums==0] = 1.0
            self.emissionprob_ = (emit_counts + 1e-8) / e_row_sums
            if verbose:
                print(f"Iter {it+1}: total_loglik = {total_ll:.6f}")
            if prev_ll is not None and abs(total_ll - prev_ll) < tol:
                if verbose:
                    print("Converged.")
                break
            prev_ll = total_ll
        self._to_log()
        return self

# --- HMM Training Function (from cell 9) ---
def train_hmm_with_K_sweep(word_list, K_options, n_iter, test_size, verbose=False):
    """
    word_list: list of words (strings) for this length/bucket
    returns: best_model, best_K, dict of {K: avg_val_ll}
    """
    # prepare sequences
    word_list = [w for w in word_list if all(ch in config.LETTER2IDX for ch in w)]
    train, val = train_test_split(word_list, test_size=test_size, random_state=config.RSEED)
    seqs_train = [utils.word_to_obs(w) for w in train]
    results = {}
    models_for_K = {}
    
    for K in K_options:
        print(f"Training K={K} on {len(train)} words ...")
        model = DiscreteHMM(n_states=K, n_symbols=config.VOCAB_SIZE, random_state=config.RSEED)
        model.fit(seqs_train, n_iter=n_iter, tol=1e-4, verbose=verbose)
        
        # eval on val
        val_lls = []
        for w in val:
            try:
                val_lls.append(model.score(utils.word_to_obs(w)))
            except:
                val_lls.append(float('-inf'))
        avg_val_ll = float(np.mean(val_lls))
        results[K] = avg_val_ll
        models_for_K[K] = model
        print(f"K={K} avg_val_ll={avg_val_ll:.3f}")
        
    # pick best K (highest avg val ll)
    best_K = max(results, key=lambda k: results[k])
    best_model = models_for_K[best_K]
    return best_model, best_K, results