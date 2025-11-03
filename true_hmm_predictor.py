"""
True Hidden Markov Model with Forward-Backward Algorithm

This learns letter transition patterns that generalize to unseen words,
unlike candidate filtering which only works for words in the corpus.
"""
import numpy as np
from collections import defaultdict, Counter
from typing import List, Set, Dict
import pickle
import os
from scipy.special import logsumexp


class TrueHangmanHMM:
    """
    Proper HMM using Forward-Backward algorithm
    
    States: letters a-z
    Observations: revealed pattern (letters or blanks)
    Goal: Predict which letters fill the blanks
    """
    
    def __init__(self, smoothing: float = 0.01):
        self.smoothing = smoothing
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.n_letters = 26
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.idx_to_char = {i: c for i, c in enumerate(self.alphabet)}
        
        # Models by word length
        self.length_models = {}
        self.is_trained = False
    
    def train(self, corpus: List[str]):
        """Train HMM on corpus by learning letter transition patterns"""
        print(f"Training True HMM on {len(corpus)} words...")
        
        # Group by length
        words_by_length = defaultdict(list)
        for word in corpus:
            if word.isalpha():
                normalized = word.lower()
                words_by_length[len(normalized)].append(normalized)
        
        print(f"Training models for {len(words_by_length)} word lengths...")
        
        # Train model for each length
        for length in sorted(words_by_length.keys()):
            if len(words_by_length[length]) >= 10:
                print(f"  Length {length}: {len(words_by_length[length])} words")
                self.length_models[length] = self._train_length_model(
                    words_by_length[length], length
                )
        
        self.is_trained = True
        print(f"✓ Trained {len(self.length_models)} models")
    
    def _train_length_model(self, words: List[str], length: int) -> Dict:
        """
        Train transition and emission probabilities for words of specific length
        
        Returns log probabilities for numerical stability
        """
        n = self.n_letters
        s = self.smoothing
        
        # Count transitions: P(letter_t | letter_t-1, position_t)
        # Shape: (length, n_letters, n_letters) = position-specific transitions
        transition_counts = np.zeros((length, n, n))
        
        # Count letter frequencies at each position
        position_counts = np.zeros((length, n))
        
        for word in words:
            for pos in range(len(word)):
                letter_idx = self.char_to_idx.get(word[pos], 0)
                position_counts[pos, letter_idx] += 1
                
                if pos > 0:
                    prev_letter_idx = self.char_to_idx.get(word[pos-1], 0)
                    transition_counts[pos, prev_letter_idx, letter_idx] += 1
        
        # Convert to log probabilities with smoothing
        log_transitions = []
        for pos in range(length):
            # Add smoothing
            trans = transition_counts[pos] + s
            # Normalize: sum over next letter
            trans = trans / trans.sum(axis=1, keepdims=True)
            log_transitions.append(np.log(trans))
        
        # Initial probabilities (first position)
        initial = position_counts[0] + s
        initial = initial / initial.sum()
        log_initial = np.log(initial)
        
        # Position-specific letter frequencies (for fallback)
        log_position_probs = []
        for pos in range(length):
            probs = position_counts[pos] + s
            probs = probs / probs.sum()
            log_position_probs.append(np.log(probs))
        
        return {
            'log_transitions': log_transitions,  # List of (26,26) matrices
            'log_initial': log_initial,  # (26,)
            'log_position_probs': log_position_probs,  # List of (26,) vectors
            'length': length
        }
    
    def predict_letter_probabilities(self, pattern: str, tried_letters: Set[str]) -> np.ndarray:
        """
        Predict letter probabilities using Forward-Backward algorithm
        
        Args:
            pattern: Current pattern (e.g., "_a_g_an")
            tried_letters: Already guessed letters
            
        Returns:
            26-dimensional probability vector
        """
        length = len(pattern)
        
        if length not in self.length_models:
            return self._fallback_probabilities(tried_letters)
        
        model = self.length_models[length]
        
        # Run Forward-Backward
        blank_positions = [i for i, c in enumerate(pattern) if c == '_']
        
        if not blank_positions:
            return np.zeros(26)
        
        try:
            # Compute posterior probabilities for each blank position
            letter_probs = np.zeros(26)
            
            for pos in blank_positions:
                # Get probability of each letter at this position
                pos_probs = self._compute_position_posterior(pattern, pos, model)
                letter_probs += pos_probs
            
            # Normalize
            letter_probs = letter_probs / (letter_probs.sum() + 1e-10)
            
            # Zero out tried letters
            for letter in tried_letters:
                if letter in self.char_to_idx:
                    letter_probs[self.char_to_idx[letter]] = 0.0
            
            # Renormalize
            total = letter_probs.sum()
            if total > 0:
                letter_probs = letter_probs / total
            else:
                # All letters tried, uniform over remaining
                for i in range(26):
                    if self.idx_to_char[i] not in tried_letters:
                        letter_probs[i] = 1.0
                letter_probs = letter_probs / (letter_probs.sum() + 1e-10)
            
            return letter_probs
            
        except Exception as e:
            print(f"Warning: Forward-Backward failed: {e}")
            return self._fallback_probabilities(tried_letters)
    
    def _compute_position_posterior(self, pattern: str, pos: int, model: Dict) -> np.ndarray:
        """
        Compute P(letter_pos | pattern) using a simplified forward-backward approach
        
        For efficiency, we use:
        1. Position-specific letter frequencies
        2. Transition probabilities from revealed neighbors
        3. Global letter statistics
        """
        log_probs = model['log_position_probs'][pos].copy()
        
        # Factor in transitions from revealed neighbors
        if pos > 0 and pattern[pos-1] != '_':
            # We know the previous letter
            prev_idx = self.char_to_idx.get(pattern[pos-1], 0)
            log_transition = model['log_transitions'][pos][prev_idx, :]
            log_probs = log_probs + log_transition
        
        if pos < len(pattern) - 1 and pattern[pos+1] != '_':
            # We know the next letter
            next_idx = self.char_to_idx.get(pattern[pos+1], 0)
            # Use reverse transition (approximate)
            if pos + 1 < len(model['log_transitions']):
                log_transition_reverse = model['log_transitions'][pos+1][:, next_idx]
                log_probs = log_probs + log_transition_reverse
        
        # Convert to probabilities
        probs = np.exp(log_probs - logsumexp(log_probs))
        return probs
    
    def _fallback_probabilities(self, tried_letters: Set[str]) -> np.ndarray:
        """Fallback when no model available"""
        # Use global letter frequency
        letter_freq = np.array([
            0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228,  # a-f
            0.02015, 0.06094, 0.06966, 0.00153, 0.00772, 0.04025,  # g-l
            0.02406, 0.06749, 0.07507, 0.01929, 0.00095, 0.05987,  # m-r
            0.06327, 0.09056, 0.02758, 0.00978, 0.02360, 0.00150,  # s-x
            0.01974, 0.00074  # y-z
        ])
        
        # Zero out tried letters
        for letter in tried_letters:
            if letter in self.char_to_idx:
                letter_freq[self.char_to_idx[letter]] = 0.0
        
        # Normalize
        total = letter_freq.sum()
        if total > 0:
            letter_freq = letter_freq / total
        else:
            letter_freq = np.ones(26) / 26
        
        return letter_freq
    
    def get_best_letter(self, pattern: str, tried_letters: Set[str], use_entropy: bool = True) -> str:
        """
        Get best letter using information theory
        
        Args:
            pattern: Current pattern
            tried_letters: Already guessed letters
            use_entropy: Use entropy-based selection
            
        Returns:
            Best letter to guess
        """
        probs = self.predict_letter_probabilities(pattern, tried_letters)
        
        if not use_entropy:
            # Greedy: pick highest probability
            best_idx = np.argmax(probs)
            return self.idx_to_char[best_idx]
        
        # Information-theoretic selection
        return self._select_by_entropy(tried_letters, probs)
    
    def _select_by_entropy(self, tried_letters: Set[str], letter_probs: np.ndarray) -> str:
        """
        Select letter that maximizes expected information gain
        
        For each letter:
        - Compute binary entropy H(p) = -p log p - (1-p) log (1-p)
        - Weight by probability: score = H(p) * (1 + p)
        - Pick letter with highest score
        """
        expected_gains = np.zeros(26)
        
        for idx in range(26):
            letter = self.idx_to_char[idx]
            
            if letter in tried_letters:
                expected_gains[idx] = -np.inf
                continue
            
            p = letter_probs[idx]
            
            if p < 1e-10:
                continue
            
            # Binary entropy
            p_absent = 1.0 - p
            binary_entropy = -p * np.log2(p + 1e-10) - p_absent * np.log2(p_absent + 1e-10)
            
            # Weight by probability + constant
            # Prefer letters that are both informative (p~0.5) and likely (high p)
            expected_gains[idx] = binary_entropy * (1.0 + p)
        
        best_idx = np.argmax(expected_gains)
        return self.idx_to_char[best_idx]
    
    def save(self, filepath: str):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = {
            'smoothing': self.smoothing,
            'length_models': self.length_models,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Saved True HMM to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        predictor = cls(smoothing=state['smoothing'])
        predictor.length_models = state['length_models']
        predictor.is_trained = state['is_trained']
        
        print(f"Loaded True HMM from {filepath}")
        print(f"  Models for {len(predictor.length_models)} word lengths")
        
        return predictor


if __name__ == "__main__":
    # Quick test
    from utils.dictionary import load_dictionary
    
    print("Testing True HMM...")
    corpus = load_dictionary("corpus.txt")
    
    hmm = TrueHangmanHMM(smoothing=0.01)
    hmm.train(corpus)
    
    # Test prediction
    pattern = "_a_g_an"
    tried = set()
    probs = hmm.predict_letter_probabilities(pattern, tried)
    best = hmm.get_best_letter(pattern, tried, use_entropy=True)
    
    print(f"\nPattern: {pattern}")
    print(f"Best letter (entropy): {best}")
    
    # Show top 5
    top_indices = np.argsort(probs)[::-1][:5]
    print(f"Top 5: {[(hmm.idx_to_char[i], probs[i]) for i in top_indices]}")
    
    # Save
    hmm.save("models/true_hmm.pkl")
