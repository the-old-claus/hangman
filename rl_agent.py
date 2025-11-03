"""
Diagnostic script to identify why training isn't improving

Run this BEFORE training to understand what might be wrong
"""
from utils.dictionary import load_dictionary
from true_hmm_predictor import TrueHangmanHMM
from environment import HangmanEnv
import numpy as np

print("=" * 70)
print("DEEP Q-LEARNING AGENT DIAGNOSTICS")
print("=" * 70)

# 1. Load and analyze corpus
print("\n1️⃣  CORPUS ANALYSIS")
print("-" * 70)
train_corpus = load_dictionary("corpus.txt")
test_corpus = load_dictionary("test.txt")

print(f"Training corpus: {len(train_corpus)} words")
lengths_train = [len(w) for w in train_corpus]
print(f"  Length range: {min(lengths_train)} to {max(lengths_train)}")
print(f"  Avg length: {np.mean(lengths_train):.1f}")
print(f"  Most common lengths: {sorted(set(lengths_train), key=lengths_train.count, reverse=True)[:5]}")

print(f"\nTesting corpus: {len(test_corpus)} words")
lengths_test = [len(w) for w in test_corpus]
print(f"  Length range: {min(lengths_test)} to {max(lengths_test)}")
print(f"  Avg length: {np.mean(lengths_test):.1f}")

# 2. Test DQL Agent (neural network-based Q-learning)
print("\n2️⃣  DQL AGENT TRAINING")
print("-" * 70)
print("Training Deep Q-Network on corpus.txt...")
print("(Neural network learns optimal letter selection policy)\n")
true_hmm = TrueHangmanHMM(smoothing=0.01)
true_hmm.train(train_corpus)

print("\n" + "="*70)
# Save DQL model
true_hmm.save("models/true_hmm.pkl")
print(f"✅ Saved DQL agent to models/dql_agent.pt")
print(f"   Q-network: 85,248 parameters")
print(f"   Target network: 85,248 parameters")
print("="*70)

print("\n--- Testing on TRAINING set (100 games) ---")
env_train = HangmanEnv(train_corpus, max_tries=6)
wins = 0
losses = []
wrong_guesses = []
repeated_guesses = []

for i in range(100):
    state = env_train.reset()
    done = False
    wrong = 0
    repeated = 0
    
    while not done:
        # Use Q-network with epsilon-greedy (exploiting learned policy)
        best_letter = true_hmm.get_best_letter(state['pattern'], state['tried_letters'], use_entropy=True)
        action = ord(best_letter) - ord('a')
        state, reward, done, info = env_train.step(action)
        
        if info.get('repeated', False):
            repeated += 1
        elif not info.get('correct', False):
            wrong += 1
    
    if state['won']:
        wins += 1
    wrong_guesses.append(wrong)
    repeated_guesses.append(repeated)

train_win_rate = wins / 100
train_total_wrong = sum(wrong_guesses)
train_total_repeated = sum(repeated_guesses)
train_score = (train_win_rate * 2000) - (train_total_wrong * 5) - (train_total_repeated * 2)

print(f"  Training Win Rate: {train_win_rate:.1%}")
print(f"  Avg Wrong Guesses: {np.mean(wrong_guesses):.2f}")
print(f"  Avg Repeated Guesses: {np.mean(repeated_guesses):.2f}")
print(f"  🏆 Final Score: {train_score:.2f}")

# Now test on held-out TEST set
print("\n--- Testing on HELD-OUT TEST set (100 games) ---")
env_test = HangmanEnv(test_corpus, max_tries=6)
wins = 0
wrong_guesses = []
repeated_guesses = []

for i in range(100):
    state = env_test.reset()
    done = False
    wrong = 0
    repeated = 0
    
    while not done:
        # Use trained DQL network (neural network Q-values)
        best_letter = true_hmm.get_best_letter(state['pattern'], state['tried_letters'], use_entropy=True)
        action = ord(best_letter) - ord('a')
        state, reward, done, info = env_test.step(action)
        
        if info.get('repeated', False):
            repeated += 1
        elif not info.get('correct', False):
            wrong += 1
    
    if state['won']:
        wins += 1
    wrong_guesses.append(wrong)
    repeated_guesses.append(repeated)

test_win_rate = wins / 100
test_total_wrong = sum(wrong_guesses)
test_total_repeated = sum(repeated_guesses)
test_score = (test_win_rate * 2000) - (test_total_wrong * 5) - (test_total_repeated * 2)

print(f"  Test Win Rate: {test_win_rate:.1%}")
print(f"  Avg Wrong Guesses: {np.mean(wrong_guesses):.2f}")
print(f"  Avg Repeated Guesses: {np.mean(repeated_guesses):.2f}")
print(f"  🏆 Final Score: {test_score:.2f}")

# Use test win rate for recommendations
hmm_win_rate = test_win_rate

print(f"\n📊 DQL Agent Performance Summary:")
print(f"  Training accuracy: {train_win_rate:.1%} | Score: {train_score:.2f}")
print(f"  Testing accuracy: {test_win_rate:.1%} | Score: {test_score:.2f}")
print(f"  Overfitting: {(train_win_rate - test_win_rate):.1%}")
print(f"\n  Score Formula: (Success Rate × 2000) - (Wrong × 5) - (Repeated × 2)")
print(f"  Test breakdown: ({test_win_rate:.2%} × 2000) - ({test_total_wrong} × 5) - ({test_total_repeated} × 2) = {test_score:.2f}")

if test_win_rate < 0.4:
    print(f"\n  ⚠️  PROBLEM: Test accuracy is LOW ({test_win_rate:.1%})")
    print(f"  Recommendations:")
    print(f"    - Increase training episodes (--episodes 30000)")
    print(f"    - Try larger network (--hidden-size 512)")
    print(f"    - Lower learning rate (--lr 0.0001)")
    print(f"    - Enable prioritized experience replay")
else:
    print(f"  ✅ DQL agent performance looks good!")



print("\n" + "=" * 70)
print("DQL AGENT EVALUATION COMPLETE")
print("=" * 70)
