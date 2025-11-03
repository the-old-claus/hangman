"""
Deep Q-Network (DQN) Training with True HMM for Hangman

Uses:
- Neural network Q-function (better generalization than Q-table)
- Experience replay buffer
- Target network for stability
- True HMM with information-theoretic selection
- Dynamic weight scheduling
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import pickle
from tqdm import tqdm
import time
import os
import json
from true_hmm_predictor import TrueHangmanHMM
from utils.dictionary import load_dictionary
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Training
    'num_episodes': 20000,
    'batch_size': 128,
    'learning_rate': 0.001,
    'discount_factor': 0.95,
    
    # Exploration
    'epsilon_start': 1.0,
    'epsilon_decay': 0.9997,
    'epsilon_min': 0.05,
    
    # DQN specific
    'replay_buffer_size': 50000,
    'target_update_freq': 500,
    'min_replay_size': 1000,
    'n_step': 3,  # N-step returns for better credit assignment
    
    # N-gram features
    'use_ngrams': True,  # Use character n-gram features
    'ngram_sizes': [2, 3],  # Bigrams and trigrams
    
    # Dynamic weight scheduling
    'hmm_weight_start': 0.9,
    'hmm_weight_end': 0.2,
    'q_weight_start': 0.1,
    'q_weight_end': 0.8,
    
    # Information theory
    'use_entropy': True,
    
    # Evaluation
    'num_test_games': 2000,
    'eval_every': 2000,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


# =============================================================================
# DEEP Q-NETWORK ARCHITECTURE
# =============================================================================

class DQN(nn.Module):
    """Deep Q-Network for Hangman"""
    
    def __init__(self, input_size, hidden_size=256, output_size=26):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# EXPERIENCE REPLAY BUFFER
# =============================================================================

class NStepReplayBuffer:
    """N-step experience replay buffer for DQN
    
    Stores transitions and computes n-step returns:
    R_t^(n) = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^{n-1}r_{t+n-1} + γ^n Q(s_{t+n}, a_{t+n})
    """
    
    def __init__(self, capacity, n_step=3, gamma=0.95):
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to n-step buffer, then push to main buffer if ready"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Only add to main buffer when we have n steps (or episode ends)
        if len(self.n_step_buffer) == self.n_step or done:
            # Compute n-step return
            n_step_return = 0
            for i, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * r
                if d:  # Episode ended
                    break
            
            # Get first state/action and last next_state/done
            first_state, first_action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            last_next_state, last_done = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
            
            # Store n-step transition
            self.buffer.append((first_state, first_action, n_step_return, last_next_state, last_done))
            
            # Clear n-step buffer if episode ended
            if done:
                self.n_step_buffer.clear()
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# HANGMAN ENVIRONMENT (same as before)
# =============================================================================

class HangmanEnvironment:
    """Hangman environment with improved reward structure"""
    
    def __init__(self, word_list, max_lives=6):
        self.word_list = word_list
        self.max_lives = max_lives
        self.reset()
    
    def reset(self, word=None):
        if word is None:
            self.current_word = np.random.choice(self.word_list).lower()
        else:
            self.current_word = word.lower()
        
        self.masked_word = '_' * len(self.current_word)
        self.guessed_letters = set()
        self.lives_remaining = self.max_lives
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        self.game_over = False
        self.won = False
        
        return self.get_state()
    
    def get_state(self):
        return {
            'masked_word': self.masked_word,
            'guessed_letters': self.guessed_letters.copy(),
            'lives_remaining': self.lives_remaining,
            'target_word': self.current_word,
            'game_over': self.game_over,
            'won': self.won
        }
    
    def step(self, letter):
        letter = letter.lower()
        
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            return self.get_state(), -20, self.game_over, {'repeated': True}
        
        self.guessed_letters.add(letter)
        
        if letter in self.current_word:
            new_masked = ''
            for i, char in enumerate(self.current_word):
                if char == letter or self.masked_word[i] != '_':
                    new_masked += char
                else:
                    new_masked += '_'
            
            revealed_count = self.current_word.count(letter)
            self.masked_word = new_masked
            
            reward = 10 * revealed_count
            progress = (len(self.current_word) - self.masked_word.count('_')) / len(self.current_word)
            reward += progress * 5
            
            if '_' not in self.masked_word:
                self.game_over = True
                self.won = True
                reward += 200
        else:
            self.lives_remaining -= 1
            self.wrong_guesses += 1
            reward = -20
            
            if self.lives_remaining <= 0:
                self.game_over = True
                self.won = False
                reward -= 100
        
        return self.get_state(), reward, self.game_over, {}


# =============================================================================
# DQN AGENT
# =============================================================================

class HangmanDQNAgent:
    """Deep Q-Network agent with True HMM guidance and N-gram features"""
    
    def __init__(self, hmm_model, config=CONFIG):
        self.hmm = hmm_model
        self.config = config
        self.device = torch.device(config['device'])
        
        # Build n-gram vocabulary from training corpus if using n-grams
        self.use_ngrams = config.get('use_ngrams', False)
        self.ngram_sizes = config.get('ngram_sizes', [2, 3])
        self.ngram_to_idx = {}
        if self.use_ngrams:
            self._build_ngram_vocab()
        
        # State representation size
        # Pattern (26 positions * max_len=24) + tried_letters (26) + HMM probs (26) + metadata (3)
        base_size = 26 * 24 + 26 + 26 + 3
        
        # Add n-gram feature size if enabled
        ngram_size = len(self.ngram_to_idx) if self.use_ngrams else 0
        self.state_size = base_size + ngram_size
        self.action_size = 26
        
        # Q-networks
        self.q_network = DQN(self.state_size, hidden_size=256, output_size=self.action_size).to(self.device)
        self.target_network = DQN(self.state_size, hidden_size=256, output_size=self.action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])
        self.criterion = nn.SmoothL1Loss()
        
        # N-step Replay buffer
        n_step = config.get('n_step', 1)
        self.replay_buffer = NStepReplayBuffer(
            config['replay_buffer_size'],
            n_step=n_step,
            gamma=config['discount_factor']
        )
        self.n_step = n_step
        
        # Training parameters
        self.gamma = config['discount_factor']
        self.epsilon = config['epsilon_start']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        
        # Dynamic weights
        self.hmm_weight_start = config['hmm_weight_start']
        self.hmm_weight_end = config['hmm_weight_end']
        self.q_weight_start = config['q_weight_start']
        self.q_weight_end = config['q_weight_end']
        self.hmm_weight = self.hmm_weight_start
        self.q_weight = self.q_weight_start
        
        # Metrics
        self.episode_rewards = []
        self.episode_wins = []
        self.episode_wrong_guesses = []
        self.training_losses = []
    
    def _build_ngram_vocab(self, max_ngrams=500):
        """Build n-gram vocabulary from HMM's training corpus"""
        from collections import Counter
        
        ngram_counts = Counter()
        
        # Extract n-grams from HMM's training words
        if hasattr(self.hmm, 'words'):
            training_words = self.hmm.words
        else:
            # Fallback: use common English n-grams
            training_words = []
        
        for word in training_words:
            word = word.lower()
            for n in self.ngram_sizes:
                for i in range(len(word) - n + 1):
                    ngram = word[i:i+n]
                    if ngram.isalpha():  # Only alphabetic n-grams
                        ngram_counts[ngram] += 1
        
        # Keep top max_ngrams most common n-grams
        top_ngrams = [ngram for ngram, _ in ngram_counts.most_common(max_ngrams)]
        self.ngram_to_idx = {ngram: idx for idx, ngram in enumerate(top_ngrams)}
        
        print(f"  Built n-gram vocabulary: {len(self.ngram_to_idx)} n-grams")
        print(f"    Sizes: {self.ngram_sizes}")
        if top_ngrams:
            print(f"    Sample: {', '.join(top_ngrams[:10])}")
    
    def state_to_features(self, state):
        """Convert game state to neural network input with N-gram features"""
        features = []
        
        # 1. Pattern encoding (one-hot for each position, padded to max_len=24)
        pattern = state['masked_word']
        max_len = 24
        for i in range(max_len):
            char_vec = np.zeros(26)
            if i < len(pattern):
                char = pattern[i]
                if char != '_':
                    char_vec[ord(char) - ord('a')] = 1
            features.extend(char_vec)
        
        # 2. Tried letters (binary vector)
        tried_vec = np.zeros(26)
        for letter in state['guessed_letters']:
            tried_vec[ord(letter) - ord('a')] = 1
        features.extend(tried_vec)
        
        # 3. HMM probabilities (information-theoretic guidance)
        hmm_probs = self.hmm.predict_letter_probabilities(
            state['masked_word'],
            state['guessed_letters']
        )
        features.extend(hmm_probs)
        
        # 4. Metadata
        features.append(state['lives_remaining'] / 6.0)  # Normalized
        features.append(pattern.count('_') / len(pattern))  # Fraction of blanks
        features.append(len(state['guessed_letters']) / 26.0)  # Fraction tried
        
        # 5. N-gram features (character sequence patterns)
        if self.use_ngrams:
            ngram_features = self._extract_ngram_features(pattern)
            features.extend(ngram_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_ngram_features(self, pattern):
        """Extract n-gram features from current pattern
        
        This captures character co-occurrence patterns like:
        - 'th', 'er', 'ing' (common bigrams/trigrams)
        - Helps predict next letter based on revealed sequences
        """
        ngram_vec = np.zeros(len(self.ngram_to_idx))
        
        # Replace underscores with wildcard for partial matching
        revealed = pattern.replace('_', '')
        
        # Extract n-grams from revealed letters
        for n in self.ngram_sizes:
            for i in range(len(pattern) - n + 1):
                ngram = pattern[i:i+n]
                
                # Only count if fully revealed (no underscores)
                if '_' not in ngram and ngram.isalpha():
                    if ngram in self.ngram_to_idx:
                        ngram_vec[self.ngram_to_idx[ngram]] = 1
        
        # Normalize by pattern length to avoid bias
        if len(pattern) > 0:
            ngram_vec = ngram_vec / len(pattern)
        
        return ngram_vec
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy with HMM guidance"""
        available = [c for c in 'abcdefghijklmnopqrstuvwxyz'
                    if c not in state['guessed_letters']]
        
        if not available:
            return None
        
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            # Explore: sample from HMM distribution if using entropy
            if self.config['use_entropy']:
                hmm_probs = self.hmm.predict_letter_probabilities(
                    state['masked_word'],
                    state['guessed_letters']
                )
                hmm_array = np.array([hmm_probs[ord(c) - ord('a')] for c in available])
                if hmm_array.sum() > 0:
                    probs = hmm_array / hmm_array.sum()
                    return np.random.choice(available, p=probs)
            return np.random.choice(available)
        
        # Exploit: weighted combination of DQN and HMM
        features = self.state_to_features(state)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(features_tensor).cpu().numpy()[0]
        
        # Get HMM probabilities
        hmm_probs = self.hmm.predict_letter_probabilities(
            state['masked_word'],
            state['guessed_letters']
        )
        
        # Combine Q-values and HMM probabilities
        combined_scores = {}
        for letter in available:
            idx = ord(letter) - ord('a')
            q_val = q_values[idx]
            hmm_prob = hmm_probs[idx]
            
            combined_scores[letter] = (
                self.q_weight * q_val +
                self.hmm_weight * (hmm_prob * 100)
            )
        
        return max(combined_scores.items(), key=lambda x: x[1])[0]
    
    def train_step(self):
        """Single training step using experience replay"""
        if len(self.replay_buffer) < self.config['min_replay_size']:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config['batch_size'])
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_batch = torch.LongTensor([ord(a) - ord('a') for a in actions]).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (using target network)
        # Note: reward_batch already contains n-step returns from buffer
        # So we discount with gamma^n instead of gamma
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * (self.gamma ** self.n_step) * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self, env, episodes=None):
        """Train DQN agent"""
        if episodes is None:
            episodes = self.config['num_episodes']
        
        print("\n" + "="*70)
        print("TRAINING DEEP Q-NETWORK WITH TRUE HMM")
        print("="*70)
        print(f"Episodes: {episodes}")
        print(f"Device: {self.device}")
        print(f"State features: {self.state_size} dimensions")
        if self.use_ngrams:
            print(f"  - N-gram features enabled ({self.ngram_sizes}-grams, {len(self.ngram_to_idx)} patterns)")
        print(f"N-step returns: {self.n_step} (better credit assignment)")
        print(f"Information-Theoretic Selection: {self.config['use_entropy']}")
        print(f"Dynamic Weights:")
        print(f"  HMM: {self.hmm_weight_start:.2f} → {self.hmm_weight_end:.2f}")
        print(f"  Q:   {self.q_weight_start:.2f} → {self.q_weight_end:.2f}")
        print(f"Network: {sum(p.numel() for p in self.q_network.parameters()):,} parameters")
        
        start_time = time.time()
        
        for episode in tqdm(range(episodes), desc="Training"):
            state = env.reset()
            episode_reward = 0
            
            while not state['game_over']:
                # Choose action
                action = self.choose_action(state, training=True)
                if action is None:
                    break
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store in replay buffer
                state_features = self.state_to_features(state)
                next_state_features = self.state_to_features(next_state)
                self.replay_buffer.push(state_features, action, reward, next_state_features, done)
                
                # Train
                loss = self.train_step()
                if loss is not None:
                    self.training_losses.append(loss)
                
                episode_reward += reward
                state = next_state
            
            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.episode_wins.append(1 if state['won'] else 0)
            self.episode_wrong_guesses.append(env.wrong_guesses)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Dynamic weight scheduling
            progress = episode / episodes
            self.hmm_weight = self.hmm_weight_start - progress * (self.hmm_weight_start - self.hmm_weight_end)
            self.q_weight = self.q_weight_start + progress * (self.q_weight_end - self.q_weight_start)
            
            # Update target network
            if (episode + 1) % self.config['target_update_freq'] == 0:
                self.update_target_network()
            
            # Periodic logging
            if (episode + 1) % self.config['eval_every'] == 0:
                recent_wins = np.mean(self.episode_wins[-100:])
                recent_wrong = np.mean(self.episode_wrong_guesses[-100:])
                recent_reward = np.mean(self.episode_rewards[-100:])
                recent_loss = np.mean(self.training_losses[-1000:]) if self.training_losses else 0
                
                print(f"\nEpisode {episode+1}/{episodes}")
                print(f"  Win Rate (last 100): {recent_wins:.2%}")
                print(f"  Avg Wrong (last 100): {recent_wrong:.2f}")
                print(f"  Avg Reward (last 100): {recent_reward:.1f}")
                print(f"  Avg Loss (last 1000): {recent_loss:.4f}")
                print(f"  Epsilon: {self.epsilon:.4f}")
                print(f"  Weights: HMM={self.hmm_weight:.2f} | Q={self.q_weight:.2f}")
        
        elapsed = time.time() - start_time
        final_win_rate = np.mean(self.episode_wins[-100:])
        
        print(f"\n✓ Training complete in {elapsed:.1f} seconds")
        print(f"  Final win rate: {final_win_rate:.2%}")
        
        return {
            'final_win_rate': final_win_rate,
            'training_time': elapsed,
            'episodes': episodes
        }
    
    def evaluate(self, env, test_words):
        """Evaluate agent on test set"""
        print("\n" + "="*70)
        print("EVALUATION ON TEST SET")
        print("="*70)
        
        self.q_network.eval()
        wins = 0
        total_wrong = 0
        total_repeated = 0
        
        start_time = time.time()
        
        for word in tqdm(test_words, desc="Evaluating"):
            state = env.reset(word)
            
            while not state['game_over']:
                action = self.choose_action(state, training=False)
                if action is None:
                    break
                state, _, _, _ = env.step(action)
            
            if state['won']:
                wins += 1
            total_wrong += env.wrong_guesses
            total_repeated += env.repeated_guesses
        
        self.q_network.train()
        elapsed = time.time() - start_time
        
        success_rate = wins / len(test_words)
        final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
        
        results = {
            'success_rate': success_rate,
            'wins': wins,
            'total_games': len(test_words),
            'total_wrong_guesses': total_wrong,
            'total_repeated_guesses': total_repeated,
            'avg_wrong_guesses': total_wrong / len(test_words),
            'avg_repeated_guesses': total_repeated / len(test_words),
            'final_score': final_score
        }
        
        print(f"\n✓ Evaluation complete in {elapsed:.1f} seconds")
        print(f"\nRESULTS:")
        print(f"  Success Rate: {results['success_rate']:.2%}")
        print(f"  Wins: {results['wins']}/{results['total_games']}")
        print(f"  Avg Wrong Guesses: {results['avg_wrong_guesses']:.2f}")
        print(f"  Avg Repeated: {results['avg_repeated_guesses']:.2f}")
        print(f"  🏆 FINAL SCORE: {results['final_score']:.2f}")
        
        if results['final_score'] > 1500:
            print("\n  🌟 EXCELLENT PERFORMANCE!")
        elif results['final_score'] > 1000:
            print("\n  ✅ GOOD PERFORMANCE")
        elif results['final_score'] > 0:
            print("\n  ⚠ ACCEPTABLE - Can improve")
        else:
            print("\n  ❌ NEEDS IMPROVEMENT")
        
        return results
    
    def save(self, filepath='models/dqn_agent.pt'):
        """Save DQN model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon
        }, filepath)
        
        print(f"Saved DQN agent to {filepath}")
    
    @classmethod
    def load(cls, filepath, hmm_model):
        """Load DQN model"""
        checkpoint = torch.load(filepath)
        agent = cls(hmm_model, checkpoint['config'])
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        print(f"Loaded DQN agent from {filepath}")
        return agent


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" DQN TRAINING WITH TRUE HMM ")
    print("="*70)
    
    # Load True HMM
    if os.path.exists('models/true_hmm.pkl'):
        print("\nLoading True HMM...")
        hmm = TrueHangmanHMM.load('models/true_hmm.pkl')
    else:
        print("\nTraining True HMM...")
        train_corpus = load_dictionary('corpus.txt')
        hmm = TrueHangmanHMM(smoothing=0.01)
        hmm.train(train_corpus)
        hmm.save('models/true_hmm.pkl')
    
    # Prepare data
    print("\n" + "="*70)
    print("PREPARING DATA")
    print("="*70)
    
    train_words = load_dictionary('corpus.txt')
    print(f"✓ Training words: {len(train_words)}")
    
    # Create environment
    env = HangmanEnvironment(train_words, max_lives=6)
    
    # Create and train DQN agent
    agent = HangmanDQNAgent(hmm, CONFIG)
    training_results = agent.train(env, episodes=CONFIG['num_episodes'])
    
    # Save
    agent.save('models/dqn_agent.pt')
    
    # Evaluate
    if os.path.exists('test.txt'):
        test_words = load_dictionary('test.txt')
        test_set = test_words[:CONFIG['num_test_games']]
        print(f"\n✓ Test words: {len(test_set)}")
        
        eval_results = agent.evaluate(env, test_set)
        
        # Save results
        all_results = {
            'training': training_results,
            'evaluation': eval_results,
            'config': CONFIG
        }
        
        with open('models/dqn_training_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "="*70)
        print(" COMPLETE ")
        print("="*70)
        print("\nFiles generated:")
        print("  ✓ models/true_hmm.pkl - True HMM model")
        print("  ✓ models/dqn_agent.pt - Trained DQN agent")
        print("  ✓ models/dqn_training_results.json - Results")
        
        return hmm, agent, all_results
    else:
        print("\n⚠ test.txt not found, skipping evaluation")
        return hmm, agent, training_results


if __name__ == "__main__":
    try:
        hmm, agent, results = main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
