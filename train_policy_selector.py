"""
Policy Selector Network for Hangman

Meta-learning approach:
- HMM provides probability distribution over letters (expert knowledge)
- Q-network learns WHICH selection strategy to apply to HMM's output
- Actions are selection strategies (top-1, top-2, entropy-based, etc.)
- Much smaller action space (5-8 strategies vs 26 letters)
- Leverages HMM's 46% baseline while learning when to deviate
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
import time
import os
import json
from true_hmm_predictor import TrueHangmanHMM
from utils.dictionary import load_dictionary

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Training
    'num_episodes': 25000,  # More episodes for convergence
    'batch_size': 128,  # Larger batch for stability
    'learning_rate': 0.0003,  # Lower LR for fine-tuning
    'discount_factor': 0.97,  # Higher gamma for long-term planning
    
    # Exploration
    'epsilon_start': 1.0,
    'epsilon_decay': 0.9998,  # Slower decay
    'epsilon_min': 0.05,  # Lower min for more exploitation
    
    # Network
    'replay_buffer_size': 50000,  # Larger buffer
    'target_update_freq': 500,  # Less frequent updates for stability
    'min_replay_size': 1000,  # More samples before training
    'hidden_size': 256,  # Larger network
    
    # N-gram features
    'use_ngrams': True,  # Character sequence patterns
    'ngram_sizes': [2, 3],  # Bigrams and trigrams
    'max_ngrams': 300,  # Vocabulary size
    
    # Checkpointing
    'checkpoint_best': True,
    'eval_during_training': True,
    'quick_eval_size': 200,
    
    # Evaluation
    'num_test_games': 2000,
    'eval_every': 1000,  # More frequent evaluation
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# =============================================================================
# SELECTION STRATEGIES
# =============================================================================

class SelectionStrategy:
    """Different ways to select a letter from HMM's probability distribution"""
    
    @staticmethod
    def top_k(hmm_probs, available_letters, k=1):
        """Pick k-th most probable letter from HMM"""
        sorted_indices = np.argsort(hmm_probs)[::-1]
        for idx in sorted_indices:
            letter = chr(ord('a') + idx)
            if letter in available_letters:
                k -= 1
                if k == 0:
                    return letter
        return available_letters[0] if available_letters else None
    
    @staticmethod
    def entropy_based(hmm_probs, available_letters):
        """Information-theoretic selection (maximize expected info gain)"""
        scores = np.zeros(26)
        for idx in range(26):
            if chr(ord('a') + idx) not in available_letters:
                scores[idx] = -np.inf
                continue
            
            p = hmm_probs[idx]
            if p < 1e-10:
                continue
            
            # Binary entropy weighted by probability
            p_absent = 1.0 - p
            binary_entropy = -p * np.log2(p + 1e-10) - p_absent * np.log2(p_absent + 1e-10)
            scores[idx] = binary_entropy * (1.0 + p)
        
        best_idx = np.argmax(scores)
        return chr(ord('a') + best_idx)
    
    @staticmethod
    def weighted_random(hmm_probs, available_letters, temperature=1.0):
        """Sample from HMM distribution with temperature"""
        available_indices = [ord(c) - ord('a') for c in available_letters]
        available_probs = hmm_probs[available_indices]
        
        if available_probs.sum() < 1e-10:
            return random.choice(available_letters)
        
        # Apply temperature
        if temperature != 1.0:
            available_probs = np.power(available_probs, 1.0/temperature)
        
        # Normalize
        available_probs = available_probs / available_probs.sum()
        
        chosen_idx = np.random.choice(len(available_letters), p=available_probs)
        return available_letters[chosen_idx]
    
    @staticmethod
    def top_k_blend(hmm_probs, available_letters, k=3):
        """Average scores of top-k and pick best"""
        sorted_indices = np.argsort(hmm_probs)[::-1]
        
        # Get top-k available letters
        top_k_letters = []
        for idx in sorted_indices:
            letter = chr(ord('a') + idx)
            if letter in available_letters:
                top_k_letters.append(letter)
                if len(top_k_letters) >= k:
                    break
        
        if not top_k_letters:
            return available_letters[0] if available_letters else None
        
        # Pick first (most probable) from top-k
        return top_k_letters[0]


# Strategy index mapping
STRATEGIES = {
    0: lambda probs, avail: SelectionStrategy.top_k(probs, avail, k=1),  # Greedy
    1: lambda probs, avail: SelectionStrategy.top_k(probs, avail, k=2),  # 2nd best
    2: lambda probs, avail: SelectionStrategy.top_k(probs, avail, k=3),  # 3rd best
    3: lambda probs, avail: SelectionStrategy.entropy_based(probs, avail),  # Info theory
    4: lambda probs, avail: SelectionStrategy.weighted_random(probs, avail, temperature=0.5),  # Low temp
    5: lambda probs, avail: SelectionStrategy.weighted_random(probs, avail, temperature=1.0),  # Medium temp
    6: lambda probs, avail: SelectionStrategy.weighted_random(probs, avail, temperature=2.0),  # High temp
    7: lambda probs, avail: SelectionStrategy.top_k_blend(probs, avail, k=3),  # Top-3 blend
}

NUM_STRATEGIES = len(STRATEGIES)

# =============================================================================
# POLICY SELECTOR NETWORK
# =============================================================================

class DuelingPolicySelectorNetwork(nn.Module):
    """
    Dueling Network Architecture for Policy Selection
    
    Separates:
    - Value stream: V(s) - how good is this state?
    - Advantage stream: A(s,a) - how much better is each strategy?
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    
    This helps learn which states are good independently of which action to take.
    """
    
    def __init__(self, state_size, num_strategies=NUM_STRATEGIES, hidden_size=256):
        super(DuelingPolicySelectorNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )
        
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_strategies)
        )
    
    def forward(self, x):
        # Extract features
        features = self.feature_layer(x)
        
        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine using dueling architecture
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


# =============================================================================
# ENVIRONMENT
# =============================================================================

class HangmanEnvironment:
    """Hangman environment"""
    
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
# POLICY SELECTOR AGENT
# =============================================================================

class PolicySelectorAgent:
    """Agent that learns to select the best strategy for HMM outputs"""
    
    def __init__(self, hmm_model, config=CONFIG):
        self.hmm = hmm_model
        self.config = config
        self.device = torch.device(config['device'])
        
        # Build N-gram vocabulary if enabled
        self.use_ngrams = config.get('use_ngrams', False)
        self.ngram_sizes = config.get('ngram_sizes', [2, 3])
        self.ngram_to_idx = {}
        if self.use_ngrams:
            self._build_ngram_vocab(config.get('max_ngrams', 300))
        
        # State: HMM probs (26) + info-theoretic (4) + game metadata (6) + n-grams
        base_size = 26 + 4 + 6
        ngram_size = len(self.ngram_to_idx) if self.use_ngrams else 0
        self.state_size = base_size + ngram_size
        self.num_strategies = NUM_STRATEGIES
        
        # Dueling Networks
        hidden_size = config.get('hidden_size', 256)
        self.q_network = DuelingPolicySelectorNetwork(self.state_size, self.num_strategies, hidden_size).to(self.device)
        self.target_network = DuelingPolicySelectorNetwork(self.state_size, self.num_strategies, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])
        self.criterion = nn.SmoothL1Loss()
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=config['replay_buffer_size'])
        
        # Training params
        self.gamma = config['discount_factor']
        self.epsilon = config['epsilon_start']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        
        # Metrics
        self.episode_rewards = []
        self.episode_wins = []
        self.episode_wrong_guesses = []
        self.training_losses = []
        self.strategy_usage = np.zeros(NUM_STRATEGIES)
        
        # Best model tracking
        self.best_eval_score = -float('inf')
        self.best_model_state = None
        self.best_episode = 0
    
    def _build_ngram_vocab(self, max_ngrams=300):
        """Build n-gram vocabulary from HMM's training corpus"""
        from collections import Counter
        
        ngram_counts = Counter()
        
        # Extract n-grams from HMM's word lists
        training_words = []
        if hasattr(self.hmm, 'words'):
            training_words = self.hmm.words
        elif hasattr(self.hmm, 'word_dict'):
            for word_list in self.hmm.word_dict.values():
                training_words.extend(word_list)
        
        for word in training_words:
            word = word.lower()
            for n in self.ngram_sizes:
                for i in range(len(word) - n + 1):
                    ngram = word[i:i+n]
                    if ngram.isalpha():
                        ngram_counts[ngram] += 1
        
        # Keep top N most common n-grams
        top_ngrams = [ngram for ngram, _ in ngram_counts.most_common(max_ngrams)]
        self.ngram_to_idx = {ngram: idx for idx, ngram in enumerate(top_ngrams)}
        
        print(f"\n  N-gram vocabulary: {len(self.ngram_to_idx)} patterns")
        if top_ngrams:
            print(f"    Sample: {', '.join(top_ngrams[:15])}")
    
    def state_to_features(self, game_state):
        """Convert game state + HMM probs to network input with information-theoretic features"""
        features = []
        
        # 1. HMM probability distribution (26 values)
        hmm_probs = self.hmm.predict_letter_probabilities(
            game_state['masked_word'],
            game_state['guessed_letters']
        )
        features.extend(hmm_probs)
        
        # 2. Information-theoretic metrics
        # Entropy of HMM distribution (uncertainty measure)
        entropy = -np.sum(hmm_probs * np.log(hmm_probs + 1e-10))
        features.append(entropy / np.log(26))  # Normalized entropy
        
        # Max probability (confidence of best guess)
        features.append(np.max(hmm_probs))
        
        # Top-3 probability mass
        top3_mass = np.sum(np.sort(hmm_probs)[-3:])
        features.append(top3_mass)
        
        # Gini coefficient (concentration of probability)
        sorted_probs = np.sort(hmm_probs)
        n = len(sorted_probs)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_probs)) / (n * np.sum(sorted_probs)) - (n + 1) / n
        features.append(gini)
        
        # 3. Game state metadata
        features.append(game_state['lives_remaining'] / 6.0)  # Normalized lives
        features.append(game_state['masked_word'].count('_') / len(game_state['masked_word']))  # Fraction blanks
        features.append(len(game_state['guessed_letters']) / 26.0)  # Fraction tried
        features.append(len(game_state['masked_word']) / 24.0)  # Normalized word length
        
        # 4. Pattern features
        # Number of distinct revealed letters
        revealed = set(c for c in game_state['masked_word'] if c != '_')
        features.append(len(revealed) / 26.0)
        
        # Ratio of revealed to total length
        revealed_count = len(game_state['masked_word']) - game_state['masked_word'].count('_')
        features.append(revealed_count / len(game_state['masked_word']))
        
        # 5. N-gram features (character sequence patterns)
        if self.use_ngrams:
            ngram_features = self._extract_ngram_features(game_state['masked_word'])
            features.extend(ngram_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_ngram_features(self, pattern):
        """Extract n-gram features from current pattern
        
        Captures character sequences like 'th', 'ing', 'tion' to help
        the network understand which selection strategy works best
        for different word patterns.
        """
        ngram_vec = np.zeros(len(self.ngram_to_idx))
        
        # Extract n-grams from revealed letters only
        for n in self.ngram_sizes:
            for i in range(len(pattern) - n + 1):
                ngram = pattern[i:i+n]
                
                # Only count fully revealed n-grams (no underscores)
                if '_' not in ngram and ngram.isalpha():
                    if ngram in self.ngram_to_idx:
                        ngram_vec[self.ngram_to_idx[ngram]] = 1
        
        # Normalize by pattern length
        if len(pattern) > 0:
            ngram_vec = ngram_vec / len(pattern)
        
        return ngram_vec
    
    def choose_action(self, game_state, training=True):
        """Choose a strategy, then apply it to get a letter"""
        available_letters = [c for c in 'abcdefghijklmnopqrstuvwxyz'
                            if c not in game_state['guessed_letters']]
        
        if not available_letters:
            return None, None
        
        # Get HMM probabilities
        hmm_probs = self.hmm.predict_letter_probabilities(
            game_state['masked_word'],
            game_state['guessed_letters']
        )
        
        # Epsilon-greedy strategy selection
        if training and np.random.random() < self.epsilon:
            strategy_idx = np.random.randint(0, self.num_strategies)
        else:
            # Use Q-network to select strategy
            features = self.state_to_features(game_state)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(features_tensor).cpu().numpy()[0]
            
            strategy_idx = np.argmax(q_values)
        
        # Apply selected strategy to get letter
        letter = STRATEGIES[strategy_idx](hmm_probs, available_letters)
        
        # Track usage
        if not training:
            self.strategy_usage[strategy_idx] += 1
        
        return letter, strategy_idx
    
    def train_step(self):
        """Single training step"""
        if len(self.replay_buffer) < self.config['min_replay_size']:
            return None
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.config['batch_size'])
        states, strategies, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        strategy_batch = torch.LongTensor(strategies).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(state_batch).gather(1, strategy_batch.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def quick_eval(self, env, num_games=200):
        """Quick evaluation during training"""
        self.q_network.eval()
        self.strategy_usage = np.zeros(NUM_STRATEGIES)
        
        wins = 0
        total_wrong = 0
        
        for _ in range(num_games):
            state = env.reset()
            while not state['game_over']:
                letter, _ = self.choose_action(state, training=False)
                if letter is None:
                    break
                state, _, _, _ = env.step(letter)
            
            if state['won']:
                wins += 1
            total_wrong += env.wrong_guesses
        
        self.q_network.train()
        
        success_rate = wins / num_games
        avg_wrong = total_wrong / num_games
        score = (success_rate * 1000) - (avg_wrong * 5)
        
        return {
            'success_rate': success_rate,
            'avg_wrong': avg_wrong,
            'score': score,
            'wins': wins
        }
    
    def train(self, env, episodes=None):
        """Train the policy selector with evaluation and checkpointing"""
        if episodes is None:
            episodes = self.config['num_episodes']
        
        print("\n" + "="*70)
        print("TRAINING DUELING POLICY SELECTOR NETWORK")
        print("="*70)
        print(f"Episodes: {episodes}")
        print(f"Device: {self.device}")
        print(f"Architecture: Dueling DQN")
        print(f"Number of strategies: {self.num_strategies}")
        print(f"State features: {self.state_size} dimensions")
        if self.use_ngrams:
            print(f"  - N-gram features: {len(self.ngram_to_idx)} patterns ({self.ngram_sizes}-grams)")
        print(f"  - Info-theoretic metrics: entropy, confidence, concentration")
        print(f"HMM baseline: 46% (target to beat)")
        print(f"Network: {sum(p.numel() for p in self.q_network.parameters()):,} parameters")
        
        start_time = time.time()
        
        for episode in tqdm(range(episodes), desc="Training"):
            state = env.reset()
            episode_reward = 0
            
            while not state['game_over']:
                # Choose strategy and get letter
                letter, strategy_idx = self.choose_action(state, training=True)
                
                if letter is None:
                    break
                
                # Take action
                next_state, reward, done, info = env.step(letter)
                
                # Store transition
                state_features = self.state_to_features(state)
                next_state_features = self.state_to_features(next_state)
                self.replay_buffer.append((state_features, strategy_idx, reward, next_state_features, done))
                
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
            
            # Update target network
            if (episode + 1) % self.config['target_update_freq'] == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Periodic evaluation with checkpointing
            if (episode + 1) % self.config['eval_every'] == 0:
                recent_wins = np.mean(self.episode_wins[-100:])
                recent_wrong = np.mean(self.episode_wrong_guesses[-100:])
                recent_reward = np.mean(self.episode_rewards[-100:])
                recent_loss = np.mean(self.training_losses[-1000:]) if self.training_losses else 0
                
                print(f"\nEpisode {episode+1}/{episodes}")
                print(f"  Win Rate (last 100): {recent_wins:.2%}")
                print(f"  Avg Wrong (last 100): {recent_wrong:.2f}")
                print(f"  Avg Reward: {recent_reward:.1f}")
                print(f"  Avg Loss: {recent_loss:.4f}")
                print(f"  Epsilon: {self.epsilon:.4f}")
                
                # Quick evaluation if enabled
                if self.config.get('eval_during_training', False):
                    eval_size = self.config.get('quick_eval_size', 200)
                    eval_results = self.quick_eval(env, num_games=eval_size)
                    print(f"  [EVAL on {eval_size}] Success: {eval_results['success_rate']:.2%} | Avg Wrong: {eval_results['avg_wrong']:.2f} | Score: {eval_results['score']:.1f}")
                    
                    # Save best model
                    if self.config.get('checkpoint_best', False) and eval_results['score'] > self.best_eval_score:
                        self.best_eval_score = eval_results['score']
                        self.best_model_state = {
                            'q_network': self.q_network.state_dict(),
                            'target_network': self.target_network.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                        self.best_episode = episode + 1
                        print(f"  ✓ New best model! Score: {self.best_eval_score:.1f} (saved)")
        
        elapsed = time.time() - start_time
        final_win_rate = np.mean(self.episode_wins[-100:])
        
        print(f"\n✓ Training complete in {elapsed:.1f} seconds")
        print(f"  Final win rate: {final_win_rate:.2%}")
        
        # Restore best model if available
        if self.best_model_state is not None:
            print(f"\n  Loading best model from episode {self.best_episode}")
            print(f"  Best score: {self.best_eval_score:.1f}")
            self.q_network.load_state_dict(self.best_model_state['q_network'])
            self.target_network.load_state_dict(self.best_model_state['target_network'])
            self.optimizer.load_state_dict(self.best_model_state['optimizer'])
        
        return {
            'final_win_rate': final_win_rate,
            'best_eval_score': self.best_eval_score,
            'best_episode': self.best_episode,
            'training_time': elapsed,
            'episodes': episodes
        }
    
    def evaluate(self, env, test_words):
        """Evaluate agent"""
        print("\n" + "="*70)
        print("EVALUATION ON TEST SET")
        print("="*70)
        
        self.q_network.eval()
        self.strategy_usage = np.zeros(NUM_STRATEGIES)
        
        wins = 0
        total_wrong = 0
        total_repeated = 0
        
        start_time = time.time()
        
        for word in tqdm(test_words, desc="Evaluating"):
            state = env.reset(word)
            
            while not state['game_over']:
                letter, _ = self.choose_action(state, training=False)
                if letter is None:
                    break
                state, _, _, _ = env.step(letter)
            
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
            'avg_wrong': total_wrong / len(test_words),
            'avg_repeated': total_repeated / len(test_words),
            'final_score': final_score,
            'strategy_usage': self.strategy_usage.tolist()
        }
        
        print(f"\n✓ Evaluation complete in {elapsed:.1f} seconds")
        print(f"\nRESULTS:")
        print(f"  Success Rate: {results['success_rate']:.2%}")
        print(f"  Wins: {results['wins']}/{results['total_games']}")
        print(f"  Avg Wrong: {results['avg_wrong']:.2f}")
        print(f"  Avg Repeated: {results['avg_repeated']:.2f}")
        print(f"  🏆 FINAL SCORE: {results['final_score']:.2f}")
        
        # Strategy usage analysis
        print(f"\nStrategy Usage:")
        strategy_names = ["Top-1", "Top-2", "Top-3", "Entropy", "Temp-0.5", "Temp-1.0", "Temp-2.0", "Top-3-Blend"]
        total_uses = self.strategy_usage.sum()
        for i, (name, count) in enumerate(zip(strategy_names, self.strategy_usage)):
            pct = (count / total_uses * 100) if total_uses > 0 else 0
            print(f"  {name:>12}: {count:>6.0f} ({pct:>5.1f}%)")
        
        if results['final_score'] > 1500:
            print("\n  🌟 EXCELLENT!")
        elif results['final_score'] > 1000:
            print("\n  ✅ GOOD")
        elif results['final_score'] > 0:
            print("\n  ⚠ ACCEPTABLE")
        else:
            print("\n  ❌ NEEDS IMPROVEMENT")
        
        return results
    
    def save(self, filepath='models/policy_selector.pt'):
        """Save model (saves best model if available)"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Use best model if available
        if self.best_model_state is not None:
            state_dict = self.best_model_state
        else:
            state_dict = {
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        
        torch.save({
            'q_network_state_dict': state_dict['q_network'],
            'target_network_state_dict': state_dict['target_network'],
            'optimizer_state_dict': state_dict['optimizer'],
            'config': self.config,
            'epsilon': self.epsilon,
            'best_eval_score': self.best_eval_score,
            'best_episode': self.best_episode
        }, filepath)
        
        score_msg = f" (best score: {self.best_eval_score:.1f} @ episode {self.best_episode})" if self.best_model_state else ""
        print(f"Saved policy selector to {filepath}{score_msg}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" POLICY SELECTOR TRAINING ")
    print("="*70)
    
    # Load HMM
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
    
    # Create and train agent
    agent = PolicySelectorAgent(hmm, CONFIG)
    training_results = agent.train(env, episodes=CONFIG['num_episodes'])
    
    # Save
    agent.save('models/policy_selector.pt')
    
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
        
        with open('models/policy_selector_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "="*70)
        print(" COMPLETE ")
        print("="*70)
        print("\nFiles generated:")
        print("  ✓ models/true_hmm.pkl")
        print("  ✓ models/policy_selector.pt")
        print("  ✓ models/policy_selector_results.json")
        
        return hmm, agent, all_results
    else:
        print("\n⚠ test.txt not found")
        return hmm, agent, training_results


if __name__ == "__main__":
    try:
        hmm, agent, results = main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
