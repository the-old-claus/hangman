"""
Hangman Game Environment for RL Training
"""
import random
from typing import Tuple, Dict, List


class HangmanEnv:
    """
    Hangman game environment compatible with RL training
    
    State: {
        'pattern': str,           # Current revealed pattern (e.g., "_a_g_an")
        'tried_letters': set,     # Set of guessed letters
        'tries_left': int,        # Remaining wrong guesses allowed
        'word_length': int,       # Length of the hidden word
        'done': bool,             # Whether game is finished
        'won': bool               # Whether player won
    }
    
    Action: int (0-25 representing letters a-z)
    
    Reward (aligned with scoring formula):
        Win: +100 (success rate weight)
        Loss: 0
        Wrong guess: -5 (penalty per wrong guess)
        Repeated guess: -2 (penalty per repeated guess)
        Correct guess: +1 (small reward for progress)
    """
    
    def __init__(self, dictionary: List[str], max_tries: int = 6):
        self.dictionary = dictionary
        self.max_tries = max_tries
        self.reset()
    
    def reset(self, word: str = None) -> Dict:
        """Reset environment for new game"""
        if word is None:
            self.word = random.choice(self.dictionary).lower()
        else:
            self.word = word.lower()
        
        self.pattern = '_' * len(self.word)
        self.tried_letters = set()
        self.tries_left = self.max_tries
        self.done = False
        self.won = False
        self.turn_count = 0
        
        return self.get_state()
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take action (guess a letter)
        
        Returns:
            state: Current game state
            reward: Reward for this action
            done: Whether game is finished
            info: Additional information
        """
        if self.done:
            return self.get_state(), 0.0, True, {'error': 'Game already finished'}
        
        # Convert action to letter
        if not (0 <= action <= 25):
            return self.get_state(), -0.5, False, {'error': 'Invalid action'}
        
        letter = chr(ord('a') + action)
        
        # Check if already tried (REPEATED GUESS PENALTY)
        if letter in self.tried_letters:
            return self.get_state(), -2.0, False, {'error': 'Letter already tried', 'repeated': True}
        
        self.tried_letters.add(letter)
        self.turn_count += 1
        
        # Check if letter is in word
        if letter in self.word:
            # Correct guess - reveal letters
            revealed_count = 0
            new_pattern = list(self.pattern)
            for i, char in enumerate(self.word):
                if char == letter:
                    new_pattern[i] = letter
                    revealed_count += 1
            
            self.pattern = ''.join(new_pattern)
            
            # Reward for correct guess
            reward = 1.0  # Small progress reward
            
            # Check if won
            if self.pattern == self.word:
                reward = 100.0  # WIN BONUS (aligned with scoring formula)
                self.done = True
                self.won = True
            
            info = {'correct': True, 'revealed': revealed_count, 'repeated': False}
        else:
            # Wrong guess - MAJOR PENALTY
            self.tries_left -= 1
            reward = -5.0  # Wrong guess penalty (aligned with scoring formula)
            
            # Check if lost
            if self.tries_left <= 0:
                reward = -5.0  # Just the wrong guess penalty, no extra loss penalty
                self.done = True
                self.won = False
            
            info = {'correct': False, 'revealed': 0, 'repeated': False}
        
        return self.get_state(), reward, self.done, info
    
    def get_state(self) -> Dict:
        """Get current game state"""
        return {
            'pattern': self.pattern,
            'tried_letters': self.tried_letters.copy(),
            'tries_left': self.tries_left,
            'word_length': len(self.word),
            'done': self.done,
            'won': self.won,
            'turn': self.turn_count,
            'word': self.word  # For debugging only
        }
    
    def render(self) -> str:
        """Render current game state"""
        tried_str = ''.join(sorted(self.tried_letters))
        return (f"Word: {self.pattern}\n"
                f"Tried: {tried_str}\n"
                f"Tries left: {self.tries_left}\n"
                f"Turn: {self.turn_count}")


class BatchHangmanEnv:
    """
    Vectorized Hangman environment for parallel training
    """
    
    def __init__(self, dictionary: List[str], batch_size: int = 64, max_tries: int = 6):
        self.dictionary = dictionary
        self.batch_size = batch_size
        self.max_tries = max_tries
        self.envs = [HangmanEnv(dictionary, max_tries) for _ in range(batch_size)]
    
    def reset(self) -> List[Dict]:
        """Reset all environments"""
        return [env.reset() for env in self.envs]
    
    def step(self, actions: List[int]) -> Tuple[List[Dict], List[float], List[bool], List[Dict]]:
        """Take parallel actions"""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        states, rewards, dones, infos = zip(*results)
        return list(states), list(rewards), list(dones), list(infos)
    
    def reset_done(self) -> int:
        """Reset finished games and return number of wins"""
        wins = 0
        for env in self.envs:
            if env.done:
                if env.won:
                    wins += 1
                env.reset()
        return wins


if __name__ == "__main__":
    # Test the environment
    test_words = ["python", "hangman", "learning", "algorithm", "computer"]
    env = HangmanEnv(test_words)
    
    state = env.reset(word="python")
    print("Initial state:")
    print(env.render())
    print()
    
    # Test some guesses
    test_actions = [
        ('p', 15),  # p
        ('y', 24),  # y
        ('z', 25),  # z (wrong)
        ('t', 19),  # t
        ('h', 7),   # h
        ('o', 14),  # o
        ('n', 13),  # n
    ]
    
    for letter, action in test_actions:
        state, reward, done, info = env.step(action)
        print(f"Guessed '{letter}': reward={reward:.2f}, info={info}")
        print(env.render())
        print()
        
        if done:
            print(f"Game finished! Won: {state['won']}")
            break
