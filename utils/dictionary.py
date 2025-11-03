"""
Dictionary loading and management utilities
"""
import os
import urllib.request
from typing import List, Dict


def load_dictionary(path: str = "data/words.txt", min_length: int = None, max_length: int = None) -> List[str]:
    """
    Load dictionary from file
    
    Args:
        path: Path to dictionary file
        min_length: Minimum word length to include (None = auto-detect from corpus)
        max_length: Maximum word length to include (None = auto-detect from corpus)
    
    Returns:
        List of words
    """
    if not os.path.exists(path):
        print(f"Dictionary not found at {path}")
        print("Attempting to download...")
        download_dictionary(path)
    
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    
    # Filter alphabetic only
    words = [w for w in words if w.isalpha()]
    
    if not words:
        print("Warning: No valid words found in dictionary!")
        return []
    
    # Auto-detect min/max lengths if not specified
    if min_length is None or max_length is None:
        lengths = [len(w) for w in words]
        detected_min = min(lengths)
        detected_max = max(lengths)
        
        if min_length is None:
            min_length = detected_min
        if max_length is None:
            max_length = detected_max
        
        print(f"Auto-detected word length range: {detected_min}-{detected_max}")
    
    # Filter by length
    words = [w for w in words if min_length <= len(w) <= max_length]
    
    print(f"Loaded {len(words)} words (length {min_length}-{max_length})")
    return words


def download_dictionary(save_path: str = "data/words.txt"):
    """
    Download a dictionary file from the internet
    """
    # Use NLTK words corpus
    try:
        import nltk
        nltk.download('words', quiet=True)
        from nltk.corpus import words as nltk_words
        
        word_list = nltk_words.words()
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to file
        with open(save_path, 'w', encoding='utf-8') as f:
            for word in word_list:
                f.write(word.lower() + '\n')
        
        print(f"Downloaded {len(word_list)} words to {save_path}")
    
    except Exception as e:
        print(f"Failed to download dictionary: {e}")
        print("Creating minimal fallback dictionary...")
        
        # Minimal fallback word list
        fallback_words = [
            "python", "hangman", "computer", "learning", "algorithm",
            "machine", "neural", "network", "training", "model",
            "data", "science", "programming", "software", "developer",
            "function", "variable", "class", "object", "method",
            "array", "string", "integer", "boolean", "dictionary"
        ]
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            for word in fallback_words:
                f.write(word + '\n')
        
        print(f"Created fallback dictionary with {len(fallback_words)} words")


def get_word_buckets(words: List[str]) -> Dict[int, List[str]]:
    """
    Organize words by length into buckets
    
    Args:
        words: List of words
    
    Returns:
        Dictionary mapping length -> list of words
    """
    buckets = {}
    for word in words:
        length = len(word)
        if length not in buckets:
            buckets[length] = []
        buckets[length].append(word)
    
    return buckets


def filter_candidates(pattern: str, wrong_letters: set, dictionary: List[str]) -> List[str]:
    """
    Filter dictionary to words matching current pattern
    
    Args:
        pattern: Current pattern (e.g., "_a_g_an")
        wrong_letters: Set of letters not in the word
        dictionary: Full word list
    
    Returns:
        List of candidate words
    """
    candidates = []
    word_length = len(pattern)
    
    for word in dictionary:
        if len(word) != word_length:
            continue
        
        # Check if word matches pattern
        match = True
        for i, (p, w) in enumerate(zip(pattern, word)):
            if p != '_' and p != w:
                match = False
                break
        
        if not match:
            continue
        
        # Check if word contains wrong letters
        if any(letter in word for letter in wrong_letters):
            continue
        
        candidates.append(word)
    
    return candidates


if __name__ == "__main__":
    # Test dictionary loading
    words = load_dictionary()
    print(f"\nTotal words: {len(words)}")
    
    buckets = get_word_buckets(words)
    print(f"\nWords by length:")
    for length in sorted(buckets.keys()):
        print(f"  Length {length}: {len(buckets[length])} words")
    
    # Test filtering
    pattern = "_a_g_an"
    wrong = {'b', 'c', 'd'}
    candidates = filter_candidates(pattern, wrong, words)
    print(f"\nCandidates for pattern '{pattern}': {len(candidates)}")
    print(f"Examples: {candidates[:10]}")
