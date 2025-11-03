"""
Dictionary loading utilities for Hangman

Only loads from local corpus.txt and test.txt files.
"""
import os
from typing import List


def load_dictionary(path: str = "corpus.txt") -> List[str]:
    """
    Load word list from local file (corpus.txt or test.txt)
    
    Args:
        path: Path to word file (default: corpus.txt)
    
    Returns:
        List of lowercase words
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    
    # Filter alphabetic only
    words = [w for w in words if w.isalpha()]
    
    if not words:
        raise ValueError(f"No valid words found in {path}")
    
    return words





if __name__ == "__main__":
    # Test loading corpus and test files
    print("Testing dictionary loader...\n")
    
    corpus = load_dictionary("corpus.txt")
    print(f"Corpus: {len(corpus)} words")
    print(f"  Sample: {corpus[:5]}")
    
    test = load_dictionary("test.txt")
    print(f"\nTest: {len(test)} words")
    print(f"  Sample: {test[:5]}")
