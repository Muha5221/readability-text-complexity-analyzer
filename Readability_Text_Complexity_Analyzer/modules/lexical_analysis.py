"""
Lexical Complexity Analysis Module for Readability and Text Complexity Analyzer.
Covers Module 1 and Module 2 concepts:
- Average word length
- Lexical Diversity (Type-Token Ratio: TTR)
- Word Frequency Distribution using NLTK FreqDist
- Vocabulary comparison against NLTK Brown reference corpus
- Identification of less frequent and polysyllabic/long words
"""

import functools
from typing import Dict, List, Set, Tuple, Any
import nltk
from nltk.probability import FreqDist

# Cached Brown corpus vocabulary set to avoid disk I/O on every request
_BROWN_VOCAB_CACHE: Set[str] = set()
_BROWN_FREQ_CACHE: Dict[str, int] = {}
_CORPUS_LOAD_ATTEMPTED = False
_CORPUS_AVAILABLE = False


def _load_reference_corpus() -> Tuple[bool, str]:
    """
    Attempt to load the NLTK Brown corpus as standard reference vocabulary.
    Includes a graceful fallback if the corpus is unavailable.
    """
    global _BROWN_VOCAB_CACHE, _BROWN_FREQ_CACHE, _CORPUS_LOAD_ATTEMPTED, _CORPUS_AVAILABLE
    
    if _CORPUS_LOAD_ATTEMPTED:
        return _CORPUS_AVAILABLE, "Corpus status already determined."

    _CORPUS_LOAD_ATTEMPTED = True

    try:
        from nltk.corpus import brown
        words = brown.words()
        # Initialize FreqDist from Brown words
        fdist = FreqDist(w.lower() for w in words if w.isalpha())
        _BROWN_VOCAB_CACHE = set(fdist.keys())
        _BROWN_FREQ_CACHE = dict(fdist)
        _CORPUS_AVAILABLE = True
        return True, "NLTK Brown corpus loaded successfully (50,000+ reference vocabulary items)."
    except Exception as e:
        # Graceful fallback: common English words list
        _CORPUS_AVAILABLE = False
        _BROWN_VOCAB_CACHE = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
            "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
            "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
        }
        return False, f"NLTK Brown corpus unavailable ({e}). Fallback vocabulary applied."


def analyze_lexical_complexity(words: List[str]) -> Dict[str, Any]:
    """
    Perform comprehensive lexical complexity analysis on cleaned words.

    Args:
        words: List of lowercased alphabetic word tokens.

    Returns:
        Dictionary containing lexical metrics, frequency distribution,
        rare words, and long words.
    """
    total_words = len(words)
    if total_words == 0:
        return {
            "total_words": 0,
            "unique_word_count": 0,
            "lexical_diversity": 0.0,
            "avg_word_length": 0.0,
            "freq_dist_table": [],
            "long_words": [],
            "long_word_count": 0,
            "long_word_percentage": 0.0,
            "less_frequent_words": [],
            "less_frequent_count": 0,
            "less_frequent_percentage": 0.0,
            "corpus_status": "No words provided",
            "corpus_available": False
        }

    # 1. Unique word count & Lexical Diversity (Type-Token Ratio)
    unique_words = sorted(list(set(words)))
    unique_word_count = len(unique_words)
    lexical_diversity = round(unique_word_count / total_words, 4)

    # 2. Average Word Length (character count)
    word_lengths = [len(w) for w in words]
    avg_word_length = round(sum(word_lengths) / total_words, 2)

    # 3. Word Frequency Distribution using NLTK FreqDist
    fdist = FreqDist(words)
    freq_dist_table = [
        {
            "word": word,
            "count": count,
            "frequency_pct": round((count / total_words) * 100, 2)
        }
        for word, count in fdist.most_common()
    ]

    # 4. Identification of Long Words (>= 7 characters)
    # Long words are frequently associated with higher morphological complexity
    long_words = sorted(list(set(w for w in words if len(w) >= 7)))
    long_word_count = len(long_words)
    long_word_percentage = round(
        (sum(1 for w in words if len(w) >= 7) / total_words) * 100, 2
    )

    # 5. Reference Corpus Analysis (Brown Corpus)
    corpus_ok, corpus_msg = _load_reference_corpus()
    
    # Identify less frequent words: words that do not appear in the Brown corpus
    # or have very low occurrences in the reference standard (< 3)
    less_frequent_words = []
    if corpus_ok and _BROWN_FREQ_CACHE:
        for w in unique_words:
            # Check occurrence in standard corpus
            ref_freq = _BROWN_FREQ_CACHE.get(w, 0)
            if ref_freq < 3 and len(w) > 3:  # Low frequency or domain-specific terminology
                less_frequent_words.append((w, ref_freq))
    else:
        # Fallback heuristic: not in top common vocabulary and length > 6
        for w in unique_words:
            if w not in _BROWN_VOCAB_CACHE and len(w) >= 6:
                less_frequent_words.append((w, 0))

    # Sort rare words alphabetically
    less_frequent_word_list = [w for w, _ in less_frequent_words]
    less_frequent_count = len(less_frequent_word_list)
    
    # Total occurrences of less frequent words in the text
    total_less_frequent_tokens = sum(1 for w in words if w in set(less_frequent_word_list))
    less_frequent_percentage = round(
        (total_less_frequent_tokens / total_words) * 100, 2
    )

    return {
        "total_words": total_words,
        "unique_word_count": unique_word_count,
        "lexical_diversity": lexical_diversity,
        "avg_word_length": avg_word_length,
        "freq_dist_table": freq_dist_table,
        "top_10_words": fdist.most_common(10),
        "long_words": long_words,
        "long_word_count": long_word_count,
        "long_word_percentage": long_word_percentage,
        "less_frequent_words": less_frequent_word_list,
        "less_frequent_count": less_frequent_count,
        "less_frequent_percentage": less_frequent_percentage,
        "corpus_status": corpus_msg,
        "corpus_available": corpus_ok
    }
