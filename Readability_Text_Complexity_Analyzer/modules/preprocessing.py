"""
Text Preprocessing Module for Readability and Text Complexity Analyzer.
Covers Module 1 (Introduction to Language Processing) concepts:
- Sentence Tokenization (nltk.sent_tokenize)
- Word Tokenization (nltk.word_tokenize)
- Punctuation removal for statistical analysis
- Case normalization (lowercasing)
- Vocabulary counting (words, sentences, unique words)
"""

import string
import re
from typing import Dict, List, Tuple, Any
import nltk

def ensure_nltk_resources() -> None:
    """Ensure essential NLTK tokenizer models are available."""
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

def validate_text(text: str) -> Tuple[bool, str]:
    """
    Validate the input text.
    
    Returns:
        (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Input text is empty. Please enter or paste English text to analyze."
    
    cleaned = text.strip()
    words = re.findall(r'\b[a-zA-Z]+\b', cleaned)
    
    if len(words) < 5:
        return False, (
            f"Input text is too short ({len(words)} word{'s' if len(words) != 1 else ''}). "
            "Please provide at least 5 words to analyze."
        )
    
    return True, ""

def preprocess_text(text: str) -> Dict[str, Any]:
    """
    Perform complete text preprocessing using NLTK.
    
    Args:
        text: Raw English input string.
        
    Returns:
        Dictionary containing sentences, raw tokens, cleaned words,
        counts, and average sentence length.
    """
    ensure_nltk_resources()
    
    raw_text = text.strip()
    
    # 1. Sentence Tokenization
    try:
        sentences = nltk.sent_tokenize(raw_text)
    except Exception:
        # Fallback regex sentence tokenizer
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_text) if s.strip()]
    
    if not sentences:
        sentences = [raw_text]
        
    # 2. Word Tokenization
    try:
        raw_tokens = nltk.word_tokenize(raw_text)
    except Exception:
        # Fallback regex word tokenizer
        raw_tokens = re.findall(r'\w+|[^\w\s]', raw_text)
    
    # 3. Clean and normalize words:
    # Remove punctuation, keep alphabetic words, convert to lowercase
    # This ensures punctuation characters do not distort vocabulary frequencies
    words = [
        token.lower() for token in raw_tokens 
        if any(c.isalpha() for c in token) and token not in string.punctuation
    ]
    
    # Extra safety: if tokenization produced no purely alphabetic words, fallback
    if not words:
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', raw_text)]
    
    total_sentences = len(sentences)
    total_words = len(words)
    unique_words = sorted(list(set(words)))
    unique_word_count = len(unique_words)
    
    # Sentence-level word breakdown
    sentence_word_counts = []
    for sent in sentences:
        s_words = [
            t.lower() for t in nltk.word_tokenize(sent) 
            if any(c.isalpha() for c in t) and t not in string.punctuation
        ]
        sentence_word_counts.append(len(s_words))
        
    avg_sentence_length = (
        round(total_words / total_sentences, 2) if total_sentences > 0 else 0.0
    )
    
    return {
        "raw_text": raw_text,
        "sentences": sentences,
        "total_sentences": total_sentences,
        "raw_tokens": raw_tokens,
        "words": words,
        "total_words": total_words,
        "unique_words": unique_words,
        "unique_word_count": unique_word_count,
        "sentence_word_counts": sentence_word_counts,
        "avg_sentence_length": avg_sentence_length
    }
