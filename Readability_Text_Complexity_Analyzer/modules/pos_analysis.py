"""
Part-of-Speech (POS) Tagging Analysis Module for Readability and Text Complexity Analyzer.
Covers Module 1 (Introduction to Language Processing) concepts:
- Part-of-Speech Tagging using NLTK (nltk.pos_tag)
- Grammatical category identification (Nouns, Verbs, Adjectives, Adverbs)
- Tagged text frequency and percentage distribution
- Corpus linguistics category breakdown for visual charts
"""

from typing import Dict, List, Tuple, Any
import nltk

def ensure_pos_resources() -> None:
    """Ensure NLTK POS tagger models are available."""
    for resource in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
        try:
            nltk.data.find(f"taggers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


def analyze_pos_tags(raw_tokens: List[str]) -> Dict[str, Any]:
    """
    Perform POS tagging and compute grammatical category distributions.
    
    Args:
        raw_tokens: List of tokens as produced by NLTK word_tokenize.
        
    Returns:
        Dictionary containing counts, percentages, tagged token pairs,
        and category-specific sample words.
    """
    ensure_pos_resources()
    
    if not raw_tokens:
        return {
            "tagged_tokens": [],
            "counts": {"Nouns": 0, "Verbs": 0, "Adjectives": 0, "Adverbs": 0, "Other": 0},
            "percentages": {"Nouns": 0.0, "Verbs": 0.0, "Adjectives": 0.0, "Adverbs": 0.0, "Other": 0.0},
            "total_tagged": 0,
            "category_samples": {"Nouns": [], "Verbs": [], "Adjectives": [], "Adverbs": []}
        }
    
    try:
        tagged_tokens: List[Tuple[str, str]] = nltk.pos_tag(raw_tokens)
    except Exception as e:
        # Graceful fallback: basic regex heuristics if tagger fails
        tagged_tokens = [(t, "NN") if t.isalpha() else (t, ".") for t in raw_tokens]

    # Initialize category counters and sample collector
    counts = {
        "Nouns": 0,
        "Verbs": 0,
        "Adjectives": 0,
        "Adverbs": 0,
        "Other": 0
    }
    
    category_samples: Dict[str, List[str]] = {
        "Nouns": [],
        "Verbs": [],
        "Adjectives": [],
        "Adverbs": []
    }

    # Filter out pure punctuation tokens from POS grammatical distribution
    grammatical_tokens = [
        (word, tag) for word, tag in tagged_tokens 
        if any(c.isalpha() for c in word)
    ]
    
    total_grammatical = len(grammatical_tokens)

    for word, tag in grammatical_tokens:
        word_clean = word.lower()
        if tag.startswith("NN"):
            counts["Nouns"] += 1
            if word_clean not in category_samples["Nouns"]:
                category_samples["Nouns"].append(word_clean)
        elif tag.startswith("VB"):
            counts["Verbs"] += 1
            if word_clean not in category_samples["Verbs"]:
                category_samples["Verbs"].append(word_clean)
        elif tag.startswith("JJ"):
            counts["Adjectives"] += 1
            if word_clean not in category_samples["Adjectives"]:
                category_samples["Adjectives"].append(word_clean)
        elif tag.startswith("RB") or tag == "WRB":
            counts["Adverbs"] += 1
            if word_clean not in category_samples["Adverbs"]:
                category_samples["Adverbs"].append(word_clean)
        else:
            counts["Other"] += 1

    # Calculate percentages
    percentages = {}
    for cat, count in counts.items():
        pct = round((count / total_grammatical) * 100, 2) if total_grammatical > 0 else 0.0
        percentages[cat] = pct

    return {
        "tagged_tokens": tagged_tokens,
        "counts": counts,
        "percentages": percentages,
        "total_tagged": total_grammatical,
        "category_samples": {
            cat: samples[:10] for cat, samples in category_samples.items()
        }
    }
