"""
WordNet Analysis Module for Readability and Text Complexity Analyzer.
Covers Module 2 (Processing and Understanding Natural Language) concepts:
- WordNet lexical database queries (nltk.corpus.wordnet)
- Synset extraction, definitions, and usage examples
- Synonym discovery from WordNet lemmas
- Semantic similarity calculation (Path Similarity) between word pairs
- Graceful handling when words lack WordNet coverage or valid synsets
"""

from typing import Dict, List, Optional, Tuple, Any
import nltk

def ensure_wordnet_resources() -> None:
    """Ensure WordNet corpus and OMW resources are available."""
    for resource in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


def get_word_details(word: str) -> Dict[str, Any]:
    """
    Retrieve lexical information and synonyms for a given word using WordNet.
    
    Args:
        word: The target word string.
        
    Returns:
        Dictionary containing synset count, definitions, POS categories,
        examples, and verified WordNet synonyms.
    """
    ensure_wordnet_resources()
    
    clean_word = word.strip().lower()
    if not clean_word:
        return {
            "found": False,
            "word": word,
            "message": "Empty word provided.",
            "synonyms": [],
            "synsets": []
        }
        
    try:
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets(clean_word)
    except Exception as e:
        return {
            "found": False,
            "word": word,
            "message": f"WordNet could not be accessed ({e}).",
            "synonyms": [],
            "synsets": []
        }
        
    if not synsets:
        return {
            "found": False,
            "word": clean_word,
            "message": f"No WordNet synsets or lexical definitions found for '{clean_word}'.",
            "synonyms": [],
            "synsets": []
        }

    # Extract distinct synonyms across all lemmas
    synonyms_set = set()
    synset_details = []

    pos_names = {
        "n": "Noun",
        "v": "Verb",
        "a": "Adjective",
        "s": "Adjective Satellite",
        "r": "Adverb"
    }

    for s in synsets:
        pos_label = pos_names.get(s.pos(), s.pos())
        definition = s.definition()
        examples = s.examples()
        lemmas = [lemma.name().replace("_", " ") for lemma in s.lemmas()]

        # Collect genuine synonyms that are not the exact query word itself
        for lemma_name in lemmas:
            if lemma_name.lower() != clean_word:
                synonyms_set.add(lemma_name)

        synset_details.append({
            "name": s.name(),
            "pos": pos_label,
            "definition": definition,
            "examples": examples,
            "lemmas": lemmas
        })

    synonyms_list = sorted(list(synonyms_set))

    return {
        "found": True,
        "word": clean_word,
        "synset_count": len(synsets),
        "synonyms": synonyms_list,
        "has_synonyms": len(synonyms_list) > 0,
        "synsets": synset_details,
        "message": f"Found {len(synsets)} synset(s) and {len(synonyms_list)} distinct synonym(s) in WordNet."
    }


def calculate_word_similarity(word1: str, word2: str) -> Dict[str, Any]:
    """
    Calculate semantic path similarity between two words using WordNet.
    
    Path similarity measures the shortest path connecting synsets in 
    the WordNet 'is-a' (hypernym/hyponym) taxonomy hierarchy.
    Range: 0.0 (distantly related or unlinked) to 1.0 (identical concept).
    
    Args:
        word1: First target word.
        word2: Second target word.
        
    Returns:
        Dictionary containing similarity score, best synset pair, and interpretation.
    """
    ensure_wordnet_resources()
    
    w1 = word1.strip().lower()
    w2 = word2.strip().lower()
    
    if not w1 or not w2:
        return {
            "valid": False,
            "similarity": 0.0,
            "message": "Both words must be non-empty for comparison.",
            "interpretation": "Invalid Input"
        }
        
    try:
        from nltk.corpus import wordnet as wn
        synsets1 = wn.synsets(w1)
        synsets2 = wn.synsets(w2)
    except Exception as e:
        return {
            "valid": False,
            "similarity": 0.0,
            "message": f"WordNet could not be queried ({e}).",
            "interpretation": "WordNet Error"
        }
        
    if not synsets1:
        return {
            "valid": False,
            "similarity": 0.0,
            "message": f"'{w1}' not found in WordNet database.",
            "interpretation": "Word 1 Unknown"
        }
    if not synsets2:
        return {
            "valid": False,
            "similarity": 0.0,
            "message": f"'{w2}' not found in WordNet database.",
            "interpretation": "Word 2 Unknown"
        }

    # Find highest path similarity across compatible synset pairs (same POS)
    max_similarity = 0.0
    best_pair: Optional[Tuple[Any, Any]] = None

    for s1 in synsets1:
        for s2 in synsets2:
            # WordNet taxonomic path similarity is only valid within the same POS
            if s1.pos() == s2.pos():
                sim = s1.path_similarity(s2)
                if sim is not None and sim > max_similarity:
                    max_similarity = sim
                    best_pair = (s1, s2)

    if best_pair is None or max_similarity == 0.0:
        return {
            "valid": False,
            "word1": w1,
            "word2": w2,
            "similarity": 0.0,
            "message": f"No taxonomically connected path found between '{w1}' and '{w2}' (they may belong to different parts of speech or separate taxonomy hierarchies).",
            "interpretation": "No Taxonomic Overlap"
        }

    score = round(max_similarity, 4)
    s1_match, s2_match = best_pair

    # Linguistic interpretation of Path Similarity
    if score >= 0.90:
        interp = "Identical or Near-Synonym (Extremely Strong Semantic Proximity)"
    elif score >= 0.50:
        interp = "Closely Related (High Taxonomic Proximity in WordNet tree)"
    elif score >= 0.25:
        interp = "Moderately Related (Share broader semantic category)"
    else:
        interp = "Distantly Related (Weak conceptual connection)"

    return {
        "valid": True,
        "word1": w1,
        "word2": w2,
        "similarity": score,
        "best_synset1": s1_match.name(),
        "best_synset1_def": s1_match.definition(),
        "best_synset2": s2_match.name(),
        "best_synset2_def": s2_match.definition(),
        "pos": s1_match.pos(),
        "interpretation": interp,
        "message": f"Path similarity score: {score:.4f}"
    }
