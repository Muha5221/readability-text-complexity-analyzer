"""
Readability Analysis Module for Readability and Text Complexity Analyzer.
Covers Module 1 and Module 2 computational linguistics concepts:
- Explainable, documented rule-based syllable counting heuristic
- Average Sentence Length (ASL = Total Words / Total Sentences)
- Average Syllables per Word (ASW = Total Syllables / Total Words)
- Flesch Reading Ease (FRE) formula
- Flesch-Kincaid Grade Level (FKGL) formula
- Readability categorization (Easy, Moderate, Difficult)
- Clear separation between numeric estimates and qualitative interpretation
"""

import re
from typing import Dict, List, Tuple, Any

def count_syllables_in_word(word: str) -> int:
    """
    Count the number of syllables in an English word using a documented heuristic algorithm.
    
    Academic Syllable Counting Rules:
    1. Normalization: Convert the word to lowercase and strip non-alphabetic characters.
    2. Minimum Length Rule: Words with 3 or fewer letters typically have 1 syllable (e.g., 'the', 'cat', 'sky').
    3. Vowel Group Counting: Identify consecutive sequences of vowels (a, e, i, o, u, y).
       Each vowel diphthong/sequence initially counts as one nucleus sound.
    4. Silent 'e' Rule: An 'e' at the end of a word is usually silent (e.g., 'make', 'drive', 'rate'),
       EXCEPT when preceded by 'l' (as in '-le' in 'ta-ble', 'sim-ple') or when the word ends in 'ee' (as in 'tree', 'see').
    5. Inflectional Endings: Suffixes '-es' or '-ed' often do not add an extra syllable sound (e.g., 'looked', 'hopes'),
       except when preceded by 'd' or 't' (e.g., 'wanted', 'needed').
    6. Minimum Floor: Every non-empty English word contains at least one syllable sound.
    
    Args:
        word: A single word string.
        
    Returns:
        Estimated syllable count (integer >= 1).
    """
    w = word.strip().lower()
    w = re.sub(r'[^a-z]', '', w)
    
    if not w:
        return 1
    if len(w) <= 3:
        return 1
        
    # Find all vowel sound sequences (a, e, i, o, u, y)
    vowels = "aeiouy"
    vowel_runs = re.findall(r'[aeiouy]+', w)
    syllable_count = len(vowel_runs)
    
    # Silent 'e' adjustment
    if w.endswith('e') and not w.endswith('ee'):
        # Check if ending with '-le' preceded by consonant (e.g. 'ta-ble', 'bot-tle')
        if len(w) >= 3 and w[-2] == 'l' and w[-3] not in vowels:
            pass  # Syllabic 'l', keep syllable
        else:
            syllable_count -= 1
            
    # Silent '-ed' adjustment (e.g., 'walked', 'jumped')
    # If ends in '-ed' not preceded by 't' or 'd'
    if w.endswith('ed') and len(w) >= 4:
        if w[-3] not in ('t', 'd'):
            syllable_count -= 1
            
    # Silent '-es' adjustment (e.g., 'rates', 'hopes')
    # If ends in '-es' preceded by consonant, except 'c', 's', 'x', 'z', 'ch', 'sh'
    if w.endswith('es') and len(w) >= 4:
        if not re.search(r'(ch|sh|[csxz])es$', w):
            syllable_count -= 1

    # Ensure minimum syllable count of 1
    return max(1, syllable_count)


def calculate_readability(
    words: List[str], 
    sentences: List[str]
) -> Dict[str, Any]:
    """
    Calculate Flesch Reading Ease and Flesch-Kincaid Grade Level.
    
    Args:
        words: List of cleaned word tokens.
        sentences: List of detected sentence strings.
        
    Returns:
        Dictionary containing ASL, ASW, FRE score, FKGL grade,
        syllable breakdown, and interpretations.
    """
    total_words = len(words)
    total_sentences = len(sentences)
    
    if total_words == 0 or total_sentences == 0:
        return {
            "total_words": 0,
            "total_sentences": 0,
            "total_syllables": 0,
            "asl": 0.0,
            "asw": 0.0,
            "flesch_reading_ease": 0.0,
            "fre_interpretation": "Insufficient Data",
            "flesch_kincaid_grade": 0.0,
            "fkgl_interpretation": "Insufficient Data",
            "overall_readability_tier": "Undefined",
            "disclaimer": "Readability metrics require a minimum of text."
        }

    # Syllables per word
    syllable_counts = [count_syllables_in_word(w) for w in words]
    total_syllables = sum(syllable_counts)
    
    # 1. Average Sentence Length (ASL)
    asl = round(total_words / total_sentences, 2)
    
    # 2. Average Syllables per Word (ASW)
    asw = round(total_syllables / total_words, 4)
    
    # 3. Flesch Reading Ease (FRE) Formula:
    # FRE = 206.835 - (1.015 * ASL) - (84.6 * ASW)
    raw_fre = 206.835 - (1.015 * asl) - (84.6 * asw)
    fre = round(raw_fre, 2)
    
    # 4. Flesch-Kincaid Grade Level (FKGL) Formula:
    # FKGL = (0.39 * ASL) + (11.8 * ASW) - 15.59
    raw_fkgl = (0.39 * asl) + (11.8 * asw) - 15.59
    fkgl = round(raw_fkgl, 2)
    
    # 5. Readability Interpretations
    if fre >= 80.0:
        fre_interp = "Easy to read (Conversational style)"
        tier = "Easy"
    elif fre >= 60.0:
        fre_interp = "Standard readability (Grade 8-9 level)"
        tier = "Medium"
    elif fre >= 50.0:
        fre_interp = "Moderate difficulty (High school level)"
        tier = "Medium"
    elif fre >= 30.0:
        fre_interp = "Difficult (College level)"
        tier = "Difficult"
    else:
        fre_interp = "Very difficult (Advanced or technical)"
        tier = "Difficult"
        
    # FKGL Grade interpretation
    if fkgl <= 6.0:
        fkgl_interp = f"Grade {max(1.0, fkgl):.1f} (Elementary)"
    elif fkgl <= 9.0:
        fkgl_interp = f"Grade {fkgl:.1f} (Middle School)"
    elif fkgl <= 12.0:
        fkgl_interp = f"Grade {fkgl:.1f} (High School)"
    else:
        fkgl_interp = f"Grade {fkgl:.1f} (College / Professional)"

    disclaimer = ""

    return {
        "total_words": total_words,
        "total_sentences": total_sentences,
        "total_syllables": total_syllables,
        "asl": asl,
        "asw": asw,
        "flesch_reading_ease": fre,
        "fre_interpretation": fre_interp,
        "flesch_kincaid_grade": max(0.0, fkgl),
        "fkgl_interpretation": fkgl_interp,
        "overall_readability_tier": tier,
        "disclaimer": disclaimer
    }
