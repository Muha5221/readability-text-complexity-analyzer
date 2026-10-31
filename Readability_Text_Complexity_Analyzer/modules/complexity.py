"""
Complexity Classification and Practical Analysis Module.
Provides rule-based classification and concise observations on text readability,
vocabulary, and sentence structure.
"""

from typing import Dict, Any


def classify_complexity(
    asl: float,
    awl: float,
    ttr: float,
    fre: float,
    rare_pct: float = 0.0
) -> Dict[str, Any]:
    """
    Classify text complexity as Easy, Medium, or Difficult using
    standard readability and structural indicators.
    """
    # Threshold rules based on standard readability and structural norms
    # 1. Easy: High reading ease (>= 70), short sentences (<= 15 words), simple words (<= 4.8 chars)
    # 2. Difficult: Low reading ease (< 50) or long sentences (>= 22 words) or long words (>= 6.0 chars)
    # 3. Medium: Everything in between

    points = 0
    
    # Readability Ease contribution
    if fre >= 70.0:
        points += 1
    elif fre >= 50.0:
        points += 2
    else:
        points += 3

    # Sentence Length contribution
    if asl <= 14.0:
        points += 1
    elif asl <= 22.0:
        points += 2
    else:
        points += 3

    # Word Length contribution
    if awl <= 4.8:
        points += 1
    elif awl <= 5.8:
        points += 2
    else:
        points += 3

    # Lexical Diversity contribution
    if ttr < 0.65:
        points += 1
    elif ttr <= 0.85:
        points += 2
    else:
        points += 3

    avg_points = points / 4.0

    if avg_points <= 1.5:
        classification = "Easy"
        color = "#10B981"  # Emerald
    elif avg_points <= 2.3:
        classification = "Medium"
        color = "#F59E0B"  # Amber
    else:
        classification = "Difficult"
        color = "#EF4444"  # Red

    return {
        "classification": classification,
        "color": color
    }


def generate_final_analysis(
    classification: str,
    fre: float,
    fkgl: float,
    asl: float,
    awl: float,
    ttr: float,
    total_words: int,
    total_sentences: int,
    long_word_count: int,
    long_word_pct: float
) -> Dict[str, str]:
    """
    Generate concise, practical observations based on calculated metrics.
    Avoids long paragraphs, academic jargon, and composite scoring.
    """
    # Readability summary
    if fre >= 80.0:
        readability_note = f"High reading ease ({fre:.1f}). Text is clear and easily understood by general audiences (Grade {fkgl:.1f})."
    elif fre >= 60.0:
        readability_note = f"Standard reading ease ({fre:.1f}). Text is accessible to most readers with plain English phrasing (Grade {fkgl:.1f})."
    elif fre >= 50.0:
        readability_note = f"Moderate reading ease ({fre:.1f}). Text requires high-school level comprehension (Grade {fkgl:.1f})."
    else:
        readability_note = f"Low reading ease ({fre:.1f}). Text contains dense structures suitable for advanced or professional readers (Grade {fkgl:.1f})."

    # Vocabulary summary
    if awl <= 4.8:
        vocab_note = f"Average word length is {awl:.2f} characters. Vocabulary is familiar and direct, with {long_word_count} longer words ({long_word_pct:.1f}% of total)."
    elif awl <= 5.8:
        vocab_note = f"Average word length is {awl:.2f} characters with a lexical diversity of {ttr:.2f}. Contains {long_word_count} words with 7+ letters ({long_word_pct:.1f}%)."
    else:
        vocab_note = f"Average word length is {awl:.2f} characters. Uses specialized or multi-syllable vocabulary, with {long_word_count} long words ({long_word_pct:.1f}% of total)."

    # Sentence structure summary
    if asl <= 14.0:
        sentence_note = f"Sentences average {asl:.1f} words across {total_sentences} sentence(s). Short sentences make the flow quick and easy to follow."
    elif asl <= 22.0:
        sentence_note = f"Sentences average {asl:.1f} words across {total_sentences} sentence(s). Sentence lengths are well-balanced for informative writing."
    else:
        sentence_note = f"Sentences average {asl:.1f} words across {total_sentences} sentence(s). Long sentence constructions require closer attention from the reader."

    return {
        "classification": classification,
        "readability_note": readability_note,
        "vocab_note": vocab_note,
        "sentence_note": sentence_note
    }
