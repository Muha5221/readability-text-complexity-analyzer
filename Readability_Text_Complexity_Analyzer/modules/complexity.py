"""
Overall Text Complexity Classification and Rule-Based Explanation Module.
Covers Module 1 and Module 2 computational linguistics concepts:
- Transparent multi-criteria rule-based scoring (No Machine Learning / No Trained Models)
- Indicator 1: Average Sentence Length (ASL)
- Indicator 2: Average Word Length (AWL)
- Indicator 3: Lexical Diversity (Type-Token Ratio: TTR)
- Indicator 4: Percentage of Less Frequent Vocabulary (from Brown reference corpus)
- Indicator 5: Flesch Reading Ease Readability Score (FRE)
- Documented threshold criteria mapping to Easy, Medium, Difficult
- Rule-based natural language linguistic summary generator (no external AI APIs)
"""

from typing import Dict, List, Any

# Transparent, Documented Scoring Thresholds
THRESHOLDS = {
    "asl": {
        "easy": (0.0, 14.99),
        "medium": (15.0, 24.99),
        "difficult": (25.0, float("inf")),
        "label": "Average Sentence Length (words/sentence)"
    },
    "awl": {
        "easy": (0.0, 4.59),
        "medium": (4.60, 5.40),
        "difficult": (5.41, float("inf")),
        "label": "Average Word Length (characters/word)"
    },
    "ttr": {
        "easy": (0.0, 0.54),
        "medium": (0.55, 0.74),
        "difficult": (0.75, 1.01),
        "label": "Lexical Diversity (Type-Token Ratio)"
    },
    "rare_pct": {
        "easy": (0.0, 5.99),
        "medium": (6.0, 15.99),
        "difficult": (16.0, 100.0),
        "label": "Less Frequent Vocabulary Ratio (%)"
    },
    "fre": {
        "easy": (70.0, float("inf")),
        "medium": (50.0, 69.99),
        "difficult": (-float("inf"), 49.99),
        "label": "Flesch Reading Ease Score"
    }
}


def _score_indicator(value: float, indicator: str) -> int:
    """Assign 1 (Easy), 2 (Medium), or 3 (Difficult) based on documented thresholds."""
    if indicator == "fre":
        # In FRE, higher score means easier
        if value >= 70.0:
            return 1
        elif value >= 50.0:
            return 2
        else:
            return 3
    else:
        # For ASL, AWL, TTR, and Rare Pct, higher values mean higher complexity
        bounds = THRESHOLDS[indicator]
        if value <= bounds["easy"][1]:
            return 1
        elif value <= bounds["medium"][1]:
            return 2
        else:
            return 3


def classify_complexity(
    asl: float,
    awl: float,
    ttr: float,
    rare_pct: float,
    fre: float
) -> Dict[str, Any]:
    """
    Perform transparent, rule-based text complexity classification.
    
    Returns:
        Dictionary containing overall classification ('Easy', 'Medium', 'Difficult'),
        composite numerical score, indicator breakdown, and explanation.
    """
    scores = {
        "asl": _score_indicator(asl, "asl"),
        "awl": _score_indicator(awl, "awl"),
        "ttr": _score_indicator(ttr, "ttr"),
        "rare_pct": _score_indicator(rare_pct, "rare_pct"),
        "fre": _score_indicator(fre, "fre")
    }

    score_labels = {1: "Easy", 2: "Medium", 3: "Difficult"}
    
    total_score = sum(scores.values())
    avg_score = round(total_score / len(scores), 2)

    # Composite classification thresholds
    if avg_score < 1.67:
        overall_classification = "Easy"
        color = "#28a745"  # Green
        summary_tone = "accessible and direct"
    elif avg_score < 2.34:
        overall_classification = "Medium"
        color = "#fd7e14"  # Orange
        summary_tone = "moderately complex and balanced"
    else:
        overall_classification = "Difficult"
        color = "#dc3545"  # Red
        summary_tone = "sophisticated, dense, or academic"

    indicator_details = [
        {
            "indicator": "Average Sentence Length (ASL)",
            "value": f"{asl} words",
            "tier": score_labels[scores["asl"]],
            "points": scores["asl"]
        },
        {
            "indicator": "Average Word Length (AWL)",
            "value": f"{awl} chars",
            "tier": score_labels[scores["awl"]],
            "points": scores["awl"]
        },
        {
            "indicator": "Lexical Diversity (TTR)",
            "value": f"{ttr:.3f}",
            "tier": score_labels[scores["ttr"]],
            "points": scores["ttr"]
        },
        {
            "indicator": "Less Frequent Vocabulary Ratio",
            "value": f"{rare_pct}%",
            "tier": score_labels[scores["rare_pct"]],
            "points": scores["rare_pct"]
        },
        {
            "indicator": "Flesch Reading Ease (FRE)",
            "value": f"{fre}",
            "tier": score_labels[scores["fre"]],
            "points": scores["fre"]
        }
    ]

    return {
        "classification": overall_classification,
        "color": color,
        "composite_score": avg_score,
        "summary_tone": summary_tone,
        "scores": scores,
        "indicator_details": indicator_details,
        "thresholds": THRESHOLDS
    }


def generate_linguistic_explanation(
    classification_data: Dict[str, Any],
    readability_data: Dict[str, Any],
    lexical_data: Dict[str, Any],
    pos_data: Dict[str, Any]
) -> str:
    """
    Generate an explainable, rule-based linguistic summary of the analyzed text.
    Uses deterministic conditional templates grounded in Computational Linguistics principles.
    No generative AI or external LLM APIs are used.
    """
    cls = classification_data["classification"]
    comp_score = classification_data["composite_score"]
    asl = readability_data["asl"]
    awl = lexical_data["avg_word_length"]
    ttr = lexical_data["lexical_diversity"]
    rare_pct = lexical_data["less_frequent_percentage"]
    fre = readability_data["flesch_reading_ease"]
    fkgl = readability_data["flesch_kincaid_grade"]
    noun_pct = pos_data["percentages"].get("Nouns", 0.0)
    verb_pct = pos_data["percentages"].get("Verbs", 0.0)

    # 1. Syntactic analysis sentence
    if asl < 15.0:
        syntax_analysis = (
            f"The syntactic structure is straightforward, with an average sentence length of "
            f"**{asl} words per sentence**. Short clause structures reduce cognitive parsing load."
        )
    elif asl <= 24.0:
        syntax_analysis = (
            f"The syntactic structure exhibits moderate variation, with an average sentence length of "
            f"**{asl} words per sentence**, typical of standard expository prose."
        )
    else:
        syntax_analysis = (
            f"The text features extended syntactic constructions, with an average sentence length of "
            f"**{asl} words per sentence**. Long sentences with multiple clauses elevate grammatical complexity."
        )

    # 2. Vocabulary & Morphology analysis sentence
    if awl < 4.6 and rare_pct < 6.0:
        vocab_analysis = (
            f"Lexically, the text relies predominantly on high-frequency, familiar English words "
            f"(average word length: **{awl} characters**; infrequent vocabulary: **{rare_pct}%**)."
        )
    elif awl <= 5.4:
        vocab_analysis = (
            f"The vocabulary demonstrates balanced linguistic diversity, with an average word length of "
            f"**{awl} characters** and **{rare_pct}%** of words identified as less frequent relative to the NLTK Brown reference corpus."
        )
    else:
        vocab_analysis = (
            f"The text employs advanced morphological forms and specialized terminology, reflected by an average "
            f"word length of **{awl} characters** and **{rare_pct}%** infrequent or specialized terms."
        )

    # 3. Grammatical category distribution note
    pos_note = (
        f"Part-of-Speech analysis indicates **{noun_pct}% nouns** and **{verb_pct}% verbs**. "
    )
    if noun_pct > 35.0:
        pos_note += "The high nominal density indicates informational, content-heavy discourse."
    elif verb_pct > 20.0:
        pos_note += "The active verbal distribution points toward dynamic or narrative prose."
    else:
        pos_note += "The grammatical distribution reflects balanced, standard structural composition."

    # 4. Readability and final pedagogical conclusion
    readability_note = (
        f"The Flesch Reading Ease score of **{fre}** corresponds to a **Flesch-Kincaid Grade Level of {fkgl}**."
    )

    final_verdict = (
        f"### Summary Classification: **{cls}** (Composite Indicator Score: {comp_score} / 3.0)\n\n"
        f"1. **Syntax & Sentence Length:** {syntax_analysis}\n\n"
        f"2. **Lexical Characteristics:** {vocab_analysis} Lexical diversity (Type-Token Ratio) is **{ttr:.3f}**.\n\n"
        f"3. **Part-of-Speech Distribution:** {pos_note}\n\n"
        f"4. **Readability Index:** {readability_note}\n\n"
        f"> **Evaluation Note:** This classification is computed using transparent rule-based thresholds on "
        f"surface syntactic and lexical metrics from Computational Linguistics Module 1 and Module 2, "
        f"without reliance on machine learning classifiers."
    )

    return final_verdict
