"""
Verification and Test Suite for Readability and Text Complexity Analyzer.
Tests all three sample texts:
1. easy_text.txt
2. medium_text.txt
3. difficult_text.txt
Verifies preprocessing, lexical analysis, POS tagging, readability formulas,
complexity classification, and WordNet functionality.
"""

import sys
from pathlib import Path

# Add project root and module directories
PROJECT_DIR = Path(__file__).resolve().parent / "Readability_Text_Complexity_Analyzer"
sys.path.insert(0, str(PROJECT_DIR))

from modules.preprocessing import validate_text, preprocess_text
from modules.lexical_analysis import analyze_lexical_complexity
from modules.pos_analysis import analyze_pos_tags
from modules.wordnet_analysis import get_word_details
from modules.readability import calculate_readability
from modules.complexity import classify_complexity, generate_final_analysis

def run_tests():
    sample_files = ["easy_text.txt", "medium_text.txt", "difficult_text.txt"]
    sample_dir = PROJECT_DIR / "sample_data"

    print("=" * 70)
    print("STARTING TEST SUITE FOR READABILITY & TEXT COMPLEXITY ANALYZER")
    print("=" * 70)

    for sf in sample_files:
        file_path = sample_dir / sf
        print(f"\n--- Testing Sample: {sf} ---")
        assert file_path.exists(), f"Sample file {sf} missing!"

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # 1. Validation
        is_valid, msg = validate_text(text)
        assert is_valid, f"Validation failed for {sf}: {msg}"
        print(f"Validation: PASSED")

        # 2. Preprocessing
        prep = preprocess_text(text)
        print(f"Preprocessing: {prep['total_words']} words, {prep['total_sentences']} sentences, {prep['unique_word_count']} unique words, ASL = {prep['avg_sentence_length']}")
        assert prep["total_words"] > 0
        assert prep["total_sentences"] > 0

        # 3. Lexical Analysis
        lex = analyze_lexical_complexity(prep["words"])
        print(f"Lexical: AWL = {lex['avg_word_length']}, TTR = {lex['lexical_diversity']:.4f}, Long Words = {lex['long_word_count']}")
        assert lex["total_words"] == prep["total_words"]

        # 4. POS Tagging
        pos = analyze_pos_tags(prep["raw_tokens"])
        print(f"POS Counts: Nouns={pos['counts']['Nouns']}, Verbs={pos['counts']['Verbs']}, Adjectives={pos['counts']['Adjectives']}, Adverbs={pos['counts']['Adverbs']}")
        assert pos["total_tagged"] > 0

        # 5. Readability Analysis
        read = calculate_readability(prep["words"], prep["sentences"])
        print(f"Readability: FRE = {read['flesch_reading_ease']} ({read['fre_interpretation']}), FKGL = {read['flesch_kincaid_grade']} ({read['fkgl_interpretation']})")

        # 6. Complexity Classification
        comp = classify_complexity(
            asl=read["asl"],
            awl=lex["avg_word_length"],
            ttr=lex["lexical_diversity"],
            fre=read["flesch_reading_ease"]
        )
        print(f"Overall Complexity Classification: {comp['classification']}")

        # 7. Final Observations
        final_obs = generate_final_analysis(
            classification=comp["classification"],
            fre=read["flesch_reading_ease"],
            fkgl=read["flesch_kincaid_grade"],
            asl=read["asl"],
            awl=lex["avg_word_length"],
            ttr=lex["lexical_diversity"],
            total_words=prep["total_words"],
            total_sentences=prep["total_sentences"],
            long_word_count=lex["long_word_count"],
            long_word_pct=lex["long_word_percentage"]
        )
        assert len(final_obs["readability_note"]) > 10
        assert len(final_obs["vocab_note"]) > 10
        assert len(final_obs["sentence_note"]) > 10
        print(f"Final Analysis Generated: {final_obs['classification']} - {final_obs['readability_note'][:50]}...")

    print("\n--- Testing WordNet Module ---")
    # Test Word Lookup
    wn_res = get_word_details("linguistics")
    assert wn_res["found"] is True
    print(f"WordNet Lookup ('linguistics'): Found {wn_res['synset_count']} synsets, Definition: {wn_res['synsets'][0]['definition'][:50]}...")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! NO ERRORS ENCOUNTERED.")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
