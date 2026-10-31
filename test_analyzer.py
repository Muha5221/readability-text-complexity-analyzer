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
from modules.wordnet_analysis import get_word_details, calculate_word_similarity
from modules.readability import calculate_readability
from modules.complexity import classify_complexity, generate_linguistic_explanation

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
        print(f"Lexical: AWL = {lex['avg_word_length']}, TTR = {lex['lexical_diversity']:.4f}, Long Words = {lex['long_word_count']}, Rare Pct = {lex['less_frequent_percentage']}%")
        assert lex["total_words"] == prep["total_words"]
        assert lex["corpus_available"] is True, f"Brown corpus not available: {lex['corpus_status']}"

        # 4. POS Tagging
        pos = analyze_pos_tags(prep["raw_tokens"])
        print(f"POS Counts: Nouns={pos['counts']['Nouns']}, Verbs={pos['counts']['Verbs']}, Adjectives={pos['counts']['Adjectives']}, Adverbs={pos['counts']['Adverbs']}")
        assert pos["total_tagged"] > 0

        # 5. Readability Analysis
        read = calculate_readability(prep["words"], prep["sentences"])
        print(f"Readability: FRE = {read['flesch_reading_ease']} ({read['fre_interpretation'][:20]}...), FKGL = {read['flesch_kincaid_grade']} ({read['fkgl_interpretation']})")

        # 6. Complexity Classification
        comp = classify_complexity(
            asl=read["asl"],
            awl=lex["avg_word_length"],
            ttr=lex["lexical_diversity"],
            rare_pct=lex["less_frequent_percentage"],
            fre=read["flesch_reading_ease"]
        )
        print(f"Overall Complexity Classification: {comp['classification']} (Score: {comp['composite_score']})")

        # 7. Linguistic Explanation
        expl = generate_linguistic_explanation(comp, read, lex, pos)
        assert len(expl) > 100
        print(f"Explanation Generated: {len(expl)} characters")

    print("\n--- Testing WordNet Module ---")
    # Test Word Lookup
    wn_res = get_word_details("linguistics")
    assert wn_res["found"] is True
    print(f"WordNet Lookup ('linguistics'): Found {wn_res['synset_count']} synsets, Definition: {wn_res['synsets'][0]['definition'][:50]}...")

    # Test Word Similarity
    sim_res = calculate_word_similarity("algorithm", "computer")
    assert sim_res["valid"] is True
    print(f"WordNet Similarity ('algorithm' vs 'computer'): Score = {sim_res['similarity']} ({sim_res['interpretation']})")

    # Test Dissimilar words
    sim_res_diff = calculate_word_similarity("dog", "algorithm")
    print(f"WordNet Similarity ('dog' vs 'algorithm'): Score = {sim_res_diff['similarity']} ({sim_res_diff['interpretation']})")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! NO ERRORS ENCOUNTERED.")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
