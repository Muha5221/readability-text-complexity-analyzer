"""
Readability and Text Complexity Analyzer Using Natural Language Processing (NLTK)
M.Tech Computational Linguistics Course Project

Main Streamlit Application Dashboard
Integrates Preprocessing, Lexical Analysis, POS Tagging, WordNet, Readability,
and Transparent Rule-Based Text Complexity Classification.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root and modules directories are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import custom modular NLP components
from modules.preprocessing import validate_text, preprocess_text, ensure_nltk_resources
from modules.lexical_analysis import analyze_lexical_complexity
from modules.pos_analysis import analyze_pos_tags
from modules.wordnet_analysis import get_word_details, calculate_word_similarity
from modules.readability import calculate_readability
from modules.complexity import classify_complexity, generate_linguistic_explanation

# Page Configuration
st.set_page_config(
    page_title="Readability & Text Complexity Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-easy {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #31C48D;
    }
    .badge-medium {
        background-color: #FEF08A;
        color: #713F12;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #E3A008;
    }
    .badge-difficult {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #F98080;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def load_sample_file(filename: str) -> str:
    """Helper to read sample texts safely."""
    path = CURRENT_DIR / "sample_data" / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


# State management callbacks
if "text_input_area" not in st.session_state:
    st.session_state["text_input_area"] = ""
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "sim_word1" not in st.session_state:
    st.session_state["sim_word1"] = "car"
if "sim_word2" not in st.session_state:
    st.session_state["sim_word2"] = "automobile"


def load_sample_callback(filename: str):
    st.session_state["text_input_area"] = load_sample_file(filename)
    st.session_state["analysis_done"] = True


def clear_text_callback():
    st.session_state["text_input_area"] = ""
    st.session_state["analysis_done"] = False


def set_sim_pair_callback(w1: str, w2: str):
    st.session_state["sim_word1"] = w1
    st.session_state["sim_word2"] = w2


# ==========================================
# SIDEBAR: Course Info & Sample Text Loaders
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/book-stack.png", width=64)
    st.title("NLP Text Analyzer")
    st.caption("M.Tech Computational Linguistics Project")

    st.markdown("---")
    st.subheader("📋 Load Sample Texts")
    st.write("Click a sample to populate and analyze:")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.button("🟢 Easy", use_container_width=True, on_click=load_sample_callback, args=("easy_text.txt",), help="Elementary narrative prose")
    with col_s2:
        st.button("🟡 Medium", use_container_width=True, on_click=load_sample_callback, args=("medium_text.txt",), help="Standard expository prose")
    with col_s3:
        st.button("🔴 Difficult", use_container_width=True, on_click=load_sample_callback, args=("difficult_text.txt",), help="Academic linguistics prose")

    st.markdown("---")
    st.subheader("📚 Syllabus Alignment")
    with st.expander("Module 1: Language Processing", expanded=False):
        st.markdown("""
        - **Word Tokenization:** NLTK `word_tokenize`
        - **Sentence Tokenization:** NLTK `sent_tokenize`
        - **POS Tagging:** Penn Treebank Tagset
        - **Frequency Distribution:** NLTK `FreqDist`
        - **Tagged Text Analysis:** Noun/Verb/Adj/Adv counts
        """)
    with st.expander("Module 2: Natural Language Understanding", expanded=False):
        st.markdown("""
        - **WordNet Lexical Database:** Synsets & definitions
        - **Synonym Extraction:** NLTK `wordnet.lemmas`
        - **Semantic Similarity:** WordNet `path_similarity`
        - **Corpus Linguistics:** NLTK Brown Reference Corpus
        """)

    st.markdown("---")
    st.caption("Grounded in pure symbolic NLP. No deep learning, external APIs, or GPU requirements.")


# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown('<div class="main-header">Readability & Text Complexity Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Analyze English text complexity, syntactic structures, vocabulary distribution, and readability scores using Natural Language Processing (NLTK).</div>',
    unsafe_allow_html=True
)

# ------------------------------------------
# SECTION 1: Text Input & Controls
# ------------------------------------------
st.markdown("### 📝 Section 1: Text Input")

input_val = st.text_area(
    "Enter or paste English text to analyze:",
    height=160,
    placeholder="Type or paste your English paragraph here, or choose a sample from the sidebar...",
    key="text_input_area"
)

col_btn1, col_btn2, col_info = st.columns([1.5, 1.5, 7])

with col_btn1:
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        st.session_state["analysis_done"] = True
with col_btn2:
    st.button("🧹 Clear Text", use_container_width=True, on_click=clear_text_callback)

# Check if text is present and analysis requested
text_to_analyze = st.session_state.get("text_input_area", "").strip()

if st.session_state.get("analysis_done", False) and text_to_analyze:
    is_valid, validation_msg = validate_text(text_to_analyze)
    if not is_valid:
        st.error(f"⚠️ {validation_msg}")
    else:
        # Preprocessing Pipeline
        with st.spinner("Processing text with NLTK..."):
            prep_data = preprocess_text(text_to_analyze)
            lex_data = analyze_lexical_complexity(prep_data["words"])
            pos_data = analyze_pos_tags(prep_data["raw_tokens"])
            read_data = calculate_readability(prep_data["words"], prep_data["sentences"])
            comp_data = classify_complexity(
                asl=read_data["asl"],
                awl=lex_data["avg_word_length"],
                ttr=lex_data["lexical_diversity"],
                rare_pct=lex_data["less_frequent_percentage"],
                fre=read_data["flesch_reading_ease"]
            )
            explanation = generate_linguistic_explanation(comp_data, read_data, lex_data, pos_data)

        st.success("✅ Analysis completed successfully using NLTK linguistic pipeline!")
        st.markdown("---")

        # ------------------------------------------
        # SECTION 2: Overview Metrics
        # ------------------------------------------
        st.markdown("### 📊 Section 2: Overview Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Total Words", prep_data["total_words"])
        with m2:
            st.metric("Total Sentences", prep_data["total_sentences"])
        with m3:
            st.metric("Unique Words", prep_data["unique_word_count"])
        with m4:
            st.metric("Avg Sentence Length", f"{prep_data['avg_sentence_length']} w/s")
        with m5:
            st.metric("Avg Word Length", f"{lex_data['avg_word_length']} chars")

        st.markdown("---")

        # ------------------------------------------
        # SECTION 3: Readability Results
        # ------------------------------------------
        st.markdown("### 📖 Section 3: Readability Results")
        
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Flesch Reading Ease", f"{read_data['flesch_reading_ease']}")
            st.caption(f"**Interpretation:** {read_data['fre_interpretation']}")
        with r2:
            st.metric("Flesch-Kincaid Grade", f"{read_data['flesch_kincaid_grade']}")
            st.caption(f"**Target Level:** {read_data['fkgl_interpretation']}")
        with r3:
            cls_tier = comp_data["classification"]
            badge_class = f"badge-{cls_tier.lower()}"
            st.markdown(f"**Overall Complexity:**")
            st.markdown(f'<div class="{badge_class}" style="font-size: 1.25rem;">{cls_tier} Complexity</div>', unsafe_allow_html=True)
            st.caption(f"Composite Score: **{comp_data['composite_score']} / 3.0** (Rule-Based)")

        st.info(f"💡 **Academic Note:** {read_data['disclaimer']}")
        st.markdown("---")

        # ------------------------------------------
        # SECTION 4: Lexical Analysis
        # ------------------------------------------
        st.markdown("### 🔍 Section 4: Lexical Analysis")

        l_col1, l_col2 = st.columns([1, 1])

        with l_col1:
            st.subheader("Vocabulary Characteristics")
            st.write(f"- **Lexical Diversity (Type-Token Ratio):** `{lex_data['lexical_diversity']:.4f}` ({lex_data['unique_word_count']} unique / {lex_data['total_words']} total words)")
            st.write(f"- **Average Word Length:** `{lex_data['avg_word_length']}` characters per word")
            st.write(f"- **Long Words (≥ 7 letters):** `{lex_data['long_word_count']}` distinct words ({lex_data['long_word_percentage']}% of text tokens)")
            if lex_data['long_words']:
                with st.expander("View Long Words List"):
                    st.write(", ".join(lex_data['long_words']))

            st.write(f"- **Less Frequent Vocabulary:** `{lex_data['less_frequent_count']}` words ({lex_data['less_frequent_percentage']}% of text tokens)")
            st.caption(f"Reference Corpus: {lex_data['corpus_status']}")
            if lex_data['less_frequent_words']:
                with st.expander("View Less Frequent Words (relative to Brown Corpus)"):
                    st.write(", ".join(lex_data['less_frequent_words']))

        with l_col2:
            st.subheader("Top Word Frequencies (NLTK FreqDist)")
            df_freq = pd.DataFrame(lex_data["freq_dist_table"])
            if not df_freq.empty:
                # Matplotlib bar chart for top words
                top_10 = df_freq.head(8)
                fig, ax = plt.subplots(figsize=(6, 3.2))
                ax.barh(top_10["word"][::-1], top_10["count"][::-1], color="#3B82F6")
                ax.set_xlabel("Frequency Count", fontsize=9)
                ax.set_title("Most Frequent Words", fontsize=11, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("View Full Word Frequency Table"):
                    st.dataframe(df_freq, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ------------------------------------------
        # SECTION 5: POS Tagging Analysis
        # ------------------------------------------
        st.markdown("### 🏷️ Section 5: Part-of-Speech (POS) Analysis")
        p1, p2, p3, p4 = st.columns(4)
        pos_counts = pos_data["counts"]
        pos_pcts = pos_data["percentages"]

        with p1:
            st.metric("Nouns", f"{pos_counts['Nouns']}", f"{pos_pcts['Nouns']}%")
        with p2:
            st.metric("Verbs", f"{pos_counts['Verbs']}", f"{pos_pcts['Verbs']}%")
        with p3:
            st.metric("Adjectives", f"{pos_counts['Adjectives']}", f"{pos_pcts['Adjectives']}%")
        with p4:
            st.metric("Adverbs", f"{pos_counts['Adverbs']}", f"{pos_pcts['Adverbs']}%")

        pos_c1, pos_c2 = st.columns([1.2, 1])
        with pos_c1:
            # POS distribution chart
            categories = ["Nouns", "Verbs", "Adjectives", "Adverbs", "Other"]
            cat_counts = [pos_counts.get(c, 0) for c in categories]
            colors = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#9CA3AF"]

            fig_pos, ax_pos = plt.subplots(figsize=(6.5, 3.2))
            bars = ax_pos.bar(categories, cat_counts, color=colors)
            ax_pos.set_ylabel("Token Count", fontsize=9)
            ax_pos.set_title("Grammatical Category Distribution (NLTK POS Tagging)", fontsize=11, fontweight="bold")
            for bar in bars:
                height = bar.get_height()
                ax_pos.annotate(f'{height}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_pos)
            plt.close(fig_pos)

        with pos_c2:
            st.subheader("Sample Tagged Words")
            samples = pos_data["category_samples"]
            st.write(f"**Nouns:** {', '.join(samples['Nouns']) if samples['Nouns'] else 'None'}")
            st.write(f"**Verbs:** {', '.join(samples['Verbs']) if samples['Verbs'] else 'None'}")
            st.write(f"**Adjectives:** {', '.join(samples['Adjectives']) if samples['Adjectives'] else 'None'}")
            st.write(f"**Adverbs:** {', '.join(samples['Adverbs']) if samples['Adverbs'] else 'None'}")
            with st.expander("View Raw POS Tagged Tokens"):
                st.write([(w, tag) for w, tag in pos_data["tagged_tokens"] if any(c.isalpha() for c in w)])

        st.markdown("---")

        # ------------------------------------------
        # SECTION 6: WordNet Explorer
        # ------------------------------------------
        st.markdown("### 🌐 Section 6: WordNet Explorer")
        st.caption("Investigate lexical semantics, synonyms, and semantic similarity via NLTK WordNet.")

        wn_tab1, wn_tab2 = st.tabs(["📖 Lexical Information & Synonyms", "📐 Word Similarity Comparison"])

        with wn_tab1:
            st.write("Select a word from the analyzed text or type a custom word to query WordNet:")
            
            # Populate dropdown with candidate vocabulary from input text
            candidate_words = sorted(list(set(prep_data["words"])))
            col_sel, col_inp = st.columns([1, 1])
            with col_sel:
                selected_from_text = st.selectbox(
                    "Pick a word from your text:",
                    options=["-- Select from text --"] + candidate_words,
                    index=0
                )
            with col_inp:
                custom_word = st.text_input("Or enter any English word:", placeholder="e.g. language, complex, analyze")

            query_word = custom_word.strip() if custom_word.strip() else (
                selected_from_text if selected_from_text != "-- Select from text --" else ""
            )

            if query_word:
                wn_result = get_word_details(query_word)
                if not wn_result["found"]:
                    st.warning(f"ℹ️ {wn_result['message']}")
                else:
                    st.write(f"#### WordNet Lexical Entry for: **{wn_result['word']}**")
                    st.info(wn_result["message"])

                    # Synonyms display
                    if wn_result["has_synonyms"]:
                        st.markdown("**Discovered WordNet Synonyms:**")
                        st.write(", ".join([f"`{syn}`" for syn in wn_result["synonyms"]]))
                    else:
                        st.write("*(No direct synonyms available in WordNet for this word)*")

                    # Synset details
                    st.markdown("**Available Synset Definitions & Examples:**")
                    for s in wn_result["synsets"][:4]:
                        with st.container():
                            st.markdown(f"- **`{s['name']}`** (*{s['pos']}*): {s['definition']}")
                            if s["examples"]:
                                st.caption(f"  *Example:* \"{s['examples'][0]}\"")
            else:
                st.info("Select or enter a word above to view WordNet lexical information and synonyms.")

        with wn_tab2:
            st.write("Compare the semantic relatedness of two words using WordNet **Path Similarity**:")
            
            # Quick presets for demonstration
            st.write("**Quick Demo Presets:**")
            p_c1, p_c2, p_c3 = st.columns(3)
            with p_c1:
                st.button("🚗 car & automobile (1.0)", on_click=set_sim_pair_callback, args=("car", "automobile"))
            with p_c2:
                st.button("🐶 dog & cat (0.20)", on_click=set_sim_pair_callback, args=("dog", "cat"))
            with p_c3:
                st.button("👨 boy & man (0.33)", on_click=set_sim_pair_callback, args=("boy", "man"))

            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                w1_input = st.text_input("Word 1:", value=st.session_state["sim_word1"], key="sim_input_w1")
            with sim_col2:
                w2_input = st.text_input("Word 2:", value=st.session_state["sim_word2"], key="sim_input_w2")

            if st.button("Calculate WordNet Similarity", type="primary"):
                sim_res = calculate_word_similarity(w1_input, w2_input)
                if not sim_res["valid"]:
                    st.warning(f"⚠️ {sim_res['message']}")
                else:
                    st.success(f"**WordNet Path Similarity Score: `{sim_res['similarity']}`** (Range: 0.0 to 1.0)")
                    st.write(f"- **Semantic Interpretation:** {sim_res['interpretation']}")
                    st.write(f"- **Synset 1:** `{sim_res['best_synset1']}` ({sim_res['best_synset1_def']})")
                    st.write(f"- **Synset 2:** `{sim_res['best_synset2']}` ({sim_res['best_synset2_def']})")
                    st.caption("Path similarity reflects the shortest path length connecting the two concepts in WordNet's hypernym/hyponym hierarchy.")

        st.markdown("---")

        # ------------------------------------------
        # SECTION 7: Final Analysis & Linguistic Summary
        # ------------------------------------------
        st.markdown("### 📝 Section 7: Final Analysis & Linguistic Summary")
        st.markdown(explanation)

        # Thresholds reference expander
        with st.expander("ℹ️ Rule-Based Classification Threshold Reference (Course Evaluation Detail)"):
            st.markdown("""
            | Linguistic Indicator | Easy Criteria | Medium Criteria | Difficult Criteria |
            | :--- | :--- | :--- | :--- |
            | **Average Sentence Length (ASL)** | < 15 words/sent | 15 – 24 words/sent | ≥ 25 words/sent |
            | **Average Word Length (AWL)** | < 4.60 chars/word | 4.60 – 5.40 chars/word | > 5.40 chars/word |
            | **Lexical Diversity (TTR)** | < 0.55 | 0.55 – 0.74 | ≥ 0.75 |
            | **Less Frequent Vocabulary** | < 6.0% | 6.0% – 15.99% | ≥ 16.0% |
            | **Flesch Reading Ease (FRE)** | ≥ 70.0 | 50.0 – 69.99 | < 50.0 |
            
            *Overall Classification is determined by the average score across all 5 linguistic indicators (1 = Easy, 2 = Medium, 3 = Difficult).*
            """)

elif st.session_state.get("analysis_done", False) and not text_to_analyze:
    st.error("⚠️ Input text is empty. Please enter or paste English text to analyze.")
else:
    # Initial landing guide when no analysis has run
    st.info("👆 Please paste English text above and click **'Analyze Text'**, or select one of the sample texts from the sidebar to begin.")
