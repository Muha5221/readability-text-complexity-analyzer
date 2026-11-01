"""
Readability & Text Complexity Analyzer
A clean, professional tool to analyze English text readability, vocabulary, and sentence structure.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root and modules directories are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import modular NLP components
from modules.preprocessing import validate_text, preprocess_text
from modules.lexical_analysis import analyze_lexical_complexity
from modules.pos_analysis import analyze_pos_tags
from modules.readability import calculate_readability
from modules.complexity import classify_complexity, generate_final_analysis

# Page Configuration
st.set_page_config(
    page_title="Readability & Text Complexity Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme CSS
st.markdown("""
<style>
    /* Main container adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Header typography */
    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.75rem;
    }
    
    /* Section Headings */
    .section-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.4rem;
    }
    
    /* Cards */
    .stat-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stat-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .stat-lbl {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    
    /* Classification Badges */
    .badge-easy {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid #059669;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid #D97706;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-difficult {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid #DC2626;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Final Analysis Box */
    .analysis-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin-top: 10px;
    }
    .analysis-item {
        margin-bottom: 12px;
        color: #CBD5E1;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .analysis-item strong {
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)


def load_sample_file(filename: str) -> str:
    """Read sample texts safely."""
    path = CURRENT_DIR / "sample_data" / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


# Session State Management
if "text_input_area" not in st.session_state:
    st.session_state["text_input_area"] = ""
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False


def load_sample_callback(filename: str):
    st.session_state["text_input_area"] = load_sample_file(filename)
    st.session_state["analysis_done"] = True


def clear_text_callback():
    st.session_state["text_input_area"] = ""
    st.session_state["analysis_done"] = False


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.subheader("Readability Analyzer")
    st.write("Analyze text readability and vocabulary complexity.")

    st.markdown("---")
    st.write("**Sample Texts**")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.button("Easy", use_container_width=True, on_click=load_sample_callback, args=("easy_text.txt",))
    with col_s2:
        st.button("Medium", use_container_width=True, on_click=load_sample_callback, args=("medium_text.txt",))
    with col_s3:
        st.button("Difficult", use_container_width=True, on_click=load_sample_callback, args=("difficult_text.txt",))


# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown('<div class="app-title">Readability & Text Complexity Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Analyze text readability, vocabulary and sentence structure.</div>', unsafe_allow_html=True)

# ------------------------------------------
# SECTION 1: Text Input
# ------------------------------------------
st.markdown('<div class="section-title">Section 1: Text Input</div>', unsafe_allow_html=True)

input_val = st.text_area(
    "Enter or paste English text to analyze:",
    height=160,
    placeholder="Type or paste your English text here, or choose a sample from the sidebar...",
    key="text_input_area",
    label_visibility="collapsed"
)

col_btn1, col_btn2, col_space = st.columns([1.5, 1.5, 7])

with col_btn1:
    if st.button("Analyze Text", type="primary", use_container_width=True):
        st.session_state["analysis_done"] = True
with col_btn2:
    st.button("Clear Text", use_container_width=True, on_click=clear_text_callback)

text_to_analyze = st.session_state.get("text_input_area", "").strip()

if st.session_state.get("analysis_done", False) and text_to_analyze:
    is_valid, validation_msg = validate_text(text_to_analyze)
    if not is_valid:
        st.error(validation_msg)
    else:
        # Compute metrics
        prep_data = preprocess_text(text_to_analyze)
        lex_data = analyze_lexical_complexity(prep_data["words"])
        pos_data = analyze_pos_tags(prep_data["raw_tokens"])
        read_data = calculate_readability(prep_data["words"], prep_data["sentences"])
        comp_data = classify_complexity(
            asl=read_data["asl"],
            awl=lex_data["avg_word_length"],
            ttr=lex_data["lexical_diversity"],
            fre=read_data["flesch_reading_ease"]
        )
        final_obs = generate_final_analysis(
            classification=comp_data["classification"],
            fre=read_data["flesch_reading_ease"],
            fkgl=read_data["flesch_kincaid_grade"],
            asl=read_data["asl"],
            awl=lex_data["avg_word_length"],
            ttr=lex_data["lexical_diversity"],
            total_words=prep_data["total_words"],
            total_sentences=prep_data["total_sentences"],
            long_word_count=lex_data["long_word_count"],
            long_word_pct=lex_data["long_word_percentage"]
        )

        st.success("Analysis completed.")

        # ------------------------------------------
        # SECTION 2: Text Statistics
        # ------------------------------------------
        st.markdown('<div class="section-title">Section 2: Text Statistics</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Total Words", prep_data["total_words"])
        with m2:
            st.metric("Total Sentences", prep_data["total_sentences"])
        with m3:
            st.metric("Unique Words", prep_data["unique_word_count"])
        with m4:
            st.metric("Average Sentence Length", f"{prep_data['avg_sentence_length']} words")
        with m5:
            st.metric("Average Word Length", f"{lex_data['avg_word_length']} chars")

        # ------------------------------------------
        # SECTION 3: Readability Analysis
        # ------------------------------------------
        st.markdown('<div class="section-title">Section 3: Readability Analysis</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Flesch Reading Ease", f"{read_data['flesch_reading_ease']}")
            st.caption(read_data["fre_interpretation"])
        with r2:
            st.metric("Flesch-Kincaid Grade Level", f"{read_data['flesch_kincaid_grade']}")
            st.caption(read_data["fkgl_interpretation"])
        with r3:
            cls_name = comp_data["classification"]
            badge_class = f"badge-{cls_name.lower()}"
            st.markdown("<div style='font-size:0.875rem; color:#94A3B8; margin-bottom:8px;'>READABILITY CLASSIFICATION</div>", unsafe_allow_html=True)
            st.markdown(f'<div class="{badge_class}" style="font-size: 1.15rem;">{cls_name}</div>', unsafe_allow_html=True)

        # ------------------------------------------
        # SECTION 4: Vocabulary Analysis
        # ------------------------------------------
        st.markdown('<div class="section-title">Section 4: Vocabulary Analysis</div>', unsafe_allow_html=True)
        v_col1, v_col2 = st.columns([1, 1])

        with v_col1:
            st.write(f"**Lexical Diversity (Type-Token Ratio):** `{lex_data['lexical_diversity']:.3f}` ({lex_data['unique_word_count']} unique / {lex_data['total_words']} total words)")
            st.write(f"**Average Word Length:** `{lex_data['avg_word_length']}` characters")
            st.write(f"**Long Words (7+ characters):** `{lex_data['long_word_count']}` words ({lex_data['long_word_percentage']}%)")
            if lex_data["long_words"]:
                with st.expander("View list of long words"):
                    st.write(", ".join(lex_data["long_words"]))

        with v_col2:
            df_freq = pd.DataFrame(lex_data["freq_dist_table"])
            if not df_freq.empty:
                top_8 = df_freq.head(8)
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_facecolor("#1E293B")
                ax.set_facecolor("#1E293B")
                
                bars = ax.barh(top_8["word"][::-1], top_8["count"][::-1], color="#38BDF8", height=0.6)
                ax.set_xlabel("Count", color="#94A3B8", fontsize=9)
                ax.set_title("Most Frequent Words", color="#F1F5F9", fontsize=10, fontweight="600")
                ax.tick_params(colors="#CBD5E1", labelsize=9)
                for spine in ax.spines.values():
                    spine.set_color("#334155")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("View full frequency table"):
                    st.dataframe(df_freq, use_container_width=True, hide_index=True)

        # ------------------------------------------
        # SECTION 5: Part-of-Speech Analysis
        # ------------------------------------------
        st.markdown('<div class="section-title">Section 5: Part-of-Speech Analysis</div>', unsafe_allow_html=True)
        pos_counts = pos_data["counts"]
        pos_pcts = pos_data["percentages"]

        p1, p2, p3, p4 = st.columns(4)
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
            categories = ["Nouns", "Verbs", "Adjectives", "Adverbs", "Other"]
            cat_counts = [pos_counts.get(c, 0) for c in categories]
            colors = ["#38BDF8", "#34D399", "#FBBF24", "#A78BFA", "#64748B"]

            fig_pos, ax_pos = plt.subplots(figsize=(6, 3))
            fig_pos.patch.set_facecolor("#1E293B")
            ax_pos.set_facecolor("#1E293B")

            bars_pos = ax_pos.bar(categories, cat_counts, color=colors, width=0.55)
            ax_pos.set_ylabel("Count", color="#94A3B8", fontsize=9)
            ax_pos.set_title("Part-of-Speech Distribution", color="#F1F5F9", fontsize=10, fontweight="600")
            ax_pos.tick_params(colors="#CBD5E1", labelsize=9)
            for spine in ax_pos.spines.values():
                spine.set_color("#334155")

            for bar in bars_pos:
                h = bar.get_height()
                ax_pos.annotate(f"{h}",
                                xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 2),
                                textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, color="#F1F5F9")
            plt.tight_layout()
            st.pyplot(fig_pos)
            plt.close(fig_pos)

        with pos_c2:
            samples = pos_data["category_samples"]
            st.write(f"**Nouns:** {', '.join(samples['Nouns']) if samples['Nouns'] else 'None'}")
            st.write(f"**Verbs:** {', '.join(samples['Verbs']) if samples['Verbs'] else 'None'}")
            st.write(f"**Adjectives:** {', '.join(samples['Adjectives']) if samples['Adjectives'] else 'None'}")
            st.write(f"**Adverbs:** {', '.join(samples['Adverbs']) if samples['Adverbs'] else 'None'}")
            with st.expander("View tagged tokens"):
                st.write([(w, tag) for w, tag in pos_data["tagged_tokens"] if any(c.isalpha() for c in w)])

        # ------------------------------------------
        # SECTION 6: Final Analysis
        # ------------------------------------------
        st.markdown('<div class="section-title">Section 6: Final Analysis</div>', unsafe_allow_html=True)

        final_cls = final_obs["classification"]
        badge_cls = f"badge-{final_cls.lower()}"

        st.markdown(f"""
        <div class="analysis-box">
            <div style="margin-bottom: 14px;">
                <span style="font-size: 0.9rem; color: #94A3B8; margin-right: 8px;">OVERALL COMPLEXITY:</span>
                <span class="{badge_cls}" style="font-size: 1rem;">{final_cls}</span>
            </div>
            <div class="analysis-item">
                <strong>Readability:</strong> {final_obs['readability_note']}
            </div>
            <div class="analysis-item">
                <strong>Vocabulary:</strong> {final_obs['vocab_note']}
            </div>
            <div class="analysis-item">
                <strong>Sentence Structure:</strong> {final_obs['sentence_note']}
            </div>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.get("analysis_done", False) and not text_to_analyze:
    st.error("Input text is empty. Please enter or paste English text to analyze.")
else:
    st.info("Please enter text above and click 'Analyze Text', or choose a sample from the sidebar.")
