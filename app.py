import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .prediction-card {
        background: white;
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
    }

    .prediction-word {
        font-size: 2.7rem;
        font-weight: 800;
        color: #4f46e5;
        text-align: center;
        padding: 0.8rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 1rem;
    }

    .info-card {
        background: rgba(255,255,255,0.85);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 3rem;
        font-size: 0.85rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 12px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model_new.h5")


@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pickle", "rb") as handle:
        return pickle.load(handle)


@st.cache_resource
def load_max_len():
    with open("max_len.pickle", "rb") as handle:
        return pickle.load(handle)


try:
    model = load_lstm_model()
    tokenizer = load_tokenizer()
    max_len = int(load_max_len())
except Exception as e:
    st.error("Unable to load the model files.")
    st.code(str(e))
    st.stop()


# ---------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------
def predict_next_words(text, top_k=5):
    """Return top-k next-word predictions and probabilities."""

    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        return []

    # The model was trained with sequences padded to max_len.
    padded = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre",
        truncating="pre"
    )

    prediction = model.predict(padded, verbose=0)[0]

    # Get top-k token IDs.
    top_indices = np.argsort(prediction)[-top_k:][::-1]

    index_word = tokenizer.index_word

    results = []
    for idx in top_indices:
        word = index_word.get(int(idx))
        if word:
            results.append((word, float(prediction[idx])))

    return results


def generate_text(seed_text, num_words):
    """Generate text one word at a time."""
    generated = seed_text.strip()

    for _ in range(num_words):
        predictions = predict_next_words(generated, top_k=1)

        if not predictions:
            break

        next_word = predictions[0][0]
        generated += " " + next_word

    return generated


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Model Information")
    st.markdown("---")

    st.metric("Vocabulary Size", f"{len(tokenizer.word_index):,}")
    st.metric("Maximum Sequence Length", max_len)

    try:
        lstm_units = model.layers[-2].units
    except Exception:
        lstm_units = "128"

    st.metric("LSTM Units", lstm_units)

    st.markdown("---")
    st.markdown("### ⚙️ How it works")
    st.markdown("""
    1. Your text is converted into token IDs.
    2. The sequence is padded to the model's input length.
    3. The LSTM processes the sequence.
    4. Softmax produces probabilities for the vocabulary.
    5. The most probable next word is selected.
    """)

    st.markdown("---")
    st.caption("Built with TensorFlow + Streamlit")


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🧠 LSTM Next Word Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict what word should come next using your trained LSTM language model.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Main input
# ---------------------------------------------------------
col1, col2 = st.columns([2.2, 1])

with col1:
    st.markdown('<div class="section-title">✍️ Enter your text</div>',
                unsafe_allow_html=True)

    text_input = st.text_area(
        "Input text",
        placeholder="Start typing a sentence... e.g. Artificial intelligence is",
        height=180,
        label_visibility="collapsed"
    )

with col2:
    st.markdown('<div class="section-title">🎯 Prediction Settings</div>',
                unsafe_allow_html=True)

    top_k = st.slider(
        "Number of predictions",
        min_value=1,
        max_value=10,
        value=5
    )

    generate_count = st.slider(
        "Words for text generation",
        min_value=1,
        max_value=20,
        value=5
    )

    st.write("")


# ---------------------------------------------------------
# Buttons
# ---------------------------------------------------------
button_col1, button_col2, button_col3 = st.columns([1, 1, 2])

with button_col1:
    predict_clicked = st.button(
        "🔮 Predict Next Word",
        use_container_width=True,
        type="primary"
    )

with button_col2:
    generate_clicked = st.button(
        "✨ Generate Text",
        use_container_width=True
    )


# ---------------------------------------------------------
# Next-word prediction
# ---------------------------------------------------------
if predict_clicked:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        predictions = predict_next_words(text_input, top_k)

        if not predictions:
            st.warning(
                "None of the entered words were found in the model vocabulary."
            )
        else:
            best_word, best_probability = predictions[0]

            st.markdown(
                '<div class="prediction-card">',
                unsafe_allow_html=True
            )

            st.markdown("### 🎯 Predicted Next Word")

            st.markdown(
                f'<div class="prediction-word">{best_word}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"<center>Confidence: <b>{best_probability * 100:.2f}%</b></center>",
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### 📊 Top Predictions")

            for rank, (word, probability) in enumerate(predictions, 1):
                c1, c2, c3 = st.columns([0.6, 2, 2])

                with c1:
                    st.write(f"**{rank}**")

                with c2:
                    st.write(f"**{word}**")

                with c3:
                    st.progress(probability)
                    st.caption(f"{probability * 100:.2f}%")


# ---------------------------------------------------------
# Text generation
# ---------------------------------------------------------
if generate_clicked:
    if not text_input.strip():
        st.warning("Please enter some starting text first.")
    else:
        with st.spinner("Generating text..."):
            generated = generate_text(text_input, generate_count)

        st.markdown("### ✨ Generated Text")

        st.markdown(
            f"""
            <div class="prediction-card">
                <p style="font-size: 1.15rem; line-height: 1.8;">
                    {generated}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    '<div class="footer">'
    'LSTM Language Model • Next Word Prediction • TensorFlow + Streamlit'
    '</div>',
    unsafe_allow_html=True
)