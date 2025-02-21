import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Urdu Poetry Generator",
    page_icon="✨",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        max-width: 800px;
        margin: 0 auto;
    }
    .stTitle {
        color: #1E3A8A;
        font-size: 2.5rem !important;
        text-align: center;
        margin-bottom: 2rem !important;
        font-family: 'Georgia', serif;
    }
    .stSubheader {
        color: #1E3A8A;
        font-size: 1.5rem !important;
        margin-top: 2rem !important;
        font-family: 'Georgia', serif;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .poetry-output {
        background-color: #F3F4F6;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 1rem;
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
        border-left: 4px solid #1E3A8A;
    }
    .input-label {
        color: #4B5563;
        font-size: 1.1rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# Load assets
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('urdu_poetry_lstm.h5')

model = load_model()
char_to_idx, idx_to_char = pickle.load(open('mappings.pkl', 'rb'))

# Text generation functions
def generate_poem(seed, model, max_length=100, num_chars=200, temperature=1.0):
    generated = seed
    for _ in range(num_chars):
        x_pred = np.zeros((1, max_length, len(char_to_idx)))
        for t, char in enumerate(generated[-max_length:]):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1.0
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = idx_to_char[next_index]
        generated += next_char
        
        if len(generated) > max_length + num_chars:
            break
            
    formatted = generated.replace(' \n ', '\n').capitalize()
    return formatted

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Streamlit UI with enhanced layout
st.title('✨ Roman-Urdu Poetry Generator ✨')

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="input-label">Enter starting words:</p>', unsafe_allow_html=True)
    seed_text = st.text_input('', 'mohabbat', key='seed_input')

with col2:
    st.markdown('<p class="input-label">Creativity Level:</p>', unsafe_allow_html=True)
    temperature = st.slider('', 0.1, 2.0, 0.8, key='temperature_slider')

st.markdown('<p class="input-label">Poem Length:</p>', unsafe_allow_html=True)
num_chars = st.slider('', 100, 500, 200, step=50, key='length_slider')

# Center the generate button
col1, col2, col3 = st.columns([1,1,1])
with col2:
    generate_button = st.button('Generate Poetry 🎨')

if generate_button:
    with st.spinner('Creating your masterpiece...'):
        generated = generate_poem(
            seed_text.lower(),
            model,
            num_chars=num_chars,
            temperature=temperature
        )
        st.markdown('### Your Generated Poem')
        st.markdown(f'<div class="poetry-output">{generated}</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; color: #6B7280; padding-top: 2rem; font-size: 0.9rem;'>
        Made with ❤️ for Urdu Poetry
    </div>
    """, unsafe_allow_html=True)
