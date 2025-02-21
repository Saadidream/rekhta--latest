import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Set page config FIRST
st.set_page_config(
    page_title="Roman-Urdu Poetry Generator",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling (must come AFTER set_page_config)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=Poppins:wght@300;500&display=swap');
    
    .main {background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);}
    
    h1, h2, h3 {font-family: 'Playfair Display', serif; color: #2a2a2a;}
    
    .stTextInput>div>div>input {border: 1px solid #6c757d; border-radius: 8px;}
    
    .stSlider [data-baseweb="slider"] {color: #4a148c;}
    
    .stButton>button {
        border: 2px solid #4a148c;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(45deg, #4a148c, #6a1b9a);
        color: white;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        border-color: #7b1fa2;
        background: linear-gradient(45deg, #6a1b9a, #8e24aa);
    }
    
    .poem-output {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-family: 'Poppins', sans-serif;
        white-space: pre-wrap;
        line-height: 1.8;
    }
    
    footer {text-align: center; padding: 1rem; font-family: 'Poppins'; color: #6c757d;}
    </style>
""", unsafe_allow_html=True)

# Rest of your code follows...

# Page configuration
st.set_page_config(
    page_title="Roman-Urdu Poetry Generator",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load assets
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('urdu_poetry_lstm.h5')

model = load_model()
char_to_idx, idx_to_char = pickle.load(open('mappings.pkl', 'rb'))

# Text generation functions (unchanged)
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

# Streamlit UI
st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Roman-Urdu Poetry Generator ✨</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    seed_text = st.text_input(
        '**Enter starting words:**',
        'mohabbat',
        help="Begin your poem with meaningful Roman-Urdu words"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("Advanced Settings", expanded=True):
        temperature = st.slider(
            '**Creativity Level:**',
            0.1, 2.0, 0.8,
            help="Higher values produce more creative but less predictable results"
        )
        num_chars = st.slider(
            '**Poem Length:**',
            100, 500, 200,
            help="Total number of characters in the generated poem"
        )

with col2:
    st.markdown("<div class='poem-output' id='output-container'>", unsafe_allow_html=True)
    if st.button('Generate Poetry ✨', use_container_width=True):
        with st.spinner('Crafting your masterpiece...'):
            generated = generate_poem(
                seed_text.lower(),
                model,
                num_chars=num_chars,
                temperature=temperature
            )
            st.subheader('Your Generated Poetry:')
            st.markdown(f"<div class='poem-text'>{generated}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <footer>
        Crafted with ❤️ by PoetryAI | v1.0.0
    </footer>
""", unsafe_allow_html=True)
