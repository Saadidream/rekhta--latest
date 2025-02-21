import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Set page config FIRST (remove duplicate)
st.set_page_config(
    page_title="Roman-Urdu Poetry Generator",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=Poppins:wght@300;500&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
    }
    
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #2a2a2a;
    }
    
    .stTextInput>div>div>input {
        border: 1px solid #6c757d;
        border-radius: 8px;
        padding: 0.5rem;
        font-family: 'Poppins', sans-serif;
    }
    
    .stSlider [data-baseweb="slider"] {
        color: #4a148c;
    }
    
    .stButton>button {
        border: 2px solid #4a148c;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(45deg, #4a148c, #6a1b9a);
        color: white;
        transition: all 0.3s;
        font-weight: 500;
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
        min-height: 300px;
    }
    
    .poem-text {
        font-size: 1.1rem;
        color: #2a2a2a;
        margin-top: 1rem;
    }
    
    footer {
        text-align: center;
        padding: 2rem;
        font-family: 'Poppins', sans-serif;
        color: #6c757d;
        margin-top: 2rem;
        border-top: 1px solid #e9ecef;
    }
    
    .title-container {
        text-align: center;
        margin-bottom: 3rem;
        padding: 1rem;
        background: linear-gradient(135deg, rgba(74, 20, 140, 0.1) 0%, rgba(106, 27, 154, 0.1) 100%);
        border-radius: 15px;
    }
    
    .expander-content {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load assets with error handling
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('urdu_poetry_lstm.h5')
    except Exception as e:
        st.error("Error loading model. Please check if the model file exists.")
        return None

try:
    model = load_model()
    char_to_idx, idx_to_char = pickle.load(open('mappings.pkl', 'rb'))
except Exception as e:
    st.error("Error loading character mappings. Please check if the mappings file exists.")
    model = None
    char_to_idx, idx_to_char = None, None

# Text generation functions
def generate_poem(seed, model, max_length=100, num_chars=200, temperature=1.0):
    try:
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
    except Exception as e:
        st.error(f"Error generating poem: {str(e)}")
        return None

def sample(preds, temperature=1.0):
    try:
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    except Exception as e:
        st.error(f"Error in sampling: {str(e)}")
        return None

# Streamlit UI
st.markdown('<div class="title-container">', unsafe_allow_html=True)
st.markdown("<h1>Roman-Urdu Poetry Generator ✨</h1>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    seed_text = st.text_input(
        '**Enter starting words:**',
        'mohabbat',
        help="Begin your poem with meaningful Roman-Urdu words"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("Advanced Settings", expanded=True):
        st.markdown('<div class="expander-content">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("<div class='poem-output'>", unsafe_allow_html=True)
    if st.button('Generate Poetry ✨', use_container_width=True):
        if model is not None and seed_text.strip():
            with st.spinner('Crafting your masterpiece...'):
                generated = generate_poem(
                    seed_text.lower(),
                    model,
                    num_chars=num_chars,
                    temperature=temperature
                )
                if generated:
                    st.subheader('Your Generated Poetry:')
                    st.markdown(f"<div class='poem-text'>{generated}</div>", unsafe_allow_html=True)
        else:
            st.error("Please ensure you've entered starting words and the model is properly loaded.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <footer>
        Crafted with ❤️ by PoetryAI | v1.0.0
    </footer>
""", unsafe_allow_html=True)
