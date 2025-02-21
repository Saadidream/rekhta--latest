import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Urdu Poetry Generator",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS with stable layout
st.markdown("""
    <style>
    /* Base styles */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Title styling */
    .main-title {
        color: #1a365d;
        font-size: 40px;
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
        font-weight: bold;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Input section styling */
    .input-section {
        background-color: #f8fafc;
        padding: 25px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
    }
    
    /* Label styling */
    .input-label {
        color: #2d3748;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    /* Button styling */
    .generate-button {
        background-color: #2c5282;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        font-weight: 600;
        width: 200px;
        margin: 20px auto;
        display: block;
    }
    
    /* Output section styling */
    .output-section {
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        margin-top: 30px;
        border: 1px solid #e2e8f0;
    }
    
    .output-title {
        color: #2d3748;
        font-size: 24px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: 600;
    }
    
    .poem-text {
        font-family: 'Arial', sans-serif;
        font-size: 18px;
        line-height: 1.8;
        white-space: pre-wrap;
        padding: 20px;
        background-color: white;
        border-left: 4px solid #2c5282;
        margin: 10px 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        color: #4a5568;
        font-size: 14px;
    }
    
    /* Error message styling */
    .error-message {
        color: #e53e3e;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        background-color: #fff5f5;
        border: 1px solid #feb2b2;
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

# Main title
st.markdown('<h1 class="main-title">Roman-Urdu Poetry Generator ✨</h1>', unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-section">', unsafe_allow_html=True)

# Input fields
st.markdown('<p class="input-label">Enter Starting Words:</p>', unsafe_allow_html=True)
seed_text = st.text_input('', value='mohabbat', key='seed')

st.markdown('<p class="input-label">Creativity Level:</p>', unsafe_allow_html=True)
temperature = st.slider('', 0.1, 2.0, 0.8, help='Higher values make the output more creative but less predictable')

st.markdown('<p class="input-label">Poem Length:</p>', unsafe_allow_html=True)
num_chars = st.slider('', 100, 500, 200, step=50, help='Number of characters to generate')

st.markdown('</div>', unsafe_allow_html=True)

# Generate button
if st.button('Generate Poetry ✨', key='generate'):
    if model is not None and seed_text.strip():
        with st.spinner('Creating your poem...'):
            generated_poem = generate_poem(
                seed_text.lower(),
                model,
                num_chars=num_chars,
                temperature=temperature
            )
            
            if generated_poem:
                st.markdown('<div class="output-section">', unsafe_allow_html=True)
                st.markdown('<h2 class="output-title">Your Generated Poem</h2>', unsafe_allow_html=True)
                st.markdown(f'<div class="poem-text">{generated_poem}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="error-message">Please enter some starting words for the poem.</div>',
            unsafe_allow_html=True
        )

# Footer
st.markdown("""
    <div class="footer">
        Created with ❤️ for Urdu Poetry Lovers
    </div>
    """, unsafe_allow_html=True)
