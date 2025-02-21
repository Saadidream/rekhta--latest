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

# Custom CSS with enhanced styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
        max-width: 900px;
        margin: 0 auto;
        background: linear-gradient(135deg, #f6f8ff 0%, #ffffff 100%);
    }
    
    /* Header styling */
    .stTitle {
        background: linear-gradient(120deg, #2E3192 0%, #1BFFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 2.5rem !important;
        font-family: 'Georgia', serif;
        padding: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Card container for inputs */
    .input-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
        border: 1px solid rgba(46, 49, 146, 0.1);
    }
    
    /* Input styling */
    .input-label {
        color: #2E3192;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-family: 'Arial', sans-serif;
    }
    
    /* Slider styling */
    .stSlider {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #2E3192 0%, #1BFFFF 100%);
        color: white;
        padding: 0.75rem 2.5rem;
        font-size: 1.2rem;
        border-radius: 15px;
        border: none;
        transition: all 0.3s ease;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        max-width: 300px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(46, 49, 146, 0.2);
    }
    
    /* Poetry output styling */
    .poetry-container {
        background: white;
        padding: 3rem;
        border-radius: 20px;
        margin-top: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(46, 49, 146, 0.1);
    }
    
    .poetry-title {
        color: #2E3192;
        font-size: 1.8rem !important;
        margin-bottom: 1.5rem !important;
        font-family: 'Georgia', serif;
        text-align: center;
    }
    
    .poetry-output {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        font-family: 'Arial', sans-serif;
        line-height: 1.8;
        font-size: 1.1rem;
        color: #2D3748;
        border-left: 4px solid #2E3192;
        margin: 1rem 0;
        white-space: pre-line;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #2E3192;
        padding-top: 3rem;
        font-size: 1rem;
        opacity: 0.8;
    }
    
    /* Decoration elements */
    .decoration {
        position: relative;
        padding: 2rem 0;
    }
    
    .decoration::before,
    .decoration::after {
        content: "✨";
        position: absolute;
        font-size: 1.5rem;
        transform: translateY(-50%);
    }
    
    .decoration::before {
        left: 0;
    }
    
    .decoration::after {
        right: 0;
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

# Main title with decoration
st.markdown('<h1 class="stTitle">✨ Roman-Urdu Poetry Generator ✨</h1>', unsafe_allow_html=True)

# Input card container
st.markdown('<div class="input-card">', unsafe_allow_html=True)

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="input-label">Starting Words</p>', unsafe_allow_html=True)
    seed_text = st.text_input('', 'mohabbat', key='seed_input')

with col2:
    st.markdown('<p class="input-label">Creativity Level</p>', unsafe_allow_html=True)
    temperature = st.slider('', 0.1, 2.0, 0.8, key='temperature_slider')

st.markdown('<p class="input-label">Poem Length</p>', unsafe_allow_html=True)
num_chars = st.slider('', 100, 500, 200, step=50, key='length_slider')

st.markdown('</div>', unsafe_allow_html=True)

# Center the generate button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    generate_button = st.button('Generate Poetry ✨')

if generate_button:
    with st.spinner('Crafting your poetic masterpiece... ✨'):
        generated = generate_poem(
            seed_text.lower(),
            model,
            num_chars=num_chars,
            temperature=temperature
        )
        
        # Poetry output container
        st.markdown('<div class="poetry-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="poetry-title">Your Poetic Creation</h2>', unsafe_allow_html=True)
        st.markdown(f'<div class="poetry-output">{generated}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <div class="decoration">
            Created with 💖 for the Love of Urdu Poetry
        </div>
    </div>
    """, unsafe_allow_html=True)
