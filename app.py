import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Set page configuration for a wider, professional look
st.set_page_config(page_title="Roman-Urdu Poetry Generator", page_icon="✨", layout="wide")

# Custom CSS styling for a beautiful frontend
st.markdown(
    """
    <style>
    /* Page background and overall styling */
    .stApp {
        background: linear-gradient(to right, #ece9e6, #ffffff);
    }
    h1 {
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
    }
    /* Sidebar styling */
    .css-1d391kg {  /* Sidebar container */
        background: #f8f9fa;
    }
    /* Input widgets spacing */
    .stTextInput, .stSlider {
        margin-bottom: 20px;
    }
    /* Button styling */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    /* Generated poem container */
    .generated-poem {
        white-space: pre-wrap;
        background: #fff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 0px 5px 2px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True
)

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

    # Format output with line breaks
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
st.title('Roman-Urdu Poetry Generator ✨')

# Use columns to nicely arrange input fields side by side
col1, col2 = st.columns(2)
with col1:
    seed_text = st.text_input('Enter starting words:', 'mohabbat')
with col2:
    temperature = st.slider('Creativity Level:', 0.1, 2.0, 0.8)

num_chars = st.slider('Poem Length:', 100, 500, 200)

if st.button('Generate Poetry'):
    generated = generate_poem(
        seed_text.lower(),
        model,
        num_chars=num_chars,
        temperature=temperature
    )
    st.subheader('Generated Poem:')
    st.markdown(f"<div class='generated-poem'>{generated}</div>", unsafe_allow_html=True)
