import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Load model and mappings
@st.cache_resource
def load_components():
    model = tf.keras.models.load_model('urdu_poetry_lstm.h5')
    with open('mappings.pkl', 'rb') as f:
        char_to_idx, idx_to_char = pickle.load(f)
    return model, char_to_idx, idx_to_char

model, char_to_idx, idx_to_char = load_components()

# Generation function
def generate_poem(seed, num_chars=200, temperature=0.8):
    # ... (same as Colab version above) ...

# Streamlit UI
st.title('Roman-Urdu Poetry Generator 🇵🇰')

col1, col2 = st.columns(2)
with col1:
    seed_text = st.text_input('Start your poem with:', 'mohabbat')
with col2:
    temperature = st.slider('Creativity Level', 0.1, 1.5, 0.7)

generate_length = st.selectbox('Poem Length', [100, 200, 300], index=1)

if st.button('Generate Poetry'):
    if seed_text.strip() == '':
        st.warning('Please enter some starting words!')
    else:
        with st.spinner('Composing your ghazal...'):
            poem = generate_poem(
                seed=seed_text,
                num_chars=generate_length,
                temperature=temperature
            )
        
        st.success('Here's your generated poetry:')
        st.code(poem, language='')
        
        # Add download button
        st.download_button(
            label='Download Poem',
            data=poem,
            file_name='generated_poem.txt',
            mime='text/plain'
        )