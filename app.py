import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

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

seed_text = st.text_input('Enter starting words:', 'mohabbat')
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
    st.text(generated)
