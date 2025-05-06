import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('best_model2.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Title and link
st.title('🔮 Shona Next Word Prediction')
st.markdown("[View Google Colab Notebook](https://colab.research.google.com/drive/1XzMg_WWvgZaIdQA7VWbvpz9kXF54lTXg?usp=sharing)", unsafe_allow_html=True)

# Prediction function
def predict_next_word(model, tokenizer, text, num_words=1):
    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences([sequence], maxlen=5, padding='pre')
        predicted_probs = model.predict(sequence, verbose=0)
        predicted = np.argmax(predicted_probs, axis=-1)

        output_word = next((word for word, index in tokenizer.word_index.items() if index == predicted), "")
        text += " " + output_word
    return ' '.join(text.split(' ')[-num_words:])

# Streamlit app logic
def main():
    user_input = st.text_input('Enter exactly **five Shona words**:')
    words = user_input.strip().split()

    if st.button("Predict Next 3 Words"):
        if len(words) == 5:
            result = predict_next_word(model, tokenizer, user_input, num_words=3)
            st.success(f"Predicted words: {result}")
        else:
            st.warning("Please enter exactly **five words**.")

if __name__ == '__main__':
    main()
