"""
English-Urdu Neural Machine Translation Streamlit App

This application uses a trained Encoder-Decoder LSTM model with Bahdanau Attention 
for English to Urdu translation.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import unicodedata
import re
import os
import json
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="English-Urdu Neural Machine Translation",
    page_icon="🌐",
    layout="wide"
)

# Function to normalize unicode strings
def unicode_to_ascii(s):
    """Normalizes unicode string to ascii"""
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    """Preprocesses a sentence: lowercase, normalize, add spaces around punctuation, add start/end tokens."""
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[\" \"]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿<>]+\s*", " ", w)
    w = w.strip()
    # Ensure start and end tokens are added only once and correctly
    if not w.startswith('<start>'):
        w = '<start> ' + w
    # Handle cases where <end> might already be present due to preprocessing
    if w.endswith('<end>'):
        # Remove potentially duplicated <end> before adding a single one
        w = w[:-len('<end>')].strip()
    if not w.endswith('<end>'):
        w = w + ' <end>'
    return w

# Model components definition
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden, training=False):
        x = self.embedding(x)
        x = self.dropout_layer(x, training=training)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        return output, state_h, state_c

    def initialize_hidden_state(self, batch_size=None):
        current_batch_size = batch_size if batch_size is not None else self.batch_sz
        return [tf.zeros((current_batch_size, self.enc_units)),
                tf.zeros((current_batch_size, self.enc_units))]

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        expanded_query = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(expanded_query)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, dropout_rate=0.0):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output, training=False):
        context_vector, attention_weights = self.attention(hidden[0], enc_output)
        x = self.embedding(x)
        x = self.dropout_layer(x, training=training)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        predictions = self.fc(output)
        return predictions, state_h, state_c, attention_weights

# Translation function
@tf.function
def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    # Process input sentence
    processed_sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index.get(word, inp_lang.word_index.get('<unk>', 0)) 
              for word in processed_sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs, dtype=tf.int32)
    
    # Initialize encoder hidden state (batch size 1)
    enc_hidden = encoder.initialize_hidden_state(batch_size=1)
    
    # Run encoder
    enc_output, enc_h, enc_c = encoder(inputs, enc_hidden, training=False)
    dec_hidden = [enc_h, enc_c]
    
    # Start with start token
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    
    result = []
    attention_weights_list = []
    
    # Decode one step at a time
    for t in range(max_length_targ):
        predictions, dec_h, dec_c, attention_weights = decoder(dec_input, dec_hidden, enc_output, training=False)
        
        # Store attention weights for visualization
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_weights_list.append(attention_weights.numpy())
        
        # Get predicted ID
        predicted_id = tf.argmax(predictions[0]).numpy()
        
        # Stop if end token is predicted
        if predicted_id == targ_lang.word_index.get('<end>', 0):
            break
            
        # Add predicted word to result
        if predicted_id in targ_lang.index_word:
            result.append(targ_lang.index_word[predicted_id])
            
        # Use predicted ID as next input
        dec_input = tf.expand_dims([predicted_id], 0)
        dec_hidden = [dec_h, dec_c]
        
    return result, attention_weights_list, inputs.numpy()[0]

def load_tokenizer(json_path):
    """Loads a tokenizer from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

def plot_attention(attention, input_sentence, predicted_sentence, ax):
    """Plots the attention weights."""
    # Clean up input sentence for display
    if isinstance(input_sentence, list):
        input_sentence = ' '.join(input_sentence)
    input_sentence = input_sentence.replace('<start>', '').replace('<end>', '').strip()
    
    # Clean up predicted sentence for display
    predicted_sentence = ' '.join(predicted_sentence)
    
    # Create the heatmap
    attention = attention[:len(predicted_sentence.split(' ')), :len(input_sentence.split(' '))]
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 10}
    
    ax.set_xticklabels([''] + input_sentence.split(' '), fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence.split(' '), fontdict=fontdict)
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title('Attention Weights')

def plot_training_history(history_path):
    """Plots the training history from the saved JSON file."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs_range = range(1, len(history['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Training and Validation Loss
    ax1.plot(epochs_range, history['loss'], label='Training Loss')
    
    # Filter out None values for validation loss
    val_loss = [vl for vl in history['val_loss'] if vl is not None]
    if val_loss:
        ax1.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Plot BLEU Score
    if any(h > 0 for h in history.get('bleu', [0])):
        ax2.plot(epochs_range, history['bleu'], label='Validation BLEU Score', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('BLEU Score')
        ax2.set_title('Validation BLEU Score')
        ax2.legend(loc='lower right')
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'BLEU Score Not Calculated\nor was zero', 
                horizontalalignment='center', 
                verticalalignment='center', 
                transform=ax2.transAxes)
        ax2.set_title('Validation BLEU Score')
    
    plt.tight_layout()
    return fig

def main():
    st.title("English-Urdu Neural Machine Translation")
    st.write("""
    This application translates English text to Urdu using a Sequence-to-Sequence
    model with attention mechanism. The model was trained on OpenSubtitles dataset.
    """)
    
    # Sidebar for model information and configuration
    with st.sidebar:
        st.title("Model Information")
        st.write("Encoder-Decoder LSTM with Bahdanau attention")
        st.write("Trained on OpenSubtitles English-Urdu parallel corpus")
        
        model_folder = st.text_input("Model Directory", "model", 
                                    help="Directory containing model weights and tokenizers")
        
        show_training_history = st.checkbox("Show Training History", True)
        show_attention = st.checkbox("Show Attention Visualization", True)
        
        st.subheader("Model Parameters")
        st.write("- Embedding Dim: 256")
        st.write("- LSTM Units: 512")
        st.write("- Dropout Rate: 0.2")
    
    # Check if model files exist
    model_path = Path(model_folder)
    required_files = [
        model_path / "best_encoder_weights.weights.h5",
        model_path / "best_decoder_weights.weights.h5",
        model_path / "tokenizer_input.json",
        model_path / "tokenizer_target.json",
        model_path / "max_lengths.json",
        model_path / "training_history.json"
    ]
    
    files_exist = all(f.exists() for f in required_files)
    
    if not files_exist:
        st.error("""
        Model files not found! Please ensure the following files are in the model directory:
        - best_encoder_weights.weights.h5
        - best_decoder_weights.weights.h5
        - tokenizer_input.json
        - tokenizer_target.json
        - max_lengths.json
        - training_history.json
        """)
        return
    
    # Load tokenizers
    inp_lang = load_tokenizer(model_path / "tokenizer_input.json")
    targ_lang = load_tokenizer(model_path / "tokenizer_target.json")
    
    # Load max lengths
    with open(model_path / "max_lengths.json", 'r') as f:
        max_lengths = json.load(f)
    max_length_inp = max_lengths['max_length_inp']
    max_length_targ = max_lengths['max_length_targ']
    
    # Create the models
    # For inference: batch size 1, dropout 0
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1
    
    embedding_dim = 256
    units = 512
    batch_sz = 1  # For inference
    
    encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_sz)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_sz)
    
    # Load model weights
    encoder.load_weights(model_path / "best_encoder_weights.weights.h5")
    decoder.load_weights(model_path / "best_decoder_weights.weights.h5")
    
    # Initialize models with dummy data to build them
    dummy_input = tf.zeros((batch_sz, max_length_inp), dtype=tf.int32)
    dummy_hidden = encoder.initialize_hidden_state()
    dummy_enc_output, _, _ = encoder(dummy_input, dummy_hidden)
    
    dummy_dec_input = tf.zeros((batch_sz, 1), dtype=tf.int32)
    dummy_dec_hidden = [tf.zeros((batch_sz, units)), tf.zeros((batch_sz, units))]
    _, _, _, _ = decoder(dummy_dec_input, dummy_dec_hidden, dummy_enc_output)
    
    st.write("✅ Model loaded successfully.")
    
    # Show training history
    if show_training_history:
        st.subheader("Training History")
        history_fig = plot_training_history(model_path / "training_history.json")
        st.pyplot(history_fig)
    
    # Translation interface
    st.header("Translation")
    
    # Example sentences dropdown
    example_sentences = [
        "Hello, how are you?",
        "What is your name?",
        "I love learning new languages.",
        "The weather is nice today.",
        "Can you help me with this translation?"
    ]
    
    selected_example = st.selectbox("Choose an example or type your own:", 
                                   [""] + example_sentences)
    
    # Text input
    user_input = st.text_area("Enter English text to translate:", 
                             value=selected_example if selected_example else "",
                             height=100)
    
    col1, col2 = st.columns(2)
    with col1:
        translate_button = st.button("Translate", type="primary")
    
    # Perform translation when button is clicked
    if translate_button and user_input.strip():
        with st.spinner("Translating..."):
            start_time = time.time()
            
            try:
                # Get translation and attention weights
                translated_words, attention_weights, input_indices = translate(
                    user_input, encoder, decoder, inp_lang, targ_lang, 
                    max_length_inp, max_length_targ
                )
                
                translation_time = time.time() - start_time
                
                # Display input and output
                st.subheader("Translation Result")
                
                # Display the translation
                if translated_words:
                    translated_text = " ".join(translated_words)
                    st.markdown(f"**Urdu:** {translated_text}")
                else:
                    st.error("Translation failed. The model couldn't generate a valid translation.")
                
                st.write(f"Translation completed in {translation_time:.2f} seconds")
                
                # Display attention visualization
                if show_attention and translated_words:
                    st.subheader("Attention Visualization")
                    
                    # Convert input indices to words for visualization
                    input_words = []
                    for idx in input_indices:
                        if idx != 0:  # Skip padding
                            if idx in inp_lang.index_word:
                                word = inp_lang.index_word[idx]
                                if word not in ['<start>', '<end>']:
                                    input_words.append(word)
                    
                    # Create attention plot
                    attention_plot = np.zeros((len(translated_words), len(input_words)))
                    for i, attention in enumerate(attention_weights[:len(translated_words)]):
                        attention_plot[i, :len(input_words)] = attention[:len(input_words)]
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plot_attention(attention_plot, input_words, translated_words, ax)
                    st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error during translation: {str(e)}")
                st.exception(e)
    
    # Information about the application
    st.header("About")
    st.write("""
    This application demonstrates Neural Machine Translation using the Encoder-Decoder architecture
    with attention mechanism. The model was trained on OpenSubtitles English-Urdu parallel corpus.
    
    **Model Architecture:**
    - Encoder: LSTM with Embedding Layer
    - Attention: Bahdanau Attention Mechanism
    - Decoder: LSTM with Attention and Dense Output Layer
    
    For more details about the model architecture and training process, refer to the source code
    or the accompanying documentation.
    """)

if __name__ == "__main__":
    main()