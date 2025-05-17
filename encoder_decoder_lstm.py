# -*- coding: utf-8 -*-
"""
Neural Machine Translation Script (Encoder-Decoder with Attention)
For Kaggle: Configurations are set directly in the Config class.

Features:
- Parallel file dataset loading
- LSTM Encoder-Decoder with Bahdanau Attention
- Adam Optimizer with Exponential Decay Learning Rate Schedule
- Checkpointing (saving the best model based on validation loss)
- Early Stopping
- Greedy Decoding for Inference using tf.while_loop (for graph compatibility)
- BLEU Score Calculation during validation
- Training History Plotting
"""

import tensorflow as tf
import numpy as np
import unicodedata
import re
import os
import io
import time
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nltk
nltk.data.find('tokenizers/punkt')
from nltk.translate.bleu_score import sentence_bleu

# Configuration Class
class Config:
    DATA_PATH_INPUT = '/content/OpenSubtitles.en-ur.en'
    DATA_PATH_TARGET = '/content/OpenSubtitles.en-ur.ur'
    OUTPUT_DIR = '/content/nmt_output'
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    NUM_EXAMPLES = None
    PREPROCESS_RECREATE_TOKENIZERS = False
    TOKENIZER_INPUT_PATH = os.path.join(OUTPUT_DIR, "tokenizer_input.json")
    TOKENIZER_TARGET_PATH = os.path.join(OUTPUT_DIR, "tokenizer_target.json")
    MAX_LENGTH_PATH = os.path.join(OUTPUT_DIR, 'max_lengths.json')

    # Model parameters
    EMBEDDING_DIM = 256
    UNITS = 512
    DROPOUT_RATE = 0.2

    BATCH_SIZE = 32
    EPOCHS = 1
    LEARNING_RATE = 0.001
    LR_DECAY_STEPS = 10000
    LR_DECAY_RATE = 0.96
    VALIDATION_SPLIT = 0.1
    EARLY_STOPPING_PATIENCE = 5
    GRADIENT_CLIP_NORM = 5.0
    CALCULATE_BLEU = True
    BLEU_MAX_SAMPLES = 200

    INFERENCE_BATCH_SIZE = 1
    BEAM_WIDTH = 1
    MODE = 'train'

    DISTRIBUTED_TRAINING = False
    BEST_MODEL_FILENAME_ENCODER = "best_encoder_weights.weights.h5"
    BEST_MODEL_FILENAME_DECODER = "best_decoder_weights.weights.h5"

    CONVERT_TFLITE = False

# --- Data Preparation ---
def unicode_to_ascii(s):
    """Normalizes unicode string to ascii"""
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    """Preprocesses a sentence: lowercase, normalize, add spaces around punctuation, add start/end tokens."""
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[\" \"]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿<>]+\s*", " ", w) # Fixed regex to handle multiple non-alphabetic chars and spaces
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

def create_or_load_tokenizer(lang_texts, tokenizer_path, recreate=False, vocab_size=None):
    """Creates a tokenizer or loads it from a file."""
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True) # Ensure directory exists
    if os.path.exists(tokenizer_path) and not recreate:
        print(f"Loading tokenizer from {tokenizer_path}")
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_json_str = json.load(f)
        lang_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json_str)
    else:
        print(f"Creating and saving tokenizer to {tokenizer_path}")
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size, filters='', oov_token='<unk>'
        )
        lang_tokenizer.fit_on_texts(lang_texts)
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
             json.dump(lang_tokenizer.to_json(), f, ensure_ascii=False, indent=4)
    return lang_tokenizer

def tokenize_and_pad(lang_texts, lang_tokenizer, max_length=None):
    """Tokenizes sentences and pads them to a maximum length."""
    tensor = lang_tokenizer.texts_to_sequences(lang_texts)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_length, padding='post')
    return tensor

def load_dataset(config, for_training=True):
    """Loads dataset from parallel files, preprocesses, tokenizes, and pads."""
    print("Loading data from parallel files...")
    if not config.DATA_PATH_INPUT or not config.DATA_PATH_TARGET:
        raise ValueError("DATA_PATH_INPUT and DATA_PATH_TARGET must be specified in Config.")
    if not os.path.exists(config.DATA_PATH_INPUT):
        raise FileNotFoundError(f"Input data file not found: {config.DATA_PATH_INPUT}")
    if not os.path.exists(config.DATA_PATH_TARGET):
        raise FileNotFoundError(f"Target data file not found: {config.DATA_PATH_TARGET}")

    try:
        with io.open(config.DATA_PATH_INPUT, encoding='UTF-8') as f_in, \
             io.open(config.DATA_PATH_TARGET, encoding='UTF-8') as f_tar:
            input_lines = f_in.read().strip().split('\n')
            target_lines = f_tar.read().strip().split('\n')
    except Exception as e:
        print(f"Error reading data files: {e}")
        raise

    print(f"Input file lines found: {len(input_lines)}")
    print(f"Target file lines found: {len(target_lines)}")

    if len(input_lines) != len(target_lines):
        # Added detail to the error message
        raise ValueError(f"Input and target files must have the same number of lines. Found {len(input_lines)} lines in input and {len(target_lines)} in target.")

    if config.NUM_EXAMPLES:
        input_lines = input_lines[:config.NUM_EXAMPLES]
        target_lines = target_lines[:config.NUM_EXAMPLES]

    print(f"Using {len(input_lines)} sentence pairs (up to NUM_EXAMPLES limit if set).")

    input_lang_processed = [preprocess_sentence(s) for s in input_lines]
    target_lang_processed = [preprocess_sentence(s) for s in target_lines]

    if for_training or not (os.path.exists(config.TOKENIZER_INPUT_PATH) and os.path.exists(config.TOKENIZER_TARGET_PATH)):
        input_tokenizer = create_or_load_tokenizer(input_lang_processed, config.TOKENIZER_INPUT_PATH, config.PREPROCESS_RECREATE_TOKENIZERS)
        target_tokenizer = create_or_load_tokenizer(target_lang_processed, config.TOKENIZER_TARGET_PATH, config.PREPROCESS_RECREATE_TOKENIZERS)
    else:
        print("Loading existing tokenizers for inference...")
        input_tokenizer = create_or_load_tokenizer([], config.TOKENIZER_INPUT_PATH, recreate=False)
        target_tokenizer = create_or_load_tokenizer([], config.TOKENIZER_TARGET_PATH, recreate=False)


    if for_training:
        input_tensor = tokenize_and_pad(input_lang_processed, input_tokenizer)
        target_tensor = tokenize_and_pad(target_lang_processed, target_tokenizer)

        max_length_inp = input_tensor.shape[1]
        max_length_targ = target_tensor.shape[1]

        os.makedirs(os.path.dirname(config.MAX_LENGTH_PATH), exist_ok=True) # Ensure dir exists
        print(f"Saving max lengths ({max_length_inp}, {max_length_targ}) to {config.MAX_LENGTH_PATH}")
        with open(config.MAX_LENGTH_PATH, 'w') as f:
            json.dump({'max_length_inp': max_length_inp, 'max_length_targ': max_length_targ}, f)
        return input_tensor, target_tensor, input_tokenizer, target_tokenizer, max_length_inp, max_length_targ
    else:
        if os.path.exists(config.MAX_LENGTH_PATH):
            print(f"Loading max lengths from {config.MAX_LENGTH_PATH}")
            with open(config.MAX_LENGTH_PATH, 'r') as f:
                max_lengths = json.load(f)
            max_length_inp = max_lengths['max_length_inp']
            max_length_targ = max_lengths['max_length_targ']
        else:
            raise FileNotFoundError(f"Max lengths file not found at {config.MAX_LENGTH_PATH}. Train the model first or provide the file.")
        return input_tokenizer, target_tokenizer, max_length_inp, max_length_targ

# --- Model Components (includes Dropout) ---
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # Use Dropout layer directly
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform') # Dropout handled by separate layer


    def call(self, x, hidden, training=False):
        # Pass Python boolean 'training' to the Dropout layer
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


# --- Optimizer, Loss, and Learning Rate Schedule ---
def get_optimizer(config):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.LEARNING_RATE,
        decay_steps=config.LR_DECAY_STEPS,
        decay_rate=config.LR_DECAY_RATE,
        staircase=True)
    return tf.keras.optimizers.Adam(learning_rate=lr_schedule)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# Loss function for single device - divides by the batch size directly
def loss_function(real, pred, batch_size):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.cast(batch_size, dtype=tf.float32)


# Training & Validation Steps
@tf.function
def train_step(inputs, batch_size, targ_lang_tokenizer, encoder, decoder, optimizer, config):
    inp, targ = inputs
    current_batch_sz = tf.shape(inp)[0]
    enc_hidden = encoder.initialize_hidden_state(batch_size=current_batch_sz)
    max_target_length_tensor = tf.shape(targ)[1]

    with tf.GradientTape() as tape:
        enc_output, enc_h, enc_c = encoder(inp, enc_hidden, training=True)
        dec_hidden = [enc_h, enc_c]
        start_token_id = tf.constant(targ_lang_tokenizer.word_index['<start>'], dtype=tf.int32)
        start_tokens = tf.fill([current_batch_sz, 1], start_token_id)
        dec_input = start_tokens

        initial_loss = tf.constant(0.0)
        initial_time = tf.constant(1, dtype=tf.int32)
        def loop_cond(time, loss, dec_input, dec_hidden_h, dec_hidden_c, max_targ_len, enc_out):
            return time < max_targ_len

        def loop_body(time, loss, dec_input, dec_hidden_h, dec_hidden_c, max_targ_len, enc_out):
            dec_hidden = [dec_hidden_h, dec_hidden_c]
            predictions, next_h, next_c, _ = decoder(dec_input, dec_hidden, enc_out, training=True)
            step_loss = loss_function(targ[:, time], predictions, current_batch_sz)
            loss += step_loss
            dec_input = tf.expand_dims(targ[:, time], 1)
            return time + 1, loss, dec_input, next_h, next_c, max_targ_len, enc_out

        initial_dec_hidden_h = dec_hidden[0]
        initial_dec_hidden_c = dec_hidden[1]

        final_time, total_loss, _, final_dec_hidden_h, final_dec_hidden_c, _, _ = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars=[initial_time, initial_loss, dec_input, initial_dec_hidden_h, initial_dec_hidden_c, max_target_length_tensor, enc_output]
        )
    batch_loss_per_token = total_loss / tf.cast(max_target_length_tensor - 1, tf.float32)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(total_loss, variables)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, config.GRADIENT_CLIP_NORM)
    optimizer.apply_gradients(zip(clipped_gradients, variables))
    return batch_loss_per_token

@tf.function
def valid_step(inputs, batch_size, targ_lang_tokenizer, encoder, decoder, config):
    inp, targ = inputs
    current_batch_sz = tf.shape(inp)[0]
    enc_hidden = encoder.initialize_hidden_state(batch_size=current_batch_sz)
    max_target_length_tensor = tf.shape(targ)[1]
    enc_output, enc_h, enc_c = encoder(inp, enc_hidden, training=False)
    dec_hidden = [enc_h, enc_c]

    start_token_id = tf.constant(targ_lang_tokenizer.word_index['<start>'], dtype=tf.int32)
    start_tokens = tf.fill([current_batch_sz, 1], start_token_id)
    dec_input = start_tokens

    initial_loss = tf.constant(0.0)
    initial_time = tf.constant(1, dtype=tf.int32)

    def loop_cond(time, loss, dec_input, dec_hidden_h, dec_hidden_c, max_targ_len, enc_out):
        return time < max_targ_len

    def loop_body(time, loss, dec_input, dec_hidden_h, dec_hidden_c, max_targ_len, enc_out):
        dec_hidden = [dec_hidden_h, dec_hidden_c]
        predictions, next_h, next_c, _ = decoder(dec_input, dec_hidden, enc_out, training=False)
        step_loss = loss_function(targ[:, time], predictions, current_batch_sz)
        loss += step_loss
        dec_input = tf.expand_dims(targ[:, time], 1)
        dec_hidden = [next_h, next_c]
        return time + 1, loss, dec_input, next_h, next_c, max_targ_len, enc_out

    initial_dec_hidden_h = dec_hidden[0]
    initial_dec_hidden_c = dec_hidden[1]

    final_time, total_loss, _, final_dec_hidden_h, final_dec_hidden_c, _, _ = tf.while_loop(
        loop_cond,
        loop_body,
        loop_vars=[initial_time, initial_loss, dec_input, initial_dec_hidden_h, initial_dec_hidden_c, max_target_length_tensor, enc_output]
    )

    batch_loss_per_token = total_loss / tf.cast(max_target_length_tensor - 1, tf.float32)
    return batch_loss_per_token


# BLEU Score Calculation
def calculate_bleu(dataset, encoder, decoder, inp_tokenizer, targ_tokenizer,
                   max_length_inp, max_length_targ, config, max_samples=100):
    print(f"\nCalculating BLEU score on up to {max_samples} samples...")
    bleu_scores = []
    actual_count = 0
    temp_encoder = Encoder(len(inp_tokenizer.word_index)+1, config.EMBEDDING_DIM, config.UNITS, 1, 0.0)
    temp_decoder = Decoder(len(targ_tokenizer.word_index)+1, config.EMBEDDING_DIM, config.UNITS, 1, 0.0)
    dummy_inp_inf = tf.random.uniform((1, max_length_inp), maxval=len(inp_tokenizer.word_index), dtype=tf.int32)
    dummy_hidden_inf = temp_encoder.initialize_hidden_state(batch_size=1)
    _, _, _ = temp_encoder(dummy_inp_inf, dummy_hidden_inf, training=False)

    dummy_target_inf_input = tf.random.uniform((1, 1), maxval=len(targ_tokenizer.word_index), dtype=tf.int32)
    dummy_enc_output_inf = tf.random.uniform((1, max_length_inp, config.UNITS))
    dummy_dec_hidden_inf = [tf.random.uniform((1, config.UNITS)), tf.random.uniform((1, config.UNITS))]
    _, _, _, _ = temp_decoder(dummy_target_inf_input, dummy_dec_hidden_inf, \
                              dummy_enc_output_inf, training=False)


    # Load the best trained weights into the inference models
    encoder_weights_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_FILENAME_ENCODER)
    decoder_weights_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_FILENAME_DECODER)

    if os.path.exists(encoder_weights_path) and os.path.exists(decoder_weights_path):
        try:
            temp_encoder.load_weights(encoder_weights_path)
            temp_decoder.load_weights(decoder_weights_path)
            print("Best trained weights loaded into inference models for BLEU calculation.")
        except Exception as e:
            print(f"Error loading best model weights for BLEU calculation: {e}")
            pass
    else:
        print(f"Warning: Best model weights not found at {config.CHECKPOINT_DIR}. BLEU calculation will use untrained models.")


    start_time = time.time()
    for inp_batch, targ_batch in dataset:
        for i in range(inp_batch.shape[0]):
            if actual_count >= max_samples: break
            inp_sentence_ids = inp_batch[i].numpy()
            input_sentence = ' '.join([inp_tokenizer.index_word.get(idx, '<unk>') for idx in inp_sentence_ids if idx != 0])
            input_sentence = input_sentence.replace('<start> ', '').replace(' <end>', '')
            if not input_sentence.strip(): continue

            targ_sentence_ids = targ_batch[i].numpy()
            reference = [targ_tokenizer.index_word.get(idx, '<unk>') for idx in targ_sentence_ids
                         if idx != 0 and idx != targ_tokenizer.word_index.get('<start>', -1)
                         and idx != targ_tokenizer.word_index.get('<end>', -1)]
            if not reference: continue

            processed_sentence = preprocess_sentence(input_sentence)
            inputs = [inp_tokenizer.word_index.get(word, inp_tokenizer.word_index.get('<unk>', 0)) for word in processed_sentence.split(' ')]
            inputs = [idx for idx in inputs if idx != 0]

            if not inputs or all(idx == inp_tokenizer.word_index.get('<unk>', 0) for idx in inputs):
                 print(f"Skipping BLEU for sentence '{input_sentence}': input contains only unknown words or is empty.")
                 continue

            inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
            inputs = tf.convert_to_tensor(inputs, dtype=tf.int32)

            enc_hidden = temp_encoder.initialize_hidden_state(batch_size=1)

            translated_ids_tensor = beam_search_translate_graph(
                inputs=inputs,
                enc_hidden=enc_hidden,
                encoder_inf=temp_encoder,
                decoder_inf=temp_decoder,
                inp_lang_word_index=inp_tokenizer.word_index,
                targ_lang_word_index=targ_tokenizer.word_index,
                max_length_targ_train=max_length_targ,
                units=config.UNITS
            )

            translated_words = []
            end_token_id_py = targ_tokenizer.word_index.get('<end>', -1)
            padding_token_id_py = 0
            unk_token_id_py = targ_tokenizer.word_index.get('<unk>', -1)

            if tf.shape(translated_ids_tensor)[0] > 0:
                for token_id in tf.squeeze(translated_ids_tensor).numpy():
                     if token_id == end_token_id_py: break
                     if token_id == padding_token_id_py: continue
                     if token_id in targ_tokenizer.index_word:
                         word = targ_tokenizer.index_word[token_id]
                         translated_words.append(word)
                     elif token_id == unk_token_id_py:
                          word = '<unk>'
                          translated_words.append(word)

            translated_text = " ".join(translated_words)

            translation_tokens = translated_text.split()

            score = sentence_bleu([reference], translation_tokens, weights=(0.25, 0.25, 0.25, 0.25))
            bleu_scores.append(score)
            actual_count += 1

        if actual_count >= max_samples: break

    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    print(f"BLEU Score calculation for {actual_count} samples took {time.time() - start_time:.2f} sec")
    return avg_bleu

# Inference with Greedy Search
@tf.function
def beam_search_translate_graph(inputs, enc_hidden, encoder_inf, decoder_inf,
                                  inp_lang_word_index, targ_lang_word_index,
                                  max_length_targ_train, units):
   
    enc_out, enc_h, enc_c = encoder_inf(inputs, enc_hidden, training=False)
    initial_dec_hidden = [enc_h, enc_c]

    start_token_id = tf.constant(targ_lang_word_index['<start>'], dtype=tf.int32)
    end_token_id = tf.constant(targ_lang_word_index['<end>'], dtype=tf.int32)
    unk_token_id = tf.constant(targ_lang_word_index['<unk>'], dtype=tf.int32)
    padding_token_id = tf.constant(0, dtype=tf.int32) 

    initial_time = tf.constant(0, dtype=tf.int32)
    initial_dec_input = tf.expand_dims([start_token_id], 0)
    initial_translated_tokens = tf.TensorArray(tf.int32, size=max_length_targ_train, dynamic_size=True, clear_after_read=False)
    initial_dec_hidden_h = initial_dec_hidden[0]
    initial_dec_hidden_c = initial_dec_hidden[1]

    def greedy_loop_cond(time, dec_input, dec_h, dec_c, translated_tokens):
        return tf.logical_and(time < max_length_targ_train,
                              tf.not_equal(tf.squeeze(dec_input), end_token_id))


    def greedy_loop_body(time, dec_input, dec_h, dec_c, translated_tokens):
        predictions, next_h, next_c, _ = decoder_inf(dec_input, [dec_h, dec_c], enc_out, training=False)
        predicted_id = tf.argmax(predictions[0], axis=-1, output_type=tf.int32)

        should_write = tf.logical_and(tf.not_equal(predicted_id, end_token_id),
                                      tf.not_equal(predicted_id, padding_token_id))

        # Conditionally write to the TensorArray
        translated_tokens = tf.cond(
            should_write,
            lambda: translated_tokens.write(time, predicted_id),
            lambda: translated_tokens # Do not write if condition is false
        )


        next_dec_input = tf.expand_dims([predicted_id], 0)
        return time + 1, next_dec_input, next_h, next_c, translated_tokens

    final_time, final_dec_input, final_dec_h, final_dec_c, final_translated_tokens = tf.while_loop(
        greedy_loop_cond,
        greedy_loop_body,
        loop_vars=[initial_time, initial_dec_input, initial_dec_hidden_h, initial_dec_hidden_c, initial_translated_tokens]
    )

    translated_ids_tensor = final_translated_tokens.stack()

    return translated_ids_tensor


def translate_sentence_interactive_mode(config, inp_lang, targ_lang, max_len_inp, max_len_targ):
    """Handles interactive translation after training."""
    print("\n--- Interactive Translation Mode ---")
    print("Setting up models for inference...")

    # Create inference models
    vocab_inp_size_inf = len(inp_lang.word_index) + 1
    vocab_tar_size_inf = len(targ_lang.word_index) + 1
    encoder_inference = Encoder(vocab_inp_size_inf, config.EMBEDDING_DIM, config.UNITS, config.INFERENCE_BATCH_SIZE, 0.0)
    decoder_inference = Decoder(vocab_tar_size_inf, config.EMBEDDING_DIM, config.UNITS, config.INFERENCE_BATCH_SIZE, 0.0)

    dummy_inp_inf = tf.zeros((config.INFERENCE_BATCH_SIZE, max_len_inp), dtype=tf.int32)
    dummy_hidden_inf = encoder_inference.initialize_hidden_state(batch_size=config.INFERENCE_BATCH_SIZE)
    _ = encoder_inference(dummy_inp_inf, dummy_hidden_inf, training=False) 

    dummy_target_inf_input = tf.zeros((config.INFERENCE_BATCH_SIZE, 1), dtype=tf.int32)
    dummy_enc_output_inf = tf.zeros((config.INFERENCE_BATCH_SIZE, max_len_inp, config.UNITS), dtype=tf.float32) 
    dummy_dec_hidden_inf = [tf.zeros((config.INFERENCE_BATCH_SIZE, config.UNITS), dtype=tf.float32), tf.zeros((config.INFERENCE_BATCH_SIZE, config.UNITS), dtype=tf.float32)] 
    _, _, _, _ = decoder_inference(dummy_target_inf_input, dummy_dec_hidden_inf, \
                                  dummy_enc_output_inf, training=False) 


    # Load the best trained weights
    encoder_weights_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_FILENAME_ENCODER)
    decoder_weights_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_FILENAME_DECODER)
    if os.path.exists(encoder_weights_path) and os.path.exists(decoder_weights_path):
        print(f"Loading best trained weights from {config.CHECKPOINT_DIR}...")
        try:
            encoder_inference.load_weights(encoder_weights_path)
            decoder_inference.load_weights(decoder_weights_path)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading trained weights: {e}")
            print("Translations will be random unless the model was just trained in this session.")
    else:
        print(f"Warning: No trained weights found at {encoder_weights_path} or {decoder_weights_path}.")
        print("Translations will be random unless the model was just trained in this session.")

    print("\nEnter a sentence to translate (or type 'quit' to exit):")
    while True:
        try:
            sentence_to_translate = input("> ")
            if sentence_to_translate.lower() == 'quit': break
            if sentence_to_translate.strip():
                start_time = time.time()

                processed_sentence = preprocess_sentence(sentence_to_translate)
                inputs = [inp_lang.word_index.get(word, inp_lang.word_index.get('<unk>', 0)) for word in processed_sentence.split(' ')]
                inputs = [idx for idx in inputs if idx != 0] 

                if not inputs or all(idx == inp_lang.word_index.get('<unk>', 0) for idx in inputs):
                     print("Input: <contains only unknown words or is empty>")
                     print("Predicted translation: <translation failed: input contains only unknown words or is empty>")
                     continue

                # Pad the input sequence
                inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_len_inp, padding='post')
                inputs = tf.convert_to_tensor(inputs, dtype=tf.int32) 
                enc_hidden = encoder_inference.initialize_hidden_state(batch_size=1)

                translated_ids_tensor = beam_search_translate_graph(
                    inputs=inputs,
                    enc_hidden=enc_hidden,
                    encoder_inf=encoder_inference,
                    decoder_inf=decoder_inference,
                    inp_lang_word_index=inp_lang.word_index,
                    targ_lang_word_index=targ_lang.word_index,
                    max_length_targ_train=max_len_targ,
                    units=config.UNITS 
                )

                # Convert translated ids to words
                translated_words = []
                end_token_id_py = targ_lang.word_index.get('<end>', -1) 
                padding_token_id_py = 0 
                unk_token_id_py = targ_lang.word_index.get('<unk>', -1)
                if tf.shape(translated_ids_tensor)[0] > 0:
                    for token_id in tf.squeeze(translated_ids_tensor).numpy():
                         if token_id == end_token_id_py: break
                         if token_id == padding_token_id_py: continue
                         if token_id in targ_lang.index_word:
                             word = targ_lang.index_word[token_id]
                             translated_words.append(word)
                         elif token_id == unk_token_id_py:
                              word = '<unk>'
                              translated_words.append(word)

                translated_text = " ".join(translated_words)

                print(f"Input: {sentence_to_translate}")
                print(f"Predicted translation: {translated_text}")
                print(f"Translation took: {time.time() - start_time:.2f} sec")
            else:
                 print("Please enter a sentence.")
        except EOFError: print("\nExiting."); break
        except KeyboardInterrupt: print("\nExiting."); break
        except Exception as e:
            print(f"Translation error: {e}")
            import traceback
            traceback.print_exc()


# --- Utility Functions (Plotting) ---

def plot_training_history(history, config):
    epochs_range = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['loss'], label='Training Loss')
    if any(vl is not None for vl in history['val_loss']):
        valid_epochs = [epochs_range[i] for i, vl in enumerate(history['val_loss']) if vl is not None]
        valid_losses = [vl for vl in history['val_loss'] if vl is not None]
        plt.plot(valid_epochs, valid_losses, label='Validation Loss')
    else:
        plt.text(0.5, 0.5, 'Validation Loss Not Calculated', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Plot Validation BLEU Score
    plt.subplot(1, 2, 2)
    if config.CALCULATE_BLEU and any(h > 0 for h in history.get('bleu', [0])):
         plt.plot(epochs_range, history['bleu'], label='Validation BLEU Score', color='green')
         plt.xlabel('Epoch')
         plt.ylabel('BLEU Score')
         plt.title('Validation BLEU Score')
         plt.legend(loc='lower right')
         plt.grid(True)
    else:
         plt.text(0.5, 0.5, 'BLEU Score Not Calculated\nor was zero', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
         plt.title('Validation BLEU Score')

    plt.tight_layout()
    plot_filename = os.path.join(config.OUTPUT_DIR, 'training_history.png')
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    try:
        plt.savefig(plot_filename)
        print(f"Training history plot saved to {plot_filename}")
    except Exception as e: print(f"Error saving training plot: {e}")
    plt.close()


# --- Training Loop ---
def train_model_single_device(config):
    print("Starting model training process (single device)...\n")

    input_tensor, target_tensor, inp_lang, targ_lang, max_len_inp, max_len_targ = load_dataset(config)

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
        input_tensor, target_tensor, test_size=config.VALIDATION_SPLIT, random_state=42)

    print(f"Training examples: {len(input_tensor_train)}, Validation examples: {len(input_tensor_val)}")

    # Calculate steps per epoch
    batch_size = config.BATCH_SIZE
    print(f"Batch size: {batch_size}")

    buffer_size = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train) // batch_size
    validation_steps = len(input_tensor_val) // batch_size

    if steps_per_epoch == 0:
        print("Error: steps_per_epoch is 0. This means the training dataset size is smaller than the batch size.")
        print(f"Training dataset size: {len(input_tensor_train)}")
        print(f"Batch size: {batch_size}")
        print("Please reduce the batch size or increase the number of training examples.")
        return None # Indicate training cannot proceed

    if validation_steps == 0:
        print("Warning: validation_steps is 0. This means the validation dataset size is smaller than the batch size.")
        print(f"Validation dataset size: {len(input_tensor_val)}")
        print(f"Batch size: {batch_size}")
        print("Validation loss and BLEU score will not be calculated per epoch.")


    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Create TensorFlow Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
    # Use the same batch size for validation
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset_for_bleu = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
    bleu_batch_size = min(config.BATCH_SIZE, 64) 
    val_dataset_for_bleu = val_dataset_for_bleu.batch(bleu_batch_size).prefetch(tf.data.experimental.AUTOTUNE)


    # Instantiate Encoder and Decoder models
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1
    print(f"Input Vocab Size: {vocab_inp_size}, Target Vocab Size: {vocab_tar_size}")
    print(f"Max input length: {max_len_inp}, Max target length: {max_len_targ}")

    encoder = Encoder(vocab_inp_size, config.EMBEDDING_DIM, config.UNITS, config.BATCH_SIZE, config.DROPOUT_RATE)
    decoder = Decoder(vocab_tar_size, config.EMBEDDING_DIM, config.UNITS, config.BATCH_SIZE, config.DROPOUT_RATE)

    build_batch_size = config.BATCH_SIZE

    # Build Encoder
    print(f"Building Encoder with batch size: {build_batch_size} and max_len_inp: {max_len_inp}...")
    dummy_encoder_input = tf.zeros((build_batch_size, max_len_inp), dtype=tf.int32)
    dummy_encoder_hidden = encoder.initialize_hidden_state(batch_size=build_batch_size)
    _ = encoder(dummy_encoder_input, dummy_encoder_hidden, training=False) 
    print("Encoder built.")

    # Build Decoder
    print(f"Building Decoder with batch size: {build_batch_size}, max_len_inp: {max_len_inp}...")
    dummy_decoder_input = tf.zeros((build_batch_size, 1), dtype=tf.int32)
    dummy_encoder_output_for_decoder = tf.zeros((build_batch_size, max_len_inp, config.UNITS), dtype=tf.float32)
    dummy_dec_hidden = [
        tf.zeros((build_batch_size, config.UNITS), dtype=tf.float32), 
        tf.zeros((build_batch_size, config.UNITS), dtype=tf.float32)  
    ]
    _ = decoder(dummy_decoder_input, dummy_dec_hidden, dummy_encoder_output_for_decoder, training=False) 
    print("Decoder built.")
    # --- End of explicit build ---
    optimizer = get_optimizer(config)
    # Setup checkpointing
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


    # Training loop
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True) 
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'loss': [], 'val_loss': [], 'bleu': []}

    print(f"\n--- Starting Training for {config.EPOCHS} epochs ---")

    for epoch in range(config.EPOCHS):
        start_time = time.time()
        total_train_loss = 0.0
        num_train_batches = 0
        train_iterator = iter(train_dataset)

        # Training steps
        for i in range(steps_per_epoch):
            batch_inputs = next(train_iterator)
            loss = train_step(batch_inputs, batch_size, targ_lang, encoder, decoder, optimizer, config) 
            total_train_loss += loss
            num_train_batches += 1
            if num_train_batches % 50 == 0: 
                 print(f'Epoch {epoch+1}/{config.EPOCHS} Batch {num_train_batches}/{steps_per_epoch} Train Loss {loss.numpy():.4f}')

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0


        # Validation iterations
        total_val_loss = 0.0
        num_val_batches = 0
        avg_val_loss = float('inf') # Initialize as infinity

        if validation_steps > 0:
            valid_iterator = iter(val_dataset) 
            for _ in range(validation_steps): 
                batch_inputs = next(valid_iterator)
                # Call the single-device validation step
                loss = valid_step(batch_inputs, batch_size, targ_lang, encoder, decoder, config) # Pass batch_size
                total_val_loss += loss
                num_val_batches +=1
            # Calculate average validation loss for the epoch
            avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        else:
             print("Warning: validation_steps is 0. Skipping validation loss calculation.")


        # Store epoch history
        history['loss'].append(float(avg_train_loss))
        history['val_loss'].append(float(avg_val_loss) if avg_val_loss != float('inf') else None)

        epoch_bleu = 0.0
        if config.CALCULATE_BLEU and validation_steps > 0:
             bleu_val_batches_to_take = (config.BLEU_MAX_SAMPLES + bleu_batch_size -1) // bleu_batch_size # Ceiling division
             bleu_val_batches_to_take = min(bleu_val_batches_to_take, validation_steps * (batch_size // bleu_batch_size)) # Don't exceed total validation data

             if bleu_val_batches_to_take > 0:
                 epoch_bleu = calculate_bleu(
                     val_dataset_for_bleu.take(bleu_val_batches_to_take),
                     encoder, decoder, inp_lang, targ_lang,
                     max_len_inp, max_len_targ, config, max_samples=config.BLEU_MAX_SAMPLES)
             history['bleu'].append(float(epoch_bleu))
             print(f'Epoch {epoch+1} Train Loss {avg_train_loss:.4f} | Validation Loss {avg_val_loss if avg_val_loss != float("inf") else "N/A":.4f} | BLEU {epoch_bleu:.4f}')
        else:
             print(f'Epoch {epoch+1} Train Loss {avg_train_loss:.4f} | Validation Loss {avg_val_loss if avg_val_loss != float("inf") else "N/A":.4f}')
             history['bleu'].append(0.0) 
        print(f'Time taken for epoch {epoch+1}: {time.time() - start_time:.2f} sec\n')
        if avg_val_loss != float('inf'):
            if avg_val_loss < best_val_loss:
                print(f'Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving best model weights...')
                best_val_loss = avg_val_loss
                encoder_save_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_FILENAME_ENCODER)
                decoder_save_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_FILENAME_DECODER)
                encoder.save_weights(encoder_save_path)
                decoder.save_weights(decoder_save_path)
                print(f"Saved weights to {encoder_save_path} and {decoder_save_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f'Validation loss did not improve for {epochs_no_improve} epoch(s).')

            # Early stopping check
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f'Early stopping triggered after epoch {epoch+1}.')
                break
        else:
            print("Validation loss not available, skipping early stopping check for this epoch.")


    print("\n--- Training Finished ---")

    # Save training history
    training_history_path = os.path.join(config.OUTPUT_DIR, 'training_history.json') # Define path
    os.makedirs(os.path.dirname(training_history_path), exist_ok=True)
    try:
        with open(training_history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {training_history_path}")
    except Exception as e:
        print(f"Error saving training history: {e}")

    # Plot training history
    plot_training_history(history, config)
    return inp_lang, targ_lang, max_len_inp, max_len_targ

# --- Main execution logic ---
def run_main():
    cfg = Config() 

    # --- Create Output Directory ---
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"Using output directory: {cfg.OUTPUT_DIR}")

    print("\n--- Configuration ---")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Execution Mode: {cfg.MODE}")
    # Print key configurations
    print(f"  Input Data: {cfg.DATA_PATH_INPUT}")
    print(f"  Target Data: {cfg.DATA_PATH_TARGET}")
    print(f"  Num Examples: {cfg.NUM_EXAMPLES if cfg.NUM_EXAMPLES else 'All'}\n")
    print(f"  Epochs: {cfg.EPOCHS}, Batch Size: {cfg.BATCH_SIZE}")
    print(f"  Distributed Training: {cfg.DISTRIBUTED_TRAINING}")
    print("---------------------\n")

    # No strategy initialization needed for single device

    if cfg.MODE == 'train':
        trained_components = train_model_single_device(cfg)
        if trained_components:
            inp_lang, targ_lang, max_len_inp, max_len_targ = trained_components
            print("\nTraining complete. Models and artifacts saved.")
        else:
            print("\nTraining did not complete successfully or was skipped due to configuration/errors.")

    elif cfg.MODE == 'translate':
        print("\n--- Translation Mode (Run after training) ---")
        print("Loading tokenizers and max lengths...")
        if not all(os.path.exists(p) for p in [cfg.TOKENIZER_INPUT_PATH, cfg.TOKENIZER_TARGET_PATH, cfg.MAX_LENGTH_PATH]):
            print("Error: Tokenizer or max length files not found. Train the model first.")
            return
        inp_lang, targ_lang, max_len_inp, max_len_targ = load_dataset(cfg, for_training=False)
        # Start interactive translation
        translate_sentence_interactive_mode(cfg, inp_lang, targ_lang, max_len_inp, max_len_targ)

    else:
        print(f"Unknown mode: {cfg.MODE}. Set Config.MODE to 'train' or 'translate'.")

# --- Entry Point ---
if __name__ == '__main__':
    run_main()
