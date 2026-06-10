# English-Urdu Neural Machine Translation

This project is a Streamlit demo for English-to-Urdu neural machine translation using an encoder-decoder LSTM architecture with Bahdanau attention.

## Features

- Translates English sentences into Urdu.
- Uses an encoder-decoder recurrent neural network architecture.
- Implements Bahdanau attention for interpretable token alignment.
- Provides a Streamlit interface for interactive translation.
- Supports attention visualization and training-history display when model artifacts are available.

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Matplotlib
- NLTK
- scikit-learn

## Repository Structure

```text
app.py                    Streamlit translation interface
encoder_decoder_lstm.py   Model architecture and translation logic
setup_model.py            Model artifact setup helper
requirements.txt          Python dependencies
nmt_output/               Generated or downloaded model outputs
```

## Model Artifacts

The app expects trained model artifacts in a `model/` directory:

```text
model/
  best_encoder_weights.weights.h5
  best_decoder_weights.weights.h5
  tokenizer_input.json
  tokenizer_target.json
  max_lengths.json
  training_history.json
```

These files should come from the training workflow or a previously trained Kaggle run.

## Installation

```bash
git clone https://github.com/attaquarks/NMT-en-ur.git
cd NMT-en-ur
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Usage

1. Enter an English sentence.
2. Click translate.
3. Review the Urdu output.
4. Enable attention visualization if the required model metadata is available.

## Project Status

This repository demonstrates sequence-to-sequence NLP with attention in a user-facing translation demo. It is useful for understanding machine translation model structure, inference artifacts, and interface integration.
