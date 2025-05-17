English-Urdu Neural Machine Translation
This Streamlit application demonstrates Neural Machine Translation from English to Urdu using an Encoder-Decoder LSTM model with Bahdanau Attention mechanism.

Features
Translation from English to Urdu
Attention mechanism visualization
Training history visualization
Easy-to-use web interface

Setup Instructions
Prerequisites
Python 3.8+
pip package manager
Installation
Clone this repository or download the files:
bash
git clone <repository-url>
cd english-urdu-nmt
Create a virtual environment (recommended):
bash
python -m venv venv
Activate the virtual environment:
On Windows:
bash
venv\Scripts\activate
On macOS/Linux:
bash
source venv/bin/activate
Install the required packages:
bash
pip install -r requirements.txt
Prepare the Model Files
Create a model directory and place the following files inside:

best_encoder_weights.weights.h5
best_decoder_weights.weights.h5
tokenizer_input.json
tokenizer_target.json
max_lengths.json
training_history.json
These files should be the ones you downloaded after training your model on Kaggle.

Running the Application
Run the Streamlit app with:

bash
streamlit run app.py
This will start the application and open it in your default web browser.

Using the Application
Enter English text in the input field or select from the example sentences.
Click the "Translate" button.
View the Urdu translation result.
If enabled, you can also see the attention visualization showing which words the model focused on during translation.
Model Architecture
Encoder: LSTM with Embedding Layer
Attention: Bahdanau Attention Mechanism
Decoder: LSTM with Attention and Dense Output Layer
The model was trained on OpenSubtitles English-Urdu parallel corpus with the following parameters:

Embedding Dimension: 256
LSTM Units: 512
Dropout Rate: 0.2
Troubleshooting
If you encounter any errors related to model loading, ensure that all the model files are correctly placed in the model directory.
If you see CUDA/GPU related errors, try running with CPU only by setting the environment variable: export CUDA_VISIBLE_DEVICES=-1 (Linux/Mac) or set CUDA_VISIBLE_DEVICES=-1 (Windows)
