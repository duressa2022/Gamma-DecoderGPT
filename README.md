# gamma-decoderGPT

![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

**gamma-decoderGPT** is a custom implementation of a GPT-like model for autoregressive text generation, built from scratch using PyTorch. This project focuses on creating a decoder-only Transformer architecture, complete with multi-head attention, positional embeddings, and utilities for subword tokenization and dataset preparation. It’s designed for researchers and developers interested in natural language processing (NLP), language modeling, or building generative models for text.

The project includes a training pipeline with a custom learning rate scheduler, masked loss function, and accuracy metrics, making it suitable for training on custom datasets. It also provides a Flask-based web interface for interactive text generation, allowing users to select model checkpoints and generate text from prompts.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [Training the Model](#training-the-model)
  - [Generating Text via Web Interface](#generating-text-via-web-interface)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Custom GPT Implementation**: A decoder-only Transformer architecture with multi-head attention, feed-forward layers, and positional embeddings, built from scratch.
- **Subword Tokenization**: Uses a custom `SubWordTokenizer` for efficient tokenization, supporting low-resource datasets.
- **Autoregressive Training**: Supports next-token prediction with a masked loss function and accuracy metric for padded sequences.
- **Learning Rate Scheduler**: Implements the Transformer learning rate schedule with warmup and decay, as per the "Attention Is All You Need" paper.
- **Web Interface**: A Flask-based app for interactive text generation, with a dropdown menu to select model checkpoints.
- **Modular Design**: Organized into reusable modules for easy experimentation and extension.

---

## Project Structure

The project is organized into three main directories: `engine` (core implementation), `models` (for saving checkpoints), and `routes`/`service` (for the web interface). Below is the folder structure:

```
gamma-decoderGPT/
├── engine/                          # Core implementation modules
│   ├── __pycache__/                 # Python cache files
│   ├── __init__.py                  # Package initialization
│   ├── build_decoderGPT.py          # Main script to build and train the GPT model
│   ├── decoder_dataset_sub.py       # Dataset class for subword tokenization
│   ├── decoder_gpt_layer.py         # Decoder layer implementation
│   ├── dot_product_attention.py     # Scaled dot-product attention mechanism
│   ├── feed_forward_layer.py        # Feed-forward neural network layer
│   ├── gamma_decoderGPT.py          # Main GPT model implementation
│   ├── gamma_inference.py           # Inference logic for text generation
│   ├── layer_normalization.py       # Layer normalization implementation
│   ├── model_evaluation.py          # Evaluation metrics (e.g., accuracy, loss)
│   ├── multi_head_attention.py      # Multi-head attention mechanism
│   ├── position_embedding.py        # Positional encoding for input embeddings
│   ├── sub_word_tokenizer.py        # Subword tokenization utilities
│   └── word_level_tokenizer.py      # Word-level tokenization utilities
├── models/                          # Directory for saving/loading model checkpoints
├── routes/                          # Directory for API routes
│   └── app.py                       # Flask app for web interface
├── service/                         # Service-related scripts (if applicable)
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

- **`engine/`**: Contains all the core components of the GPT model, including layers, attention mechanisms, tokenization, and training utilities.
- **`models/`**: Stores trained model checkpoints (e.g., `weights_epoch_3.pt`).
- **`routes/`**: Contains the Flask app (`app.py`) for the web interface.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/gamma-decoderGPT.git
   cd gamma-decoderGPT
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` doesn’t exist yet, create it with the following content:
   ```
   torch>=2.0.0
   tokenizers>=0.13.0
   numpy>=1.21.0
   flask>=2.0.0
   ```
   Then run the above command.

4. **Verify Installation**:
   Run a quick test to ensure PyTorch is installed:
   ```python
   import torch
   print(torch.__version__)
   ```

---

## Usage

### Dataset Preparation

The project uses `DecoderDataSetSubWord` to load and tokenize text data. You’ll need a JSON file with text sequences (e.g., `english_data.json`) and a pre-trained tokenizer (e.g., `decoder_token.pkl`).

1. **Prepare Your Dataset**:
   Create a JSON file (`english_data.json`) with a list of text sequences:
   ```json
   [
       "The house is big and beautiful",
       "I went to the park yesterday",
       ...
   ]
   ```

2. **Train the Tokenizer**:
   If you don’t have a pre-trained tokenizer, train one using `SubWordTokenizer`:
   ```python
   from engine.sub_word_tokenizer import SubWordTokenizer

   tokenizer = SubWordTokenizer()
   tokenizer.train(["The house is big", "I went to the park"], vocab_size=10000)
   tokenizer.save("decoder_token.pkl")
   ```

3. **Load the Dataset**:
   The `DecoderDataSetSubWord` class will load and tokenize the data:
   ```python
   from engine.decoder_dataset_sub import DecoderDataSetSubWord

   dataset = DecoderDataSetSubWord(
       fileName="english_data.json",
       decoder_path="decoder_token.pkl"
   )
   ```

### Training the Model

Train the `GammaGPT` model using the provided script `build_decoderGPT.py`:

```bash
python -m engine.build_decoderGPT
```

This script:
- Loads the dataset and tokenizes it.
- Initializes the `GammaGPT` model with the specified hyperparameters.
- Trains the model for 10 epochs, saving checkpoints every 3 epochs in the `models/` directory.

#### Example Output
```
Epoch: 1/10, Batch: 1/10, Loss: 5.1234, Accuracy: 0.1500, LR: 0.0000123
Epoch 1 Summary - Avg Loss: 5.1234, Avg Accuracy: 0.1500
...
Epoch 3 Summary - Avg Loss: 4.5678, Avg Accuracy: 0.2200
Saved model checkpoint to models/weights_epoch_3.pt
```

### Generating Text via Web Interface

The project includes a Flask-based web interface for interactive text generation.

1. **Run the Flask App**:
   ```bash
   python routes/app.py
   ```

2. **Access the Interface**:
   Open your browser and go to `http://localhost:5000`. You’ll see a webpage with:
   - A dropdown menu to select a model checkpoint (e.g., `weights_epoch_3.pt`).
   - A text area to enter a prompt.
   - A button to generate text.

3. **Generate Text**:
   - Select a checkpoint from the dropdown.
   - Enter a prompt (e.g., “The house is big”).
   - Click “Generate Text” to see the model’s output.

#### Example Interaction
- **Prompt**: “The house is big”
- **Generated Text**: “The house is big and beautiful, standing tall near the river.”

---

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- `tokenizers` (Hugging Face) for subword tokenization
- NumPy for numerical operations
- Flask for the web interface

Install them via `requirements.txt` as shown in the [Installation](#installation) section.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Inspired by the "Attention Is All You Need" paper by Vaswani et al. (2017) and the GPT architecture.
- Thanks to the PyTorch and Hugging Face communities for their excellent libraries.
- Special appreciation to contributors working on open-source NLP projects.
