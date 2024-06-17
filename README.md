# ChessGPT

This repo contains code for training your own NanoGPT-based models on chess data.

## Getting Started

Follow these steps to set up and train your ChessGPT model:

### Prerequisites

Ensure you have Python 3.7 or later installed.

### Setup Instructions

1. **Clone the Repo**:
   Begin by cloning this repository to your local machine using:
   ```bash
   git clone https://github.com/jmtzt/ChessGPT.git
   cd ChessGPT
   ```

   Also make sure to install the pre-commit hooks by running the following command:
   ```bash
   pre-commit install
   ```

2. **Download the Dataset**:
   Download the chess dataset from [Hugging Face Datasets](https://huggingface.co/datasets/adamkarvonen/chess_games). Place the downloaded files in the `data/lichess` directory.
   For the initial experiments, I've used the `lichess_6gb_blocks` dataset.

3. **Install Dependencies**:
   Install the required Python packages in your virtual env using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data**:
   Run the `src/datamodules.py` script to tokenize and preprocess the data:
   ```bash
   python src/datamodules.py
   ```
   This will save the preprocessed files in binary format in the `data/lichess` directory.

4. **Configure the Model**:
   Adjust the model configuration in `src/train.py` to suit your requirements.

   The training script includes several configuration options you can adjust:
   - **Model Parameters**: Change the model's architecture, initialization, number of attention heads, embedding size, etc.
   - **Data Handling**: Configure how the data is loaded and processed.
   - **Training Parameters**: Set the number of epochs, batch size, learning rate as needed.

   Also make sure to either setup [Weights & Biases](https://docs.wandb.ai/guides/integrations/lightning) or comment out the relevant lines in the training script.

5. **Train the Model**:
   To start training the model, simply execute:
   ```bash
   python src/train.py
   ```
   This script initializes the model, sets up the data module, and begins the training process.

## Acknowledgements

- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [Adam Karvnonen's Hugging Face Dataset](https://huggingface.co/datasets/adamkarvonen/chess_games)
