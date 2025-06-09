# Molecular Property Prediction

A deep learning model for predicting molecular properties using SMILES strings. The model combines a Transformer encoder with an MLP for property prediction.

## Project Structure

```
.
├── Model/
│   ├── transformer_encoder.py  # Transformer encoder implementation
│   ├── mlp.py                 # MLP implementation
│   ├── trainer.py             # Training utilities
│   └── utils.py              # Model utilities
├── Preprocess/
│   └── utils.py              # Data preprocessing utilities
└── requirements.txt          # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Preprocess your SMILES data:
```python
from Preprocess.utils import SMILESTokenizerBuilder, RawDataLoader

# Initialize tokenizer
tokenizer = SMILESTokenizerBuilder()
# Load and preprocess data
loader = RawDataLoader("your_data.csv")
```

2. Train the model:
```python
from Model.transformer_encoder import TransformerEncoder
from Model.mlp import MLP
from Model.trainer import Trainer

# Initialize models
transformer = TransformerEncoder(...)
mlp = MLP(...)
model = CombinedModel(transformer, mlp)

# Train
trainer = Trainer(model, criterion, optimizer)
trainer.train(train_loader, num_epochs=100)
```

## Model Architecture

- **Transformer Encoder**: Processes SMILES sequences
- **MLP**: Predicts molecular properties from transformer outputs
- **Combined Model**: Integrates transformer and MLP

## License

Private - All rights reserved 