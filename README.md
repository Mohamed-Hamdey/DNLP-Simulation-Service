# DNLP Simulation Service

This repository contains the **DNLP Simulation Service**, a machine learning-based service extraction and simulation framework for natural language processing tasks. The model, built with PyTorch and Transformers, leverages BERT embeddings and multi-head attention for robust service classification and information extraction.

## Features

- **Service Extraction**: Extracts service-related details from unstructured text.
- **Simulation**: Simulates customer interactions and service response times.
- **Customizable NLP Pipeline**: Built with flexible NLP components for tailored use cases.
- **Cloud-Ready**: Compatible with cloud platforms and frameworks like Kaggle and Colab for cloud training and fine-tuning.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Mohamed-Hamdey/DNLP-Simulation-Service.git
cd DNLP-Simulation-Service
pip install -r requirements.txt
```

Alternatively, install the package directly:

```bash
pip install DNLP-Simulation-Service
```

## Usage

### Training

To train the model, prepare a dataset in JSON format and run:

```python
from DNLP_Simulation_Service.trainer import NLPService
from DNLP_Simulation_Service.config import ModelConfig

config = ModelConfig(...)
service = NLPService(config=config)
service.train(train_data="path/to/dataset.json")
```

### Prediction

Once trained, make predictions using:

```python
text = "Example service description text."
prediction = service.predict(text)
print(prediction)
```

### Cloud Training

To train on Kaggle or Colab:
1. Upload the repository files to the environment.
2. Use the provided `trainer.py` script to start training on cloud GPUs.

## API Reference

### Configuration

The `ModelConfig` class allows adjustments to:
- `embedding_dim`
- `hidden_dim`
- `num_heads`
- `dropout`

### Data Preparation

Use `DataProcessor` for preprocessing data and preparing labels.

## Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature-branch`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push the branch (`git push origin feature-branch`)
5. Open a pull request

## License

This project is licensed under the MIT License.
