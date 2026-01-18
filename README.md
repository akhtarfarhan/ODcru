# Sentiment Analysis Project (ODCRU)

A machine learning project for sentiment analysis using both Naive Bayes and Deep Learning models. This project includes data preprocessing, model training, evaluation, and a FastAPI-based REST API for predictions.

## Project Overview

This project implements sentiment analysis with two distinct approaches:
- **Naive Bayes**: A probabilistic classifier using TF-IDF vectorization
- **Deep Learning**: A neural network model using TensorFlow/Keras

Both models are trained on sentiment data and can classify text into three categories:
- Negative
- Neutral
- Positive

## Project Structure

```
ODCRU/
├── api/                          # FastAPI REST API
│   └── app.py                    # API endpoints and model serving
├── src/                          # Source code modules
│   ├── config.py                 # Configuration settings
│   ├── data_loader.py            # Data loading utilities
│   ├── evaluation.py             # Model evaluation metrics
│   ├── feature_engineering.py    # Feature extraction
│   ├── preprocessing.py          # Text preprocessing
│   ├── utils.py                  # Utility functions
│   └── models/
│       ├── naive_bayes.py        # Naive Bayes implementation
│       └── deep_learning.py      # Deep Learning implementation
├── data/
│   └── sentiment_analysis.csv    # Training dataset
├── models_saved/                 # Saved model files
│   ├── naive_bayes.pkl           # Trained Naive Bayes model
│   ├── deep_learning.h5          # Trained Deep Learning model
│   └── tfidf_vectorizer.pkl      # TF-IDF vectorizer
├── Results/                      # Model evaluation results
│   ├── naive_bayes/
│   │   └── classification_report.txt
│   └── deep_learning/
│       └── classification_report.txt
├── save_model.py                 # Script to train and save models
├── requirement.txt               # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

1. Clone or download this project:
```bash
cd ODCRU
```

2. Install dependencies:
```bash
pip install -r requirement.txt
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **nltk**: Natural language processing
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **fastapi**: Web framework for API
- **uvicorn**: ASGI server for FastAPI
- **joblib**: Model serialization
- **tensorflow**: Deep learning framework

## Usage

### Training Models

To train and save both models:

```bash
python save_model.py
```

This script will:
1. Load data from `data/sentiment_analysis.csv`
2. Preprocess the text data
3. Train both Naive Bayes and Deep Learning models
4. Save models to `models_saved/`
5. Generate evaluation reports in `Results/`

### Running the API

Start the FastAPI server:

```bash
uvicorn api.app:app --reload
```

The API will be available at `http://localhost:8000`

#### API Endpoints

**Health Check:**
```
GET /
```
Returns: `{"status": "API running"}`

**Sentiment Prediction:**
```
POST /predict
Content-Type: application/json

{
  "text": "This product is amazing!",
  "model": "naive_bayes"
}
```

**Parameters:**
- `text` (string, required): Text to analyze
- `model` (string, optional): Model to use - `"naive_bayes"` or `"deep_learning"` (default: `"naive_bayes"`)

**Response:**
```json
{
  "input_text": "This product is amazing!",
  "model_used": "naive_bayes",
  "predicted_sentiment": "positive"
}
```

### Configuration

Edit `src/config.py` to modify:
- `RANDOM_STATE`: Random seed for reproducibility
- `TEST_SIZE`: Train/test split ratio (default: 0.25)
- `MAX_FEATURES`: Maximum TF-IDF features (default: 3000)
- `EPOCHS`: Number of training epochs for deep learning (default: 5)
- `BATCH_SIZE`: Batch size for model training (default: 32)

## Model Performance

Detailed classification reports are available in `Results/`:
- `Results/naive_bayes/classification_report.txt`
- `Results/deep_learning/classification_report.txt`

## Data Format

The training data should be in CSV format with the following columns:
- `text`: The text content to analyze
- `label`: Sentiment label (0: negative, 1: neutral, 2: positive)

## Key Components

### Data Processing (`src/preprocessing.py`)
- Text cleaning and normalization
- Tokenization
- Stop word removal
- Vectorization using TF-IDF

### Model Training
- **Naive Bayes**: Using scikit-learn's Multinomial Naive Bayes
- **Deep Learning**: Neural network with TensorFlow/Keras

### Evaluation (`src/evaluation.py`)
- Precision, Recall, F1-Score
- Classification reports
- Confusion matrices

## Notes

- The API runs with CUDA disabled (`CUDA_VISIBLE_DEVICES="-1"`) for CPU inference
- Models are loaded from `models_saved/` directory
- Text input is automatically cleaned before prediction
- All three sentiment classes are supported: negative, neutral, positive

## Author

Farhan Akhtar
