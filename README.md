# sentiment-analysis-bert-llm-as-a-judge-vader

## Sentiment Analysis Pipeline: VADER, BERT & LLM Comparison

#### This project provides a complete sentiment analysis pipeline and evaluation framework comparing Bert and Vader using LLM as a judge using the following approach:

#### VADER (rule-based, lexicon-based sentiment analysis)
####BERT (transformer-based embeddings + logistic regression)
####LLMs (Large Language Models via OpenAI / OpenRouter APIs)

###It includes tools for:

####Text classification using VADER and Sentence-BERT
####LLM-powered annotation of training data
####Comparative evaluation via classification reports and confusion matrices
Features

### sentiment_vader: 
#### Classifies raw text using VADER and returns all score components
#### Annotates text with LLMs (OpenAI, LLaMA via OpenRouter)
#### Trains lightweight classifiers using BERT embeddings
#### Compares model predictions using scikit-learn evaluation tools
#### Outputs structured CSVs for reproducible experiments

### Project Structure

├── dataset_train.csv                # Unlabeled comments from restaurant reviews
├── dataset_valid.csv                # Validation set for final evaluation
├── vader_emo_com_sentiment.csv      # Parsed VADER sentiment with scores and labels
├── utils.py                         # Sentiment functions (e.g., sentiment_vader)
├── BERT_Log_Reg_Classifier.ipynb    # BERT training pipeline
├── Evaluation.ipynb                 # Model comparison and reporting
├── requirements.txt
📌 Example Output

#### sentiment_vader("This place was amazing!")
##### → (neg=0.0, neu=0.3, pos=0.7, compound=0.7269, label='Positive')

###Classification Example (BERT vs LLM)

##### Label	Precision	Recall	F1-Score
##### negative	0.70	0.61	0.65
##### neutral	0.00	0.00	0.00
##### positive	0.81	0.93	0.87

### Use Cases

##### Benchmarking classical vs neural sentiment methods
##### Creating weak supervision pipelines using LLM annotations
##### NLP education and model interpretability studies

### Requirements

###### pip install -r requirements.txt

##### Make sure you have:

##### Python 3.9+
##### nltk, scikit-learn, pandas, sentence-transformers, openai, matplotlib


### Credits

##### VADER (https://github.com/cjhutto/vaderSentiment)
##### HuggingFace Transformers (https://chatgpt.com/c/681a6999-8ca8-8003-853c-cbf973490621#:~:text=HuggingFace%20Transformers)
##### OpenAI (https://platform.openai.com/)
