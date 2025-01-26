# Automated Customer Reviews - NLP Project


## Project Overview

This project aims to develop an NLP model to automate the processing of customer feedback for a retail company. The goal is to classify customer reviews into positive, neutral, or negative categories, summarize reviews based on star ratings and product categories, and create a dynamic visualization dashboard using Plotly.

## Executive Summary

The company receives thousands of text reviews every month, making it challenging to manually categorize, analyze, and visualize them. An automated system can save time, reduce costs, and provide real-time insights into customer sentiment.

## Project Goals

1. **Classify Customer Reviews**: Classify customer reviews (textual content) into positive, neutral, or negative.
2. **Summarize Reviews**: Summarize reviews for each product category broken down by star rating.
3. **Handle Multiple Product Categories**: Manage a feasible number of product categories, e.g., top 10 or top 50.
4. **Create a Dynamic Dashboard**: Develop an interactive visualization dashboard to present insights.

## Data Collection

- Utilized publicly available datasets of Amazon customer reviews.
- Ensured computing resources could handle the dataset size and machine learning processes.
- Combined all three available datasets to gather as much data as possible for training the model.

## Traditional NLP & ML Approaches

### Data Preprocessing

1. **Data Analysis**
   - Relevant Columns: Explored the data to identify useful columns for the model (`doRecommend`, `numHelpful`, `rating`, `text`, `title`).
   - Data Balance: Plotted the distribution of ratings and `doRecommend` to assess data balance. Conclusion: The data is very unbalanced, necessitating resampling.
   - Length and Rating Correlation: Explored the correlation between review length and rating, finding no significant correlation.
   - Word Frequency: Created word clouds for each sentiment to visually analyze the most used words. Added common words lacking sentimental context to the stopword list.

2. **Data Cleaning**
   - Encoding Ratings into Sentiments: Encoded 1 & 2-star ratings as negative (0), 3-star as neutral (1), and 4-5 stars as positive (2).
   - Standard Data Cleaning: Removed special characters, punctuation, unnecessary whitespace, and rows with null values. Converted text to lowercase and removed stopwords.
   - Tokenization and Lemmatization: Tokenized text data and applied lemmatization.
   - Vectorization: Used CountVectorizer or TF-IDF Vectorizer to convert text data into numerical vectors.

3. **Resampling**
   - Applied SMOTE to balance the training data.

### Model Building

1. **Model Selection**
   - Explored various algorithms: Naive Bayes, Logistic Regression, Support Vector Machines, and Random Forest. Selected the best based on performance metrics.
2. **Model Training and Fine-Tuning**
   - Selected Random Forest Classifier based on accuracy, precision, recall, and F1-score.
   - Employed class weight adjustment and hyperparameter tuning with Grid Search to improve results.

### Model Evaluation

- Accuracy: 0.96
- Precision: 0.95
- Recall: 0.96
- F1 Score: 0.95
- Confusion Matrix: 


## Sequence-to-Sequence Modeling with LSTM

- Built a Bidirectional LSTM model with four layers: embedding layer, two hidden layers, and an activation layer.
- Applied the same preprocessing steps as for traditional models.
- Achieved good results but lagged behind the Random Forest Classifier.

## Transformer Approach (HuggingFace API)

### Data Preprocessing

- Data Cleaning and Tokenization: Cleaned and tokenized the customer review data.
- Data Encoding and Padding: Encoded tokenized input sequences and applied padding.
- Dataset Reduction: Implemented a `reduce_data` function to reduce dataset size while maintaining diversity.

### Model Building

- Model Selection: Explored BERT, RoBERTa, and DistilBERT.
- RoBERTa: Chosen for robustness and state-of-the-art performance.
- DistilBERT: Chosen for efficiency and high performance.

- Model Fine-Tuning: Fine-tuned pre-trained models with parameters: batch size 32, learning rate \(2 \times 10^{-5}\), and 10 training epochs.

### Model Evaluation

- Accuracy: 81%
- Precision, Recall, F1-score:
- Negative: Precision=0.85, Recall=0.85, F1-score=0.85
- Neutral: Precision=0.73, Recall=0.72, F1-score=0.72
- Positive: Precision=0.85, Recall=0.87, F1-score=0.81
- Confusion Matrix: 


## Category Rating Summary and Interactive Dashboard

### Category Rating Summary Using T5

- **Data Preparation**: Cleaned, tokenized, and processed the review data.
- **Summarization Process**: Generated summaries of reviews for each product category and star rating using T5.
- **Results**: Summarized reviews provided clear insights into customer feedback for each rating within a product category.

### Interactive Dashboard Using Plotly

Developed an interactive Plotly dashboard to visualize the summarized reviews for each product category and star rating.

#### Features

1. **Dropdown Menu**: Allows users to select a specific category to view its review summaries.
2. **Dynamic Summaries**: Displays the summarized reviews, including ratings, for the selected category.
3. **Interactive Interface**: User-friendly interface for exploring customer feedback.

#### Implementation

```python
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
  html.H1("Review Summaries Dashboard"),
  dcc.Dropdown(
      id='category-dropdown',
      options=dropdown_options,
      value=categories[0],
      clearable=False,
      style={'width': '50%'}
  ),
  html.Div(id='summary-output')
])

# Callback to update the summary output based on selected category
@app.callback(
  Output('summary-output', 'children'),
  [Input('category-dropdown', 'value')]
)
def update_summary(category):
  summaries = summary_dict.get(category, [])
  return html.Ul([html.Li(summary) for summary in summaries])

# Run the app
if __name__ == '__main__':
  app.run_server(debug=True)

```

## Conclusion
The category rating summary using T5 and the interactive Plotly dashboard provide valuable insights into customer feedback. These tools enable the company to efficiently analyze and visualize reviews, facilitating data-driven decisions to enhance products and services.
