# NLP-Learning Visualization Features

This document provides comprehensive documentation for the visualization features added to the NLP-Learning project.

## Overview

The visualization features enhance the pretraining pipeline by providing visual insights into how review characteristics vary by rating and comparing model performance. These visualizations help in understanding patterns and relationships between review text, ratings, and model effectiveness.

## Rating-Based Visualizations

### 1. Word Frequency by Rating

**Implementation**: `create_word_frequency_by_rating()` in `DataProcessor` class

**Purpose**: Shows the top N most frequent words for each rating category (1-5 stars).

**Features**:
- Filters out common English stop words
- Creates separate subplot for each rating
- Sorts words by frequency
- Displays top 10 words for each rating

**Usage Example**:
```python
data_processor = DataProcessor()
word_freq_path = data_processor.create_word_frequency_by_rating()
```

### 2. Review Length by Rating

**Implementation**: `create_review_length_by_rating()` in `DataProcessor` class

**Purpose**: Visualizes the distribution of review lengths (character count) across different ratings.

**Features**:
- Box plot shows the quartiles and outliers of review length
- Swarm plot overlay shows the actual distribution of review lengths
- Helps identify if higher/lower ratings correlate with review length

**Usage Example**:
```python
data_processor = DataProcessor()
review_len_path = data_processor.create_review_length_by_rating()
```

### 3. Word Clouds by Rating

**Implementation**: `create_wordclouds_by_rating()` in `DataProcessor` class

**Purpose**: Generates word clouds for each rating category, with word size proportional to frequency.

**Features**:
- Creates visual representation of most common terms for each rating
- Word size indicates frequency in the corpus
- Uses distinct subplot for each rating
- Limited to top 100 words for clarity
- Supports multiprocessing for parallel generation of word clouds
- Automatically samples large datasets to improve performance

**Usage Example**:
```python
data_processor = DataProcessor()
# Basic usage
wordclouds_path = data_processor.create_wordclouds_by_rating()

# With optimization parameters
wordclouds_path = data_processor.create_wordclouds_by_rating(
    max_samples=2000,           # Limit to 2000 reviews per rating
    use_multiprocessing=True    # Use parallel processing
)
```

**Optimization Notes**: 
The word cloud generation has been optimized to:
- Sample reviews when dataset is large (configurable with `max_samples`)
- Utilize multiprocessing to generate word clouds in parallel
- Optimize WordCloud parameters for faster generation

### 4. Sentiment Distribution

**Implementation**: `create_sentiment_distribution()` in `DataProcessor` class

**Purpose**: Shows the overall distribution of reviews across rating categories.

**Features**:
- Bar chart showing the count of reviews for each rating
- Displays numeric counts above each bar
- Helps identify class imbalance in the dataset

**Usage Example**:
```python
data_processor = DataProcessor()
sentiment_path = data_processor.create_sentiment_distribution()
```

## Model Comparison Features

### 1. Model Metrics Comparison

**Implementation**: `ModelComparison` class in `model_comparison.py`

**Purpose**: Compares performance metrics across different models.

**Features**:
- Loads metrics from all available models
- Creates a comparative table highlighting the best performance for each metric
- Generates bar charts comparing key metrics across models
- Visualizes training and validation loss across different models

**Usage Example**:
```python
model_comparison = ModelComparison(save_data_dir="/path/to/data")
metrics_df = model_comparison.load_model_metrics()
comparison_table = model_comparison.create_comparison_table(metrics_df)
visualizations = model_comparison.create_comparison_visualizations()
```

## Dashboard Generation

### 1. Model-Specific Dashboard

**Implementation**: `generate_model_dashboard()` in `DashboardGenerator` class

**Purpose**: Creates a dashboard for a specific model with its metrics and visualizations.

**Features**:
- Displays model performance metrics
- Shows rating-based visualizations
- Includes model-specific charts and plots

**Usage Example**:
```python
dashboard_generator = DashboardGenerator(save_data_dir="/path/to/data")
dashboard_path = dashboard_generator.generate_model_dashboard(
    model_results=metrics,
    model_type="ensemble"
)
```

### 2. Model Comparison Dashboard

**Implementation**: `generate_model_comparison_dashboard()` in `DashboardGenerator` class

**Purpose**: Creates a dashboard comparing all available models.

**Features**:
- Displays a comparative table of model metrics
- Shows performance comparison visualizations
- Includes rating-based visualizations
- Provides links to individual model dashboards

**Usage Example**:
```python
dashboard_generator = DashboardGenerator(save_data_dir="/path/to/data")
dashboard_path = dashboard_generator.generate_model_comparison_dashboard()
```

## Running the Tools

### Enhanced Pretraining Pipeline

The enhanced pretraining pipeline integrates these visualizations:

```bash
python src/enhanced_pretraining_pipeline.py
```

Optional arguments:
- `--max_features`: Maximum number of features for vectorization (default: 10000)
- `--batch_size`: Batch size for processing (default: 10000)
- `--skip_visuals`: Skip generating visualizations

### Rating Dashboard Generation

To generate a dashboard with rating visualizations for a specific model type:

```bash
python src/generate_rating_dashboard.py --model_type ensemble
```

Valid model types: `simple`, `deep`, `stacked`, `ensemble`

### Model Comparison Dashboard

To generate a comprehensive comparison of all available models:

```bash
python src/generate_model_comparison.py
```

This will create a dashboard comparing the performance of all trained models and include rating visualizations.
