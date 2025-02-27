import pytest
import pandas as pd
import numpy as np
from src.preprocessing_model import PreprocessingModel
from src.nlpmodel import NlpModel
import re
from nltk.tokenize import word_tokenize


class TestPreprocessingModel:
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            'Id': [1, 2, 3],
            'Score': [5, 4, 3],
            'Summary': ['Good product', 'Average', 'Poor'],
            'Text': [
                'This is a <b>great</b> product! Check it at http://example.com 😊',
                'The product was ok... but could be better :)',
                'BAD product!!! Not worth the $$$'
            ],
            'ProductId': ['PRODUCT-0001'] * 3,
            'UserId': ['USER-0001'] * 3,
            'ProfileName': ['tester-master-of-none'] * 3,
            'HelpfulnessNumerator': [5] * 3,
            'HelpfulnessDenominator': [5] * 3,
            'Time': ['break-time'] * 3
        })

    @pytest.fixture
    def model(self, sample_df):
        """Initialize model with sample data"""
        model = PreprocessingModel()
        model.df = sample_df
        model.patterns = {
            'html': re.compile(r"<[^>]+>"),
            'url': re.compile(r"(http|ftp|https):\/\/[^\s/$.?#].[^\s]*"),
            'emoji': re.compile(r"[\U0001F600-\U0001F64F]"),
            'punctuation': re.compile(r"[!\'#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]"),
            'extra_punctuation': re.compile(r'(\W)\1+'),
            'whitespace': re.compile(r'\s+')
        }
        model.word_replacements = {'tbh': 'to be honest', 'imo': 'in my opinion', 'gr8': 'great', 'gud': 'good'}
        return model

    def test_html_tag_removal(self, model):
        """Test HTML tag removal"""
        texts = pd.Series(['<p>Test</p>', '<div>Another test</div>'])
        processed, _ = model._process_chunk((texts, model.patterns, model.word_replacements))
        assert all('<' not in text and '>' not in text for text in processed)

    def test_edge_cases(self, model):
        """Test handling of edge cases"""
        edge_cases = pd.DataFrame({
            'Id': [1, 2, 3, 4],
            'Score': [5, 4, 3, 2],
            'Summary': ['Test'] * 4,
            'Text': ['', '   ', None, '!!!!!!!'],
            'ProductId': ['PRODUCT-0001'] * 4,
            'UserId': ['USER-0001'] * 4,
            'ProfileName': ['tester-master-of-none'] * 4,
            'HelpfulnessNumerator': [5] * 4,
            'HelpfulnessDenominator': [5] * 4,
            'Time': ['break-time'] * 4
        })

        model.df = edge_cases
        model.preprocess_dataframe()

        assert len(model.df) == 4
        assert all(isinstance(text, str) for text in model.df['Text'].fillna(''))
        assert all(isinstance(tokens, list) for tokens in model.df.get('Tokens', [[]]))

    def test_performance(self, model):
        """Test performance with larger dataset"""
        large_df = pd.DataFrame({
            'Id': range(1000),
            'Score': np.random.randint(1, 6, 1000),
            'Summary': ['Test'] * 1000,
            'Text': ['This is test text ' * 10] * 1000,
            'ProductId': ['PRODUCT-0001'] * 1000,
            'UserId': ['USER-0001'] * 1000,
            'ProfileName': ['tester-master-of-none'] * 1000,
            'HelpfulnessNumerator': [5] * 1000,
            'HelpfulnessDenominator': [5] * 1000,
            'Time': ['break-time'] * 1000
        })

        model.df = large_df

        import time
        start_time = time.time()
        model.preprocess_dataframe()
        processing_time = time.time() - start_time

        assert processing_time < 30  # Should process 1000 rows in less than 30 seconds

    def test_save_load_csv(self, model, tmp_path):
        """Test CSV save and load functionality"""
        model.preprocess_dataframe()
        test_file = tmp_path / "test_output.csv"
        model.save_to_csv(str(test_file))

        assert test_file.exists()
        assert test_file.stat().st_size > 0

        loaded_df = pd.read_csv(test_file)
        assert len(loaded_df) == len(model.df)
        assert all(col in loaded_df.columns for col in model.df.columns)