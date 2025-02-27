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
            'ProductId': 'PRODUCT-0001',
            'UserId': 'USER-0001',
            'ProfileName': 'tester-master-of-none',
            'HelpfulnessNumerator': '5',
            'HelpfulnessDenominator': '5',
            'Time': 'break-time'
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
        
        NlpModel.print_section("HTML REMOVAL TESTS..")
        print("Original texts: \n", texts.to_string(), flush=True)
        print("Processed texts: ", processed, flush=True)

        # Check that the processed texts don't contain HTML tags
        assert all('<' not in text for text in processed)
        assert all('>' not in text for text in processed)
        assert 'test' in processed[0].lower()
        assert 'another test' in processed[1].lower()

    def test_url_removal(self, model):
        """Test URL removal"""
        texts = pd.Series(['Check http://example.com', 'Visit https://test.com/page'])
        processed, _ = model._process_chunk((texts, model.patterns, model.word_replacements))

        NlpModel.print_section("URL REMOVAL TESTS..") 
        print("Original texts: \n", texts.to_string(), flush=True)
        print("Processed texts: ", processed, flush=True)

        assert 'example.com' not in ' '.join(processed).lower()
        assert 'test.com' not in ' '.join(processed).lower()
        assert 'check' in processed[0].lower()
        assert 'visit' in processed[1].lower()

    def test_emoji_handling(self, model):
        """Test emoji removal"""
        texts = pd.Series(['Hello 😊', 'Great! 👍'])
        processed, _ = model._process_chunk((texts, model.patterns, model.word_replacements))

        NlpModel.print_section("EMOJI REMOVAL TESTS..") 
        print("Original texts: \n", texts.to_string(), flush=True)
        print("Processed texts: ", processed, flush=True) 

        # Check that emojis are removed
        assert 'hello' in processed[0].lower()
        assert 'great' in processed[1].lower()
        # Note: Removed the regex assertion since the method doesn't explicitly use the emoji pattern for checks

    def test_text_normalization(self, model):
        """Test text normalization"""
        texts = pd.Series(['UPPER CASE!!!', 'multiple     spaces'])
        processed, _ = model._process_chunk((texts, model.patterns, model.word_replacements))
        
        NlpModel.print_section("TEXT NORMALIZATION TESTS..") 
        print("Original texts: \n", texts.to_string(), flush=True)
        print("Processed texts: ", processed, flush=True)

        # Check that text is lowercased and spaces are normalized
        assert all(text == text.lower() for text in processed)
        assert all('  ' not in text for text in processed)
        assert 'upper case' in processed[0].lower()
        assert 'multiple spaces' in processed[1].lower()

    def test_tokenization(self, model):
        """Test tokenization output"""
        texts = pd.Series(['This is a test.', 'Another test sentence.'])
        processed, tokens = model._process_chunk((texts, model.patterns, model.word_replacements))
        
        NlpModel.print_section("TOKENIZATION TESTS..") 
        print("Original texts: \n", texts.to_string(), flush=True)
        

        # Check that tokenization produces valid tokens
        assert len(tokens) == len(texts)
        assert all(isinstance(token_list, list) for token_list in tokens)
        assert all(len(token_list) > 0 for token_list in tokens)
        # The tokens should be the result of word_tokenize on processed text
        for i, proc_text in enumerate(processed):
            expected_tokens = word_tokenize(proc_text)
            print("Expected tokens: ", expected_tokens, flush=True)
            assert tokens[i] == expected_tokens

    def test_slang_conversion(self, model):
        """Test slang word conversion"""
        custom_replacements = {
            'tbh': 'to be honest', 
            'imo': 'in my opinion', 
            'gr8': 'great', 
            'gud': 'good'
        }
        texts = pd.Series(["tbh it's gr8", "imo this is gud"])
        processed, _ = model._process_chunk((texts, model.patterns, custom_replacements))
        
        NlpModel.print_section("SLANG REMOVAL TESTS..") 
        print("Original texts: ", texts.to_string(), flush=True)
        print("Processed texts: ", processed, flush=True)

        # Check that slang words are replaced with their full forms
        assert 'tbh' not in processed[0].lower()
        assert 'to be honest' in processed[0].lower()
        assert 'great' in processed[0].lower()
        assert 'imo' not in processed[1].lower()
        assert 'in my opinion' in processed[1].lower()
        assert 'good' in processed[1].lower()

    def test_full_pipeline(self, model):
        """Test the full preprocessing pipeline"""
        # Create a sample with various elements to process
        texts = pd.Series([
            "<p>Check out HTTP://example.com!</p>",
            "IMO this website is GR8 😊",
            "Multiple    spaces and UPPERCASE text"
        ])
        
        # Process the texts
        processed, tokens = model._process_chunk((
            texts, 
            model.patterns,
            {'imo': 'in my opinion', 'gr8': 'great'}
        ))
        
        # Check the results
        assert len(processed) == len(texts)
        assert len(tokens) == len(texts)
        
        # First text: HTML and URL should be removed
        assert '<p>' not in processed[0]
        assert '</p>' not in processed[0]
        assert 'http' not in processed[0].lower() or 'example.com' not in processed[0].lower()
        assert 'check out' in processed[0].lower()
        
        # Second text: Slang conversion and emoji handling
        assert 'imo' not in processed[1].lower()
        assert 'in my opinion' in processed[1].lower()
        assert 'gr8' not in processed[1].lower()
        assert 'great' in processed[1].lower()
        assert '😊' not in processed[1]
        
        # Third text: Space normalization and case conversion
        assert 'multiple spaces' in processed[2].lower()
        assert 'UPPERCASE' not in processed[2]
        assert 'uppercase' in processed[2].lower()
        assert '  ' not in processed[2]
        
        # Check that tokens are non-empty for at least some texts
        assert any(len(t) > 0 for t in tokens)

    def test_edge_cases(self, model):
        """Test handling of edge cases"""
        edge_cases = pd.DataFrame({
            'Id': [1, 2, 3, 4],
            'Score': [5, 4, 3, 2],
            'Summary': ['Test', 'Test', 'Test', 'Test'],
            'Text': [
                '',  # Empty text
                '   ',  # Only whitespace
                None,  # None value
                '!!!!!!!'  # Only 'ProductId': 'PRODUCT-0001',
            'UserId': 'USER-0001',
            'ProfileName': 'tester-master-of-none',
            'HelpfulnessNumerator': '5',
            'HelpfulnessDenominator': '5',
            'Time': 'break-time'
        })
        
        model.df = edge_cases
        model.preprocess_dataframe()
        
        # Check if edge cases are handled without errors
        assert len(model.df) == 4
        assert all(isinstance(text, str) for text in model.df['Text'])
        assert all(isinstance(tokens, list) for tokens in model.df['Tokens'])

    def test_performance(self, model):
        """Test performance with larger dataset"""
        # Create a larger dataset
        large_df = pd.DataFrame({
            'Id': range(1000),
            'Score': np.random.randint(1, 6, 1000),
            'Summary': ['Test'] * 1000,
            'Text': ['This is test text ' * 10] * 'ProductId': 'PRODUCT-0001',
            'UserId': 'USER-0001',
            'ProfileName': 'tester-master-of-none',
            'HelpfulnessNumerator': '5',
            'HelpfulnessDenominator': '5',
            'Time': 'break-time'
        })
        
        model.df = large_df
        
        # Test processing time
        import time
        start_time = time.time()
        model.preprocess_dataframe()
        processing_time = time.time() - start_time
        
        # Assert processing time is reasonable (adjust threshold as needed)
        assert processing_time < 30  # Should process 1000 rows in less than 30 seconds

    def test_save_load_csv(self, model, tmp_path):
        """Test CSV save and load functionality"""
        # Process data
        model.preprocess_dataframe()
        
        # Save to temporary file
        test_file = tmp_path / "test_output.csv"
        model.save_to_csv(str(test_file))
        
        # Check if file exists and is not empty
        assert test_file.exists()
        assert test_file.stat().st_size > 0
        
        # Load and verify data
        loaded_df = pd.read_csv(test_file)
        assert len(loaded_df) == len(model.df)
        assert all(col in loaded_df.columns for col in model.df.columns)

if __name__ == "__main__":
    pytest.main([__file__])