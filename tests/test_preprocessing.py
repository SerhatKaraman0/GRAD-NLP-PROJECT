import pytest
import pandas as pd
import os
from src.preprocessing_model import PreprocessingModel
import emoji

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Id': [1, 2, 3],
        'Score': [5, 4, 3],
        'Summary': ['Good', 'Ok', 'Bad'],
        'Text': [
            'Check this link https://example.com and some HTML <div>content</div> 😊',
            'Another text with http://test.org and <span>HTML</span>',
            'Plain text without any special content'
        ]
    })

@pytest.fixture
def preprocessing_model(sample_df):
    model = PreprocessingModel()
    model.df = sample_df
    return model

class TestPreprocessingModel:
    def test_initialization(self, preprocessing_model):
        assert preprocessing_model.SAVE_DATA_DIR is not None
        assert preprocessing_model.STATS_DIR is not None
        assert preprocessing_model.patterns is not None
        assert preprocessing_model.word_replacements is not None

    def test_pattern_matching(self, preprocessing_model):
        count, percentage = preprocessing_model._count_pattern_matches('html', preprocessing_model.df)
        assert count == 2  # Two entries have HTML


    def test_process_text(self, preprocessing_model):
        test_text = "Check this URL https://example.com and <div>HTML</div> 😊"
        processed_text, tokens, sentences = preprocessing_model._process_text(test_text)
        
        assert "https://example.com" not in processed_text
        assert "<div>" not in processed_text
        assert "html" in processed_text
        assert isinstance(tokens, list)
        assert isinstance(sentences, list)

    def test_process_chunk(self, preprocessing_model):
        chunk = pd.Series(["Test text 1", "Test text 2"])
        patterns = preprocessing_model.patterns
        word_replacements = preprocessing_model.word_replacements
        
        processed, tokenized, sentence_tokenized = preprocessing_model._process_chunk(
            (chunk, patterns, word_replacements)
        )
        
        assert len(processed) == 2
        assert len(tokenized) == 2
        assert len(sentence_tokenized) == 2

    def test_preprocess_dataframe_small(self, preprocessing_model):
        preprocessing_model.preprocess_dataframe()
        
        assert 'Text' in preprocessing_model.df.columns
        assert 'Token_count' in preprocessing_model.df.columns
        assert 'Sentence_count' in preprocessing_model.df.columns
        assert len(preprocessing_model.df) == 3

    def test_save_to_csv(self, preprocessing_model, tmp_path):
        output_path = os.path.join(tmp_path, "test_output.csv")
        preprocessing_model.preprocess_dataframe()
        preprocessing_model.save_to_csv(output_path)
        
        assert os.path.exists(output_path)
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == len(preprocessing_model.df)
        assert all(col in loaded_df.columns for col in ['Id', 'Score', 'Summary', 'Text'])

    def test_get_statistics(self, preprocessing_model, capsys):
        preprocessing_model.get_statistics(preprocessing_model.df)
        captured = capsys.readouterr()
        assert "Pattern Statistics" in captured.out
        assert "html" in captured.out
        assert "url" in captured.out

    def test_remove_foreign_words(self, preprocessing_model):
        preprocessing_model.df['Text'] = ['English text', '你好世界', 'More English']
        preprocessing_model.remove_foreign_words()
        assert preprocessing_model.df['Text'].iloc[1] == ''  # Non-English text should be empty
        assert preprocessing_model.df['Text'].iloc[0] != ''  # English text should remain

    def test_edge_cases(self, preprocessing_model):
        edge_cases_df = pd.DataFrame({
            'Id': [1, 2, 3],
            'Score': [5, 4, 3],
            'Summary': ['Test1', 'Test2', 'Test3'],
            'Text': ['', '   ', '<script>alert("test")</script>']
        })
        
        preprocessing_model.df = edge_cases_df
        preprocessing_model.preprocess_dataframe()
        
        assert len(preprocessing_model.df) == 3
        assert '<script>' not in preprocessing_model.df['Text'].iloc[2]