from common_imports import *
from nlpmodel import NlpModel
from logging_config import *
from utils.helper import CONTRACTIONS_DICT

class PreprocessingModel(NlpModel):
    def __init__(self):
        super().__init__()
        self.df = self.cleaned_df

    def count_html_tags(self) -> None:
        """Counts the number of texts containing HTML tags."""
        html_tag_pattern = re.compile(r"<[^>]+>")  
        count = sum(1 for item in self.df["Text"] if isinstance(item, str) and html_tag_pattern.search(item))
        self.logger.info(f"Number of data containing HTML tags: {count} | Percentage: {100 * count / len(self.df):.2f}%")
    
    def count_url(self) -> None:
        """Counts the number of texts containing URLs."""
        url_pattern = re.compile(r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])")
        count = sum(1 for item in self.df["Text"] if isinstance(item, str) and url_pattern.search(item))
        self.logger.info(f"Number of data containing URLs: {count} | Percentage: {100 * count / len(self.df):.2f}%")
    
    def count_emoji(self) -> None:
        """Counts the number of texts containing emojis."""
        emoji_pattern = re.compile(r"[\U0001F600-\U0001F64F]")  # Better regex for emojis
        count = sum(1 for item in self.df["Text"] if isinstance(item, str) and emoji_pattern.search(item))
        self.logger.info(f"Number of data containing emojis: {count} | Percentage: {100 * count / len(self.df):.2f}%")
    
    def lowercase(self, text: str) -> str:
        """Converts text to lowercase."""
        return text.lower()

    def expand_contractions(self, text: str) -> str:
        """Expands contractions in the text."""
        words = text.split()
        expanded_words = [CONTRACTIONS_DICT.get(word.lower(), word) for word in words]
        return " ".join(expanded_words)

    def remove_html_tags(self, text: str) -> str:
        """Removes HTML tags from the text."""
        return re.sub(r"<[^>]+>", "", text)

    def remove_urls(self, text: str) -> str:
        """Removes URLs from the text."""
        return re.sub(r"(http|ftp|https):\/\/[\w_-]+(?:\.[\w_-]+)+(?:[\w.,@?^=%&:\/~+#-]*)?", "", text)
    
    def remove_punctuations(self, text: str) -> str:
        """Removes punctuations from the text"""
        return re.sub(r"[!\'#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]", "", text)
    
    def decode_text(self, text: str) -> str:
        """Encodes string to bytes and decodes to utf-8"""
        return text.encode("utf-8-sig").decode("utf-8-sig")
    
    def remove_stop_words(self, text: str) -> str:
        """Removes stop words"""
        pass
    
    def process_text(self, text: str) -> str:
        """Applies multiple preprocessing steps to the text."""
        text = self.decode_text(text)
        text = self.lowercase(text)
        text = self.remove_punctuations(text)
        text = self.expand_contractions(text)
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        return text

    def preprocess_dataframe(self) -> None:
        """Applies text preprocessing to the entire DataFrame."""
        self.df["Text"] = self.df["Text"].astype(str).apply(self.process_text)

if __name__ == "__main__":
    model = PreprocessingModel()
    model.preprocess_dataframe()
    print(model.df.head())  
