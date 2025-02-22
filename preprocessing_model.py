from common_imports import *
from nlpmodel import NlpModel
from logging_config import *

class PreprocessingModel(NlpModel):
    def __init__(self):
        super().__init__()
        self.df = self.cleaned_df

    def count_html_tags(self) -> None:
        """
        Counts how many elements in the given list contain HTML tags.

        :param data: Strings
        :return: Integer count of elements containing HTML tags
        """
        
        html_tag_pattern = re.compile(r"<[^>]+>")  
        
        count = sum(1 for item in self.df["Text"] 
                    if isinstance(item, str) and html_tag_pattern.search(item))
        
        self.logger.info(f"Number of data contains html tags: {count} | Percentage: {100 * count / len(self.df)}%")
    
    def count_url(self) -> None:
        url_pattern = re.compile(r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])")
        count = sum(1 for item in self.df["Text"] 
                    if isinstance(item, str) and url_pattern.search(item))
        
        self.logger.info(f"Number of data contains url: {count} | Percentage: {100 * count / len(self.df)}%")
    
    def count_emoji(self) -> None:
        emoji_pattern = re.compile(r"/(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])/g")
        count = sum(1 for item in self.df["Text"] 
                    if isinstance(item, str) and emoji_pattern.search(item))
        
        self.logger.info(f"Number of data contains emoji: {count} | Percentage: {100 * count / len(self.df)}%")
    
    # TODO: Data Cleaning Functions
    def clean_func(self):
        pass
    # TODO: Lemmazation and Tokenization Functions

if __name__ == "__main__":
    model = PreprocessingModel()
    model.count_html_tags()
    model.count_url()
    model.count_emoji()
    

    
