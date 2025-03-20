from src.common_imports import * # noqa: F403, F405
from src.nlpmodel import NlpModel
from src.logging_config import *  # noqa: F403, F405
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT  

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import swifter

from sklearn.feature_extraction.text import CountVectorizer

from scipy import sparse

import numpy as np
from tqdm import tqdm
import gc
import dask.dataframe as dd
import seaborn as sns


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()


class FeatureEngineering(NlpModel):
    def __init__(self):
            super().__init__()
            self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
            self.STATS_DIR = os.path.join(self.BASE_DIR, "stats")
            self.PREPROCESSED_DATA_DIR = os.path.join(self.SAVE_DATA_DIR, "PREPROCESSED_Reviews.csv")
            
            self.df = pd.read_csv(self.PREPROCESSED_DATA_DIR)
            self.df_size = len(self.df)

            self.batch_size = 10_000
            self.n_batches = (self.df_size + self.batch_size - 1) // self.batch_size

            self.vectorizer = CountVectorizer()

    def word_freq(self):
        X = self.vectorizer.fit_transform(self.df['Text'])
        
        word_counts = X.sum(axis=0).A1
        feature_names = self.vectorizer.get_feature_names_out() 
        
        freq = dict(zip(feature_names, word_counts))
        
        sorted_freq = sorted(freq.items(), key=lambda item: item[1], reverse=True)

        # Convert to DataFrame for plotting
        freq_df = pd.DataFrame(sorted_freq[:10], columns=['Word', 'Frequency'])

        sns.barplot(x='Word', y='Frequency', data=freq_df)
        plt.xticks(rotation=45)
        plt.show()
        
        

    def create_bow(self):
        """For creating the bag of words from preprocessed data and saving the result into a DataFrame"""
        self.logger.info("BAG OF WORDS ACTION HAS STARTED..")

        # Ensure NaN values are handled
        self.df['Text'] = self.df['Text'].fillna('')
        
        # Initialize CountVectorizer with more aggressive feature reduction
        self.vectorizer = CountVectorizer(
            min_df=5,            # Ignore terms that appear in less than 5 documents
            max_df=0.5,          # Ignore terms that appear in more than 50% of documents
            max_features=10000   # Only keep top 10,000 features
        )
        
        self.vectorizer.fit(self.df['Text'])

        sparse_matrices = []

        # Process in smaller batches to reduce memory usage
        batch_size = min(5000, self.batch_size)  # Use smaller batches if needed
        
        for i in track(range(0, len(self.df), batch_size), description="Processing batches"):
            batch = self.df.iloc[i:i+batch_size]['Text']
            batch_matrix = self.vectorizer.transform(batch)
            sparse_matrices.append(batch_matrix)
            gc.collect()

        sparse_matrices = sparse.vstack(sparse_matrices)

        feature_names = self.vectorizer.get_feature_names_out()
        bow_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrices, columns=feature_names)

        self.bow_df = bow_df  

        return bow_df



    def save_to_parquet(self, df, output_path: str = "processed_data.parquet") -> None:
        """Save the processed DataFrame to Parquet with gzip compression"""
        self.logger.info("SAVING TO PARQUET STARTED..")
        self.print_section("SAVING TO PARQUET STARTED..")

     # Check if DataFrame has sparse data
        has_sparse = hasattr(df, 'sparse') and hasattr(df.sparse, 'to_dense')
    
        if has_sparse:
        # Convert sparse DataFrame to dense
            dense_df = df.sparse.to_dense()
        else:
            self.logger.warning("Input DataFrame does not contain sparse data.")
            dense_df = df
    
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
        # Save to Parquet format with gzip compression
        dense_df.to_parquet(f"{output_path}.parquet.gz", compression="gzip")

        self.logger.info(f"DF SAVED TO {output_path}.parquet.gz")

        return output_path


       


if __name__ == "__main__":
    model = FeatureEngineering()
    bag_of_words_df = model.create_bow()
    
    OUTPUT_DIR = os.path.join(model.SAVE_DATA_DIR, "bag_of_words", "bag_of_words")

    model.save_to_parquet(model.df, OUTPUT_DIR)





