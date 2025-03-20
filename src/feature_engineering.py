from src.common_imports import * # noqa: F403, F405
from src.nlpmodel import NlpModel
from src.logging_config import *  # noqa: F403, F405
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT  

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import swifter

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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
        from collections import Counter
        import seaborn as sns
        import matplotlib.pyplot as plt

        all_words = ' '.join(self.df['Text']).split()
        word_counts = Counter(all_words)
        
        common_words = word_counts.most_common(20)
        words, counts = zip(*common_words)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(words), y=list(counts), palette="viridis")
        plt.xticks(rotation=45)
        plt.title("Top 20 Most Common Words")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        
    def _vectorize_tfidf(self):
        # Ensure NaN values are handled
        self.df['Text'] = self.df['Text'].fillna('')

        tfidf_vectorizer = TfidfVectorizer()
        X = tfidf_vectorizer.fit_transform(self.df["Text"])

        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_score = X.mean(axis=0).A1

        df_tfidf = pd.DataFrame({
                                "word": feature_names,
                                "score": tfidf_score
                                }) 
        
        sorted_df = df_tfidf.sort_values(by="score", ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(sorted_df[:20]["word"]), y=list(sorted_df[:20]["score"]), palette="viridis")
        plt.xticks(rotation=45)
        plt.title("Top 20 Words by TF-IDF Score")
        plt.xlabel("Words")
        plt.ylabel("TF-IDF Score")
        plt.tight_layout()
        plt.show()

        return sorted_df
    

    def create_bow(self):
        """For creating the bag of words from preprocessed data and saving the result into a DataFrame"""
        self.logger.info("BAG OF WORDS ACTION HAS STARTED..")

        # Ensure NaN values are handled
        self.df['Text'] = self.df['Text'].fillna('')
        
        # Initialize CountVectorizer
        self.vectorizer = CountVectorizer()
        
        self.vectorizer.fit(self.df['Text'])

        sparse_matrices = []

        for i in track(range(0, 50_000, self.batch_size), description="Processing batches"):
            batch = self.df.iloc[i:i+self.batch_size]['Text']
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

        # Convert sparse DataFrame to dense
        dense_bow_df = df.sparse.to_dense()
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save to Parquet format with gzip compression
        dense_bow_df.to_parquet(f"{output_path}.parquet.gz", compression="gzip")

        self.logger.info(f"DF SAVED TO {output_path}.parquet.gz")

        return output_path


    def read_parquet(self, df, input_path: str):
        data = pd.read_parquet(input_path)
        print(data.head())
        print(data.count)



if __name__ == "__main__":
    model = FeatureEngineering()
    
    OUTPUT_DIR = os.path.join(model.SAVE_DATA_DIR, "bag_of_words", "bag_of_words.parquet.gz")

    tfidf_df = model._vectorize_tfidf()

    tfidf_output_path = os.path.join(model.SAVE_DATA_DIR, "tfidf_scores.csv")
    tfidf_df.to_csv(tfidf_output_path, index=False)
    
    print(f"TF-IDF scores saved to {tfidf_output_path}")
    






