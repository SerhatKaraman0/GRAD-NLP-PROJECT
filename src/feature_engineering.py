from src.common_imports import * # noqa: F403, F405
from src.nlpmodel import NlpModel
from src.logging_config import *  # noqa: F403, F405
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT  
from nltk.tokenize import word_tokenize, sent_tokenize
import gc
import itertools
from multiprocessing import get_context
from nltk.corpus import stopwords
import swifter
from tqdm import tqdm
import string 
from textblob import TextBlob
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings("ignore", category=FutureWarning)


class FeatureEngineering(NlpModel):
    def __init__(self):
            super().__init__()
            self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
            self.STATS_DIR = os.path.join(self.BASE_DIR, "stats")
            self.PREPROCESSED_DATA_DIR = os.path.join(self.SAVE_DATA_DIR, "PREPROCESSED_Reviews.csv")
            


