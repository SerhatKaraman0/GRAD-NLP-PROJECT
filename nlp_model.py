from common_imports import *
import logging
from logging_config import *

class NLP_Model:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing NLP_Model")
        
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data", "Reviews.csv")
        
        self.df = pd.read_csv(self.DATA_DIR)
        self.cleaned_df = self.clean_unnecessary()
        self.logger.info("NLP_Model initialized successfully")

    def print_section(self, title):
        """Prints a formatted section header"""
        line = "═" * 30
        print(Fore.CYAN + line + f" {title} " + line + Style.RESET_ALL)

    def class_analysis(self):
        self.print_section("CLASS ANALYSIS")
        for i in range(1, 6):
            count = len(self.cleaned_df[self.cleaned_df["Score"] == i])
            percentage = 100 * count / len(self.cleaned_df)
            self.logger.debug(f"Class {i}: {count} values, {percentage:.2f}%")
            print(f"Number of values in class {i}: {count} | Percentage of class: {percentage:.2f}%")

    def inspect_df(self):
        self.print_section("DF SHAPE")
        self.logger.debug(f"DataFrame shape: {self.df.shape}")
        print(self.df.shape)

        self.print_section("DF NULL VAL COUNT")
        null_counts = self.df.isnull().sum()
        self.logger.debug(f"Null values per column: {null_counts}")
        print(null_counts)

        self.print_section("FIRST 5 ROWS OF DF")
        self.logger.debug(f"First 5 rows of DataFrame: {self.df.head(5).to_string(index=False)}")
        print(self.df.head(5).to_string(index=False))

        self.print_section("DF INFO")
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        info_str = buffer.getvalue()
        self.logger.debug(f"DataFrame info: {info_str}")
        print("\n".join(textwrap.wrap(info_str, width=80)))

        self.print_section("DF SUMMARY")
        summary = self.df.describe().to_string()
        self.logger.debug(f"DataFrame summary: {summary}")
        print(summary)

    def grab_col_names(self, cat_th: int = 10, car_th: int = 20) -> tuple[list, list, list]:
        categorical_cols = [col for col in self.df.columns if self.df[col].dtype == "O"]
        numerical_but_cat = [col for col in self.df.columns if self.df[col].nunique() < cat_th and self.df[col].dtype in ["int64", "float64"]]
        categorical_but_car = [col for col in categorical_cols if self.df[col].nunique() > car_th]
        cat_cols = [col for col in categorical_cols + numerical_but_cat if col not in categorical_but_car]
        num_cols = [col for col in self.df.columns if self.df[col].dtype in ["int64", "float64"] and col not in numerical_but_cat]

        self.logger.debug(f"Categorical columns: {cat_cols}")
        self.logger.debug(f"Numerical columns: {num_cols}")
        self.logger.debug(f"Categorical but cardinal columns: {categorical_but_car}")

        return cat_cols, num_cols, categorical_but_car

    def null_values(self) -> tuple[int, float, int]:
        null_occur_bool = self.df.isnull().values.any()
        null_count = self.df.isnull().sum().sort_values(ascending=False)
        total_null_val = null_count.sum()
        null_count_percent = (self.df.isnull().sum() / self.df.shape[0] * 100).sort_values(ascending=False)

        self.logger.debug(f"Null occurrence: {null_occur_bool}")
        self.logger.debug(f"Total null count: {total_null_val}")
        self.logger.debug(f"Null count per column: {null_count}")
        self.logger.debug(f"Null percentage per column: {null_count_percent}")

        return null_occur_bool, null_count, total_null_val, null_count_percent

    def clean_unnecessary(self):
        removed_cols = ["ProductId", "UserId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator", "Time"]
        cleaned_df = self.df.drop(removed_cols, axis=1)
        self.logger.info(f"Removed columns: {removed_cols}")
        return cleaned_df

if __name__ == "__main__":
    model = NLP_Model()
    model.inspect_df()
    model.class_analysis()
    model.grab_col_names()
    model.null_values()


