from pandas import DataFrame

from common_imports import *
from logging_config import *

class NlpModel:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing NLP_Model")
        
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data", "Reviews.csv")
        
        self.df = pd.read_csv(self.DATA_DIR)
        self.cleaned_df = self.clean_unnecessary()
        self.logger.info("NLP_Model initialized successfully")

    def print_section(self, title: str) -> None:
        """Prints a formatted section header"""
        line = "═" * 30
        print(Fore.CYAN + line + f" {title} " + line + Style.RESET_ALL)

    def class_analysis(self):
        """
        Analyzes the distribution of a specific column ('Score') in a DataFrame
        by counting the occurrences of each class (1 through 5) and calculating
        their respective percentages. Logs the detailed analysis for each class
        and prints the results to the console.

        :param self: An instance of the class containing a DataFrame attribute
                     (`cleaned_df`) and a logger (`logger`) that are used for
                     computation and logging.
        :type self: object

        :return: None
        """
        self.print_section("CLASS ANALYSIS")
        for i in range(1, 6):
            count = len(self.cleaned_df[self.cleaned_df["Score"] == i])
            percentage = 100 * count / len(self.cleaned_df)
            self.logger.debug(f"Class {i}: {count} values, {percentage:.2f}%")
            print(f"Number of values in class {i}: {count} | Percentage of class: {percentage:.2f}%")

    def inspect_df(self):
        """
        Analyzes and displays various details about the DataFrame, providing insights
        into its structure, content, and summary statistics. This utility function is
        helpful for debugging and understanding the DataFrame at a glance. The method
        is structured into distinct sections, providing comprehensive information
        including shape, null value counts, first few rows, general info, and summary
        statistics.

        :return: None
        """
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
        """
        Identifies and categorizes the columns of a DataFrame into categorical, numerical,
        and categorical but cardinal (high cardinality) columns based on specified thresholds.
        The method examines the types and uniqueness of columns to classify them.

        :param cat_th: Threshold for the maximum unique values a numerical column can have
            to be considered as a categorical column.
        :type cat_th: int
        :param car_th: Threshold for the minimum unique values a categorical column can have
            to be considered as high cardinality.
        :type car_th: int
        :return: A tuple containing three lists:
            1. cat_cols - Columns classified as categorical.
            2. num_cols - Columns classified as numerical.
            3. categorical_but_car - Columns classified as categorical but with high cardinality.
        :rtype: tuple[list, list, list]
        """
        categorical_cols = [col for col in self.df.columns if self.df[col].dtype == "O"]
        numerical_but_cat = [col for col in self.df.columns if self.df[col].nunique() < cat_th and self.df[col].dtype in ["int64", "float64"]]
        categorical_but_car = [col for col in categorical_cols if self.df[col].nunique() > car_th]
        cat_cols = [col for col in categorical_cols + numerical_but_cat if col not in categorical_but_car]
        num_cols = [col for col in self.df.columns if self.df[col].dtype in ["int64", "float64"] and col not in numerical_but_cat]

        self.logger.debug(f"Categorical columns: {cat_cols}")
        self.logger.debug(f"Numerical columns: {num_cols}")
        self.logger.debug(f"Categorical but cardinal columns: {categorical_but_car}")

        return cat_cols, num_cols, categorical_but_car

    def null_values(self) -> tuple[bool, int, int, float]:
        """
        Analyzes the presence and distribution of null values within the dataset.

        This method checks whether null values exist in the DataFrame, calculates the total
        number of null values, and determines the count and percentage of null values for each
        column. Results are logged for debugging purposes.

        :return: A tuple containing a boolean indicating the presence of any null values,
            a Series representing the count of null values by column, the total number of
            null values, and a Series representing the percentage of null values by column
        :rtype: tuple[bool, pandas.Series, int, pandas.Series]
        """
        null_occur_bool = self.df.isnull().values.any()
        null_count = self.df.isnull().sum().sort_values(ascending=False)
        total_null_val = null_count.sum()
        null_count_percent = (self.df.isnull().sum() / self.df.shape[0] * 100).sort_values(ascending=False)

        self.logger.debug(f"Null occurrence: {null_occur_bool}")
        self.logger.debug(f"Total null count: {total_null_val}")
        self.logger.debug(f"Null count per column: {null_count}")
        self.logger.debug(f"Null percentage per column: {null_count_percent}")

        return null_occur_bool, null_count, total_null_val, null_count_percent

    def clean_unnecessary(self) -> DataFrame:
        """
        Cleans the dataframe by removing unnecessary columns.

        This method removes a predefined set of columns from the dataframe. Additionally,
        a log entry is created recording which columns were removed. The cleaned dataframe
        is then returned for further use.

        :raises KeyError: Raised if the specified columns to be removed do not exist
            in the dataframe.
        :returns: A new dataframe without the removed columns.
        :rtype: pandas.DataFrame
        """
        removed_cols = ["ProductId", "UserId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator", "Time"]
        cleaned_df = self.df.drop(removed_cols, axis=1)
        self.logger.info(f"Removed columns: {removed_cols}")
        return cleaned_df

if __name__ == "__main__":
    model = NlpModel()
    model.inspect_df()
    model.class_analysis()
    model.grab_col_names()
    model.null_values()


