from common_imports import *

from nlpmodel import NlpModel
from logging_config import *

matplotlib.use("Agg")

class EDAModel(NlpModel):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing EDA_Model")
        
        self.text_data = self.cleaned_df["Text"]
        self.SAVE_DIR = os.path.join(self.BASE_DIR, "images", "plots")
        self.logger.info("EDA_Model initialized successfully")

    async def plot_histogram(self, data, x_label: str,
                             y_label: str, title: str,
                             filename: str):
        """
        Generates and saves a histogram plot based on the provided data using Seaborn and Matplotlib.
        It customizes the plot with labels, title, and other visual parameters and saves the
        output as an image file. Additionally, logs the save location to a file
        and outputs it to the console.

        :param data: Data to create the histogram from.
        :type data: list or pandas.Series
        :param x_label: Label for the x-axis.
        :type x_label: str
        :param y_label: Label for the y-axis.
        :type y_label: str
        :param title: Title for the histogram plot.
        :type title: str
        :param filename: The path and name of the file where the plot image should
            be saved.
        :type filename: str
        :return: None
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=30, kde=True)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, plt.savefig, filename)
        plt.close()

        async with aiofiles.open(os.path.join(self.SAVE_DIR, "log.txt"), "a") as log_file:
            await log_file.write(f"Histogram saved at: {filename}\n")

        self.logger.info(f"Histogram saved at: {filename}")
        print(f"Histogram saved at: {filename}")

    async def data_length_histogram(self):
        """
        Generate and save a histogram representing the distribution of text data length
        based on character count. The histogram is generated from the text data stored
        within the instance, and the output is saved as an image file for further use.

        :param self: Instance of the class that contains text data and methods required
            for generating and saving the histogram.
        :type self: object
        :raises OSError: Raised if the directory creation fails for any reason.
        :raises ValueError: Raised if text data is invalid or unavailable for computation.
        :return: None
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        FILE_NAME = f"DATA_LENGTH_HIST_{timestamp}.png"
        LOCAL_SAVE_DIR = os.path.join(self.SAVE_DIR, "DATA_LENGTH_HIST", FILE_NAME)
        os.makedirs(os.path.dirname(LOCAL_SAVE_DIR), exist_ok=True)

        self.logger.info("Generating data length histogram")
        await self.plot_histogram(
            self.text_data.str.len(),
            "Number of Characters", "Frequency",
            "Distribution of Text Lengths (Character Count)",
            LOCAL_SAVE_DIR
        )

    async def data_word_length_histogram(self):
        """
        Generates a histogram for the distribution of the number of words in the text
        data and saves the resulting plot to a specified directory. The histogram
        visualizes the frequency distribution of word counts in the text data, providing
        insight into the textual structure or length distribution.

        :raises FileNotFoundError: If the SAVE_DIR or specified directories cannot
            be accessed or created.
        :raises TypeError: If the input text_data is not in the expected format.
        :raises Exception: For any general errors that occur during the histogram
            generation or saving process.

        :return: None
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        FILE_NAME = f"DATA_WORD_LENGTH_HIST_{timestamp}.png"
        LOCAL_SAVE_DIR = os.path.join(self.SAVE_DIR, "DATA_WORD_LENGTH_HIST", FILE_NAME)
        os.makedirs(os.path.dirname(LOCAL_SAVE_DIR), exist_ok=True)

        self.logger.info("Generating data word length histogram")
        await self.plot_histogram(
            self.text_data.str.split().map(len),
            "Number of Words", "Frequency",
            "Distribution of Text Lengths (Word Count)",
            LOCAL_SAVE_DIR
        )

    async def data_word_average_length_histogram(self):
        """
        Generates a histogram displaying the distribution of average word lengths
        in the text data and saves it as a PNG file in a specified directory. It
        computes the average word length per text entry and uses this data to
        create and save the histogram.

        :raises OSError: Raised if there is an issue creating the required
            directory to save the generated file.
        :return: None
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        FILE_NAME = f"DATA_WORD_AVERAGE_LENGTH_HIST_{timestamp}.png"
        LOCAL_SAVE_DIR = os.path.join(self.SAVE_DIR, "WORD_AVG_LENGTH_HIST", FILE_NAME)
        os.makedirs(os.path.dirname(LOCAL_SAVE_DIR), exist_ok=True)

        self.logger.info("Generating data word average length histogram")
        await self.plot_histogram(
            self.text_data.str.split().apply(lambda x: np.mean([len(i) for i in x])),
            "Average Word Length", "Frequency",
            "Distribution of Average Word Lengths",
            LOCAL_SAVE_DIR
        )

async def main():
    model = EDAModel()
    await asyncio.gather(
        model.data_length_histogram(),
        model.data_word_length_histogram(),
        model.data_word_average_length_histogram()
    )

if __name__ == "__main__":
    asyncio.run(main())