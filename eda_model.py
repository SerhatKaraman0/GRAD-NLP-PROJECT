from common_imports import *
from nlp_model import NLP_Model
import logging
from logging_config import *

matplotlib.use("Agg")

class EDA_Model(NLP_Model):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing EDA_Model")
        
        self.text_data = self.cleaned_df["Text"]
        self.SAVE_DIR = os.path.join(self.BASE_DIR, "images", "plots")
        self.logger.info("EDA_Model initialized successfully")

    async def plot_histogram(self, data, xlabel, ylabel, title, filename):
        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=30, kde=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
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
    model = EDA_Model()
    await asyncio.gather(
        model.data_length_histogram(),
        model.data_word_length_histogram(),
        model.data_word_average_length_histogram()
    )

if __name__ == "__main__":
    asyncio.run(main())