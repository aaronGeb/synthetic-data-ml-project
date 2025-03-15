#!/usr/bin/env python3
import openai
import os
import csv
import pandas as pd
from dotenv import load_dotenv


class SyntheticDatasetGenerator:
    """Class to generate and save a synthetic dataset using OpenAI's API."""

    def __init__(self):
        """Initialize the generator by loading the API key from environment variables."""
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )

        openai.api_key = self.api_key

    def generate_synthetic_dataset(self, num_rows=1000):
        """Generates a synthetic dataset using OpenAI's chat model and returns it as a list of lists."""

        prompt = f"""
        Generate a synthetic dataset in CSV format with the following columns: user_id, product_id, rating, and timestamp.
        - user_id: a random integer between 1 and 1000.
        - product_id: a random integer between 1 and 500.
        - rating: a random integer between 1 and 5.
        - timestamp: a random datetime within the past year, formatted as YYYY-MM-DD HH:MM:SS.
        Generate {num_rows} rows, without a header. Just the data.
        """

        try:
            client = openai.OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )

            dataset_string = response.choices[0].message.content.strip()

            # Convert string response to list of lists
            dataset = [row.split(",") for row in dataset_string.split("\n")]

            return dataset

        except openai.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def save_dataset_to_csv(
        self,
        dataset,
        filename="/Users/Aaron/synthetic-data-ml-project/data/user_interactions.csv",
    ):
        """Saves the dataset to a CSV file using pandas."""
        try:
            df = pd.DataFrame(
                dataset, columns=["user_id", "product_id", "rating", "timestamp"]
            )
            df.to_csv(filename, index=False)
            print(f"Synthetic dataset saved as {filename}")

        except Exception as e:
            print(f"Error saving to CSV: {e}")


if __name__ == "__main__":
    generator = SyntheticDatasetGenerator()
    synthetic_data = generator.generate_synthetic_dataset(num_rows=1000)

    if synthetic_data:
        generator.save_dataset_to_csv(synthetic_data)
