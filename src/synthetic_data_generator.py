#!/usr/bin/env python3
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import format_instructions
from pydantic import BaseModel, Field
import json
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import random

# Load environment variables
load_dotenv()

# Example data for few-shot learning
EXAMPLE_DATA = [
    # Electronics
    {
        "example": 'product_id: 1, category: Electronics, brand: Sony, condition: New, price: 950, seller_reputation: 4, description: "Brand new Sony Xperia 1", suspected_fraud: false'
    },
    {
        "example": 'product_id: 2, category: Electronics, brand: Xiaomi, condition: Used, price: 300, seller_reputation: 2, description: "Used Xiaomi Mi 9", suspected_fraud: false'
    },
    {
        "example": 'product_id: 3, category: Electronics, brand: OnePlus, condition: New, price: 700, seller_reputation: 5, description: "Brand new OnePlus 9 Pro", suspected_fraud: false'
    },
    {
        "example": 'product_id: 4, category: Electronics, brand: Huawei, condition: Used, price: 400, seller_reputation: 3, description: "Used Huawei P30", suspected_fraud: false'
    },
    {
        "example": 'product_id: 5, category: Electronics, brand: Google, condition: New, price: 850, seller_reputation: 5, description: "Brand new Google Pixel 5", suspected_fraud: false'
    },
    {
        "example": 'product_id: 6, category: Electronics, brand: Motorola, condition: Used, price: 250, seller_reputation: 4, description: "Used Motorola Edge", suspected_fraud: false'
    },
    {
        "example": 'product_id: 7, category: Electronics, brand: Oppo, condition: New, price: 600, seller_reputation: 4, description: "Brand new Oppo Find X3", suspected_fraud: false'
    },
    # Clothing
    {
        "example": 'product_id: 8, category: Clothing, brand: Puma, condition: New, price: 60, seller_reputation: 3, description: "Brand new Puma hoodie", suspected_fraud: false'
    },
    {
        "example": 'product_id: 9, category: Clothing, brand: Under Armour, condition: Used, price: 25, seller_reputation: 2, description: "Used Under Armour running shoes", suspected_fraud: false'
    },
    {
        "example": 'product_id: 10, category: Clothing, brand: Gucci, condition: New, price: 200, seller_reputation: 3, description: "Brand new Gucci handbag", suspected_fraud: True'
    },
    {
        "example": 'product_id: 11, category: Clothing, brand: Reebok, condition: Used, price: 35, seller_reputation: 3, description: "Used Reebok workout gear", suspected_fraud: false'
    },
    {
        "example": "product_id: 12, category: Clothing, brand: Levi's, condition: New, price: 45, seller_reputation: 4, description: \"Brand new Levi's jeans\", suspected_fraud: false"
    },
    {
        "example": 'product_id: 13, category: Clothing, brand: Zara, condition: Used, price: 20, seller_reputation: 2, description: "Used Zara summer dress", suspected_fraud: false'
    },
    {
        "example": 'product_id: 14, category: Clothing, brand: Hugo Boss, condition: New, price: 150, seller_reputation: 3, description: "Brand new Hugo Boss blazer", suspected_fraud: True'
    },
    {
        "example": 'product_id: 15, category: Clothing, brand: Uniqlo, condition: Used, price: 15, seller_reputation: 1, description: "Used Uniqlo T-shirt", suspected_fraud: false'
    },
    # Furniture
    {
        "example": 'product_id: 16, category: Furniture, brand: Burrow, condition: New, price: 1200, seller_reputation: 5, description: "Brand new Burrow modular sofa", suspected_fraud: false'
    },
    {
        "example": 'product_id: 17, category: Furniture, brand: FLOYD, condition: Used, price: 800, seller_reputation: 4, description: "Lightly used FLOYD sectional sofa", suspected_fraud: false'
    },
    {
        "example": 'product_id: 18, category: Furniture, brand: Maiden Home, condition: New, price: 2000, seller_reputation: 5, description: "Luxury Maiden Home leather armchair", suspected_fraud: false'
    },
    {
        "example": 'product_id: 19, category: Furniture, brand: Castlety, condition: New, price: 1500, seller_reputation: 3, description: "Brand new Castlety mid-century dining table", suspected_fraud: true'
    },
    {
        "example": 'product_id: 20, category: Furniture, brand: Benchmade Modern, condition: Used, price: 1800, seller_reputation: 4, description: "Used but excellent condition Benchmade Modern sofa", suspected_fraud: false'
    },
    {
        "example": 'product_id: 21, category: Furniture, brand: Burrow, condition: New, price: 1700, seller_reputation: 5, description: "Brand new Burrow coffee table and TV stand", suspected_fraud: false'
    },
    {
        "example": 'product_id: 22, category: Furniture, brand: Maiden Home, condition: New, price: 2200, seller_reputation: 4, description: "Brand new Maiden Home velvet sectional sofa", suspected_fraud: false'
    },
    {
        "example": 'product_id: 23, category: Furniture, brand: Castlety, condition: Used, price: 1300, seller_reputation: 3, description: "Used Castlety king-size bed frame", suspected_fraud: false'
    },
    {
        "example": 'product_id: 24, category: Furniture, brand: FLOYD, condition: New, price: 950, seller_reputation: 4, description: "Brand new FLOYD bed frame with storage", suspected_fraud: false'
    },
    {
        "example": 'product_id: 25, category: Furniture, brand: Benchmade Modern, condition: New, price: 2100, seller_reputation: 5, description: "Custom Benchmade Modern sectional", suspected_fraud: true'
    },
]



class ProductListing(BaseModel):
    product_id: int = Field(..., description="Unique identifier for the product", example=1234)
    category: str = Field(..., description="Product category (Electronics, Clothing, or Furniture)", example="Electronics")
    brand: str = Field(..., description="Brand name", example="Samsung")
    condition: str = Field(..., description="Product condition (New or Used)", example="New")
    price: float = Field(..., description="Product price in USD", gt=0, example=1200.00)
    seller_reputation: int = Field(..., description="Seller reputation score (1-5)", ge=1, le=5, example=4)
    description: str = Field(..., description="Detailed product description", example="Brand new Samsung Galaxy S21 Ultra with 256GB storage")
    suspected_fraud: bool = Field(..., description="Whether the listing is suspected of fraud", example=False)

    model_config = {
        "arbitrary_types_allowed": True
    }


class DataGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.parser = PydanticOutputParser(pydantic_object=ProductListing)
        self.categories = ["Electronics", "Clothing", "Furniture"]

        examples_text = "\n".join(
            [f"Example {i+1}:\n{ex['example']}" for i, ex in enumerate(EXAMPLE_DATA)]
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                     "system",
            """You are a synthetic data generator specializing in e-commerce product listings. 
            Your task is to generate **realistic but fictional** product data that adheres to the provided structure.
            
            ### **Rules for Data Generation**
            - **Product ID**: Must be a unique integer and serve as the primary key.
            - **Pricing Guidelines**:
                - Electronics: $200 - $2500
                - Clothing: $10 - $500
                - Furniture: $100 - $5000
            - **Seller Reputation**: An integer between 1 and 5.
            - **Fraud Probability**: Each product should have a **15-20% chance** of `suspected_fraud=True`.
            - **Condition**: Must be either `"New"` or `"Used"`, distributed realistically.
            - **Description**: Must be concise but informative.
            - **Brand Diversity**: Avoid excessive repetition of brands.
            - **Formatting**: Follow the exact structure of the provided examples.
            - **Realism**: No duplicate `product_id` values and no unrealistic entries.

            ### **Example Data Structure**
            ```
            {examples}
            ```

            Ensure the generated output follows this exact structure with **unique** `product_id` values and realistic fraud probabilities.
            """,
        ),
        (
            "user",
            """Generate a new **realistic** product listing for the **{category}** category.  
            - Ensure `product_id` is **unique** and sequentially increasing.  
            - `suspected_fraud` should be **randomly set** to `True` **15-20%** of the time.  
            - Output must **strictly** follow this format:  

            {format_instructions}
            """,
                ),
            ]
        )

    def generate_single_record(self, category: str) -> ProductListing:
        formatted_prompt = self.prompt.format_messages(
            examples="\n".join([f"{ex['example']}" for ex in EXAMPLE_DATA]),
            category=category,
            format_instructions=self.parser.get_format_instructions(),
        )
        try:
            response = self.llm.invoke(formatted_prompt)
            return self.parser.parse(response.content)
        except Exception as e:
            print(f"Error generating record: {e}")
            return None

    def generate_dataset(self, num_records: int) -> pd.DataFrame:
        records = []
        with tqdm(total=num_records, desc="Generating records") as pbar:
            while len(records) < num_records:
                category = random.choice(self.categories)
                record = self.generate_single_record(category)
                if record is not None:
                    records.append(record.dict())
                    pbar.update(1)

        return pd.DataFrame(records)


def main():
    # Initialize the generator
    generator = DataGenerator()

    # Generate 1000 synthetic product listingsa
    print("Generating 1000 synthetic product listings...")
    df = generator.generate_dataset(10)

    # Save to CSV
    output_file = "/Users/Aaron/synthetic-data-ml-project/data/synthetic_product_listings.csv"
    df.to_csv(output_file, index=False)
    print(f"\nGenerated synthetic product listings and saved to '{output_file}'")
    print("\nDataset Statistics:")
    print(f"Total Records: {len(df)}")
    print("\nCategory Distribution:")
    print(df["category"].value_counts())
    print("\nFraud Distribution:")
    print(df["suspected_fraud"].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))
    print("\nSample of generated data:")
    print(df.head())


if __name__ == "__main__":
    main()
