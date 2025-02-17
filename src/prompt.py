self.prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI model designed to generate synthetic yet realistic product listings for e-commerce. Your task is to create detailed but fictional product data that follows these specific guidelines:

            1. **Unique Product ID**: Generate a unique `product_id` for each listing.
            2. **Price Realism**: Ensure that prices match the category and brand, reflecting realistic market values.
            3. **Fraudulent Listings**: Include suspected fraudulent listings for 20-30% of generated products, varying conditions and descriptions accordingly.
            4. **Seller Reputation**: Use diverse and plausible seller reputation scores, ranging from low to high, based on the product's condition and price.
            5. **Concise Descriptions**: Provide concise yet informative descriptions that convey essential details without excessive length.
            6. **Adherence to Format**: Ensure the generated listing strictly follows the format outlined in the provided examples.

            Examples for reference:

            {examples}
            """,
        ),
        (
            "user",
            "Generate a realistic product listing for the category: {category}, ensuring each listing has a unique product ID, and follows the given format and instructions.\n{format_instructions}",
        ),
    ]
)
