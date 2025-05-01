from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Optional, List

class ProductData(BaseModel):
    """Pydantic model for storing extracted Sephora product data."""
    product_name: str = Field(..., description="The name of the product.")
    url: HttpUrl = Field(..., description="The canonical URL of the product page.")
    price: Optional[float] = Field(None, description="The price of the product.")
    currency: Optional[str] = Field(None, description="The currency code (e.g., USD).")
    stock_info: Optional[str] = Field(None, description="Information about product availability (e.g., 'In Stock', 'Out of Stock').")
    image_url: Optional[HttpUrl] = Field(None, description="URL of the main product image.")
    description: Optional[str] = Field(None, description="The product description text.")
    # Using str for ingredients for simplicity; could be List[str] if reliably parsed
    ingredients_list: Optional[str] = Field(None, description="Full list of ingredients.")
    how_to_use: Optional[str] = Field(None, description="Instructions on how to use the product.")

    # Validator for price (using Pydantic v1/v2 compatible decorator)
    @validator('price', pre=True, always=True)
    def check_price_non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError('Price must be non-negative')
        return v

    def to_pinecone_dict(self) -> dict:
        """Serializes the model to a dictionary suitable for Pinecone metadata.

        Ensures all values are primitive types (str, float, int, bool, list[str])
        and removes fields with None values.
        HttpUrl fields are automatically converted to strings by model_dump(mode='json').
        """
        # Use model_dump for Pydantic v2
        # exclude_none=True removes fields that were not set or explicitly set to None
        # mode='json' ensures types like HttpUrl, Decimal, etc., are serialized to JSON-compatible types (like strings)
        pinecone_meta = self.model_dump(mode='json', exclude_none=True)

        # Ensure URL is included as it's a required field and useful metadata, even if used as ID
        # model_dump(mode='json') already converts HttpUrl to string
        # pinecone_meta['url'] = str(self.url) # No longer needed due to mode='json'

        # Optional: Further check/flattening if complex types remained (shouldn't with this model)
        # for key, value in pinecone_meta.items():
        #     if isinstance(value, (dict, list)) and not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        #           logger.warning(f"Field '{key}' might not be Pinecone compatible: {value}. Attempting conversion or skipping.")
        #           # Add specific conversion logic here if needed

        return pinecone_meta

    class Config:
        # Example for generating JSON schema (optional)
        # schema_extra = {
        #     "example": {
        #         "product_name": "Ultra Facial Cleanser",
        #         "url": "https://www.sephora.com/product/ultra-facial-cleanser-P122764",
        #         "price": 24.00,
        #         "currency": "USD",
        #         "stock_info": "In Stock",
        #         "image_url": "https://www.sephora.com/productimages/sku/s554201-main-zoom.jpg",
        #         "description": "A mild facial cleanser formulated for all skin types.",
        #         "ingredients_list": "Aqua/Water, Sodium Laureth Sulfate, Decyl Glucoside, Glycerin, ...",
        #         "how_to_use": "Apply a small amount to clean fingertips..."
        #     }
        # }
        validate_assignment = True # Ensure type checks on assignment 