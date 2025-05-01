# Configuration loading module 
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_environment_variables():
    """Loads environment variables from .env file."""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if not os.path.exists(env_path):
        logger.warning(f".env file not found at {env_path}. Please create it based on .env.example.")
        # Attempt to load from system environment anyway
        load_dotenv()
    else:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loaded environment variables from {env_path}")

class Settings(BaseModel):
    """Pydantic model for application settings, loaded from environment variables."""
    pinecone_api_key: str = Field(..., alias='PINECONE_API_KEY')
    # Environment is no longer required for Pinecone client initialization (v6+)
    # pinecone_environment: str = Field(..., alias='PINECONE_ENVIRONMENT')
    openai_api_key: str = Field(..., alias='OPENAI_API_KEY')
    # Added Pinecone index configuration
    pinecone_index_name: str = Field(default="sephora-products", alias='PINECONE_INDEX_NAME')
    pinecone_embedding_dim: int = Field(default=1536, alias='PINECONE_EMBEDDING_DIM')

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore' # Ignore extra variables defined in .env
        # If using Pydantic v1 style BaseSettings, replace above with:
        # env_prefix = '' # No prefix
        # case_sensitive = False

settings: Settings = None

def get_settings() -> Settings:
    """Loads and validates settings. Raises SystemExit if validation fails."""
    global settings
    if settings is None:
        load_environment_variables()
        try:
            # Pydantic v2 automatically reads from environment variables
            # based on field names/aliases if env_file is not explicitly loaded
            # We use os.getenv here as a fallback/explicit read after load_dotenv
            settings_data = {
                'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
                'PINECONE_ENVIRONMENT': os.getenv('PINECONE_ENVIRONMENT'),
                'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY')
            }
            settings = Settings(**settings_data)
            logger.info("Settings loaded and validated successfully.")

        except ValidationError as e:
            missing_vars = []
            for error in e.errors():
                if error['type'] == 'value_error.missing':
                     # Attempt to get the original alias if available
                    field_alias = Settings.model_fields.get(error['loc'][0], {}).alias
                    missing_vars.append(field_alias or error['loc'][0])

            logger.error(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
            logger.error("Please ensure they are set in your .env file or system environment.")
            # Optionally, raise an exception or exit
            raise SystemExit("Missing required environment variables.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading settings: {e}")
            raise SystemExit("Failed to load settings.")
    return settings

# Load settings on module import
current_settings = get_settings()

if __name__ == '__main__':
    # Example of accessing settings
    # This part will only run if the script is executed directly
    # In a real app, other modules would import `current_settings`
    print("Attempting to load settings...")
    try:
        loaded_settings = get_settings()
        print("Settings loaded successfully!")
        # Avoid printing sensitive keys directly
        print(f"Pinecone Environment: {loaded_settings.pinecone_environment}")
        print(f"OpenAI API Key Loaded: {bool(loaded_settings.openai_api_key)}")
        print(f"Pinecone API Key Loaded: {bool(loaded_settings.pinecone_api_key)}")
    except SystemExit as e:
        print(f"Failed to load settings: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 