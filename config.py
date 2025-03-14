import os
import dotenv

# load the .env file
dotenv.load_dotenv()

ZONING_DATA_PATH = os.getenv("DATA_PATH")