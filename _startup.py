import dotenv
from typing import Tuple
from pymongo import MongoClient
import os


def startup() -> Tuple[
    MongoClient,
]:
    dotenv.load_dotenv()

    client = MongoClient(
        os.getenv("MONGODB_URI")
    )

    return (client,)

