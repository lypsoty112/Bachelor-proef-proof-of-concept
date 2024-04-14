import dotenv
from typing import Tuple
from pymongo import MongoClient
import os
import logging


def startup() -> Tuple[MongoClient,]:
    dotenv.load_dotenv()

    logging.basicConfig(level=logging.INFO)

    client = MongoClient(os.getenv("MONGODB_URI"))

    return (client,)
