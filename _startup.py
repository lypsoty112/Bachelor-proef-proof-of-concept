import dotenv
from typing import Any, Mapping, Tuple
from pymongo import MongoClient
import os
import logging

from pymongo.database import Database


def startup() -> tuple[Database[Mapping[str, Any] | Any]]:
    dotenv.load_dotenv()

    logging.basicConfig(level=logging.INFO)

    client = MongoClient(os.getenv("MONGODB_URI"))["BACHELORS-THESIS_PROOF-OF-CONCEPT"]
    return (client,)
