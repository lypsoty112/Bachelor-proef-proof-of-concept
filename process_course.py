"""
This file attempts to process the course data from the course data file.
It chunks the data, and then processes the data. For each processed chunk, learning goals & a learning plan are generated
"""
import io

import streamlit as st
from fastapi import UploadFile
from httpx import Headers

from _startup import startup
from src.chains.plannerChain import PlannerChain
from src.models.chains.planner import PlannerInput

database, = startup()
chain = PlannerChain()
chain.build()

st.title("Course processor")

database = database["course-info"]


def convert_to_upload_file(file_to_convert):
    # Create a BytesIO object from the uploaded file data
    file_bytes = file_to_convert.getvalue()
    file_stream = io.BytesIO(file_bytes)

    # Create a FastAPI UploadFile object
    upload_file = UploadFile(file_stream, filename=file_to_convert.name, headers=Headers({"content-type": "application/pdf"}))
    return upload_file


if file := st.file_uploader("Upload the course here", type=["pdf"], accept_multiple_files=False):
    st.toast("File uploaded successfully")

name = st.text_input("Enter the name of the course")

if st.button("Process course") and file:
    if not name:
        st.toast("Please enter the name of the course")
        st.stop()
    results = chain.run(PlannerInput(file=convert_to_upload_file(file)))
    # Process the results
    converted_results = []
    db_to_write = {}
    for plan, document in zip(results.plan, results.chunks):
        converted_results.append({"learning goals": plan, "chunk": {"page_content": document.page_content, "metadata": document.metadata}})
        # Add an ID
        db_to_write["results"] = converted_results
        db_to_write["_id"] = name

    database.insert_one(db_to_write)
    st.success("Course processed successfully")
