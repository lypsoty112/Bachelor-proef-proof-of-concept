import datetime
import io

import streamlit as st
from fastapi import UploadFile
from starlette.datastructures import Headers

from _startup import startup
from src.chains.plannerChain import PlannerChain
from src.models.chains.planner import PlannerInput


def convert_to_upload_file(file):
    # Create a BytesIO object from the uploaded file data
    file_bytes = file.getvalue()
    file_stream = io.BytesIO(file_bytes)

    # Create a FastAPI UploadFile object
    upload_file = UploadFile(file_stream, filename=file.name, headers=Headers({
        "content-type": "application/pdf"
    }))
    return upload_file


database, = startup()

database = database["BT-POC"]["learning-preferences"]
planner = PlannerChain()
planner.build()

st.sidebar.title("Teacher")
if uploaded_file := st.sidebar.file_uploader("Upload your HR-course file here. This file should be a pdf file.", type=["pdf"]):
    # Process the file
    # Convert the file to FastAPI UploadFile
    planner_result = planner.run(PlannerInput(file=convert_to_upload_file(uploaded_file)))
    # TODO: Generate a first message
    st.session_state.messages.append({
        "role": "assistant", "content": "Welcome to the course. I am your teacher.py. How can I help you?", "timestamp": datetime.datetime.now()
    })

st.title("Bachelor Thesis Proof of Concept")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if len(st.session_state.messages) == 0:
    st.warning("Please upload the HR-course file to start learning the course.")
else:
    if prompt := st.chat_input("Type your message here"):
        st.session_state.messages.append({
            "role": "user", "content": prompt, "timestamp": datetime.datetime.now()
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # TODO:  Respond
        st.session_state.messages.append({
            "role": "teacher.py", "content": "I am sorry, I am not able to respond to your message at the moment.", "timestamp": datetime.datetime.now()
        })
