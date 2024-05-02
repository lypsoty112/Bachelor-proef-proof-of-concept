"""
This file is a teacher, which will be used to teach the student the HR-course.
"""
import json

import streamlit as st

from _startup import startup
from src.chains.teacherChain import TeacherChain
from src.models.chains.chat import ChatInput
import os

database, = startup()
teacherChain: TeacherChain | None = None

course_collection = database["course-info"]
user_collection = database["learning-preferences"]

if "database_content" not in st.session_state:
    st.session_state["database_content"] = []

if "learning_preferences" not in st.session_state:
    st.session_state["learning_preferences"] = {}

if "metadata" not in st.session_state:
    st.session_state["metadata"] = {}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "tests" not in st.session_state:
    st.tests = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

ids = [doc['_id'] for doc in course_collection.find({}, {'_id': 1})]
selected_id = st.sidebar.selectbox("Select the course", ids)

user_ids = [doc['_id'] for doc in user_collection.find({}, {'_id': 1})]
selected_user_id = st.sidebar.selectbox("Select the user", user_ids)

if st.sidebar.button("Select"):
    database_content = course_collection.find_one({'_id': selected_id})
    st.session_state["database_content"] = database_content["results"]
    st.session_state["learning_preferences"] = user_collection.find_one({'_id': selected_user_id})
    st.session_state["teacherChain"] = TeacherChain(course_contents=database_content["results"], user_preferences=st.session_state["learning_preferences"])
    st.session_state["teacherChain"].build()

    starting_message = st.session_state["teacherChain"].starting_message().response
    st.session_state.messages = [{"role": starting_message.role, "content": starting_message.content}]
    with st.chat_message(starting_message.role):
        st.markdown(starting_message.content)
    st.toast(f"Course '{selected_id}' & user selected.", icon="✔")

# Only show chat input if course and user are selected
if "teacherChain" in st.session_state:
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        response = st.session_state["teacherChain"].run(data=ChatInput(messages=st.session_state.messages, metadata=st.session_state["metadata"]))

        # Process the metadata
        if response.metadata.get("type") == "questions":
            # Process the questions
            questions = json.loads(response.response.content)
            st.write("Questions:")
            for question in questions:
                st.write(question["question"])
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response.response.content)
        # Add assistant response to chat history
        st.session_state["metadata"] = response.metadata
        st.session_state["messages"].append({"role": response.response.role, "content": response.response.content})
        if os.getenv("LOGGING_LEVEL") == "DEBUG":
            st.sidebar.title("Metadata")
            st.sidebar.json(st.session_state["metadata"])
