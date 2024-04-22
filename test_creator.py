"""
This file creates test questions for each chunk of the course.
"""

import streamlit as st

from _startup import startup
from src.chains.testCreatorChain import TestCreatorChain
from src.models.chains.testCreator import TestCreatorInput

database, = startup()
collection = database['course-info']

test_creator = TestCreatorChain()
test_creator.build()

st.title("Test Creator")
# Ask for the file name

# Get the available filename from the database
ids = [doc['_id'] for doc in collection.find({}, {'_id': 1})]
selected_id = st.selectbox("Select the course", ids)

if st.button("Generate the tests"):
    # Get the course information
    course_info = collection.find_one({'_id': selected_id})
    pct_done = 0
    amnt_chunks = len(course_info['results'])
    progress_bar = st.progress(pct_done)
    updated_course_info = []
    for i, chunk in enumerate(course_info['results']):
        pct_done = int((i + 1) / amnt_chunks * 100)
        questions = test_creator.run(TestCreatorInput(course=chunk["chunk"]["page_content"], learning_goals=chunk['learning goals']))
        # Overwrite the course information with the new test questions
        updated_course_info.append({**chunk, **questions.model_dump()})
        progress_bar.progress(pct_done)

    # Update the course information in the database
    collection.update_one({'_id': selected_id}, {'$set': {'results': updated_course_info}})
    st.toast("The tests have been generated")

