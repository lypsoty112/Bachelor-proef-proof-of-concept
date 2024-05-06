import pandas as pd

from _startup import startup
from src.chains.testCorrectorChain import TestCorrectorChain
import streamlit as st
import random

database, = startup()
testCorrector: TestCorrectorChain | None = None

st.sidebar.title("Test")

collection = database['course-info']

# Get the available filename from the database
ids = [doc['_id'] for doc in collection.find({}, {'_id': 1})]
selected_id = st.sidebar.selectbox("Select the course", ids)

amount_of_questions = st.sidebar.number_input("Number of questions", min_value=1, max_value=100, step=1, value=10)

if "questions" not in st.session_state:
    st.session_state.questions = []
if st.sidebar.button("Test me!"):
    # Get the questions
    course_info = collection.find_one({'_id': selected_id})

    # Get the questions
    questions = [q for x in course_info["results"] for q in x["questions"]]
    # Randomly select the questions
    st.session_state.questions = random.sample(questions, amount_of_questions)

inputs = [{"answer": st.text_area(f"Question {i + 1}: **{question['question']}**"), "question": question} for i, question in enumerate(st.session_state.questions)]

submitted = st.button("Submit")

if submitted:
    testCorrector = TestCorrectorChain()
    testCorrector.build()
    results = []

    progressbar = st.progress(0, text="Evaluating answers...")
    i = 0
    for answer in inputs:
        provided_answer = answer["answer"] if answer["answer"].strip() else "No answer provided"
        result = testCorrector.run({
            "question": answer["question"]["question"], "student_answer": provided_answer, "expected_answer": answer["question"]["answer"],
        })
        i += 1
        results.append({
            "Question": answer["question"]["question"], "Student's Answer": provided_answer, "Expected Answer": answer["question"]["answer"], "Grade": result.grade, "Reasoning": result.reasoning
        })
        progressbar.progress(int(i / len(inputs) * 100), text=f"Evaluating answer {i}...")

    # Provide the answers as a table
    df = pd.DataFrame.from_records(results)

    total_grade = df["Grade"].mean()
    if total_grade <= 25:
        st.error(f"Total grade: {total_grade:.2f}/100. You need to study more.")
    elif total_grade <= 50:
        st.warning(f"Total grade: {total_grade:.2f}/100. You need to study more.")
    elif total_grade <= 75:
        st.info(f"Total grade: {total_grade:.2f}/100. You're doing well.")
    else:
        st.success(f"Total grade: {total_grade:.2f}/100. You're doing great!")


    st.dataframe(df)
