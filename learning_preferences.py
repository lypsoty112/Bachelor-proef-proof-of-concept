import streamlit as st

from _startup import startup

import datetime

database, = startup()

database = database["learning-preferences"]

st.title("Finding your learning preferences")

st.write("Please provide the following information to help us understand your learning preferences.")

with st.form("learning_preferences_form") as form:
    username = st.text_input("Please write down a username to use while studying.", value="User")
    preferences = {"motivation":                                                                      st.text_area("What is your motivation for learning the subject or skill?",
                                                                                                                   placeholder="I want to learn. It's very interesting."),
                   "skill_level": st.selectbox(label="What is your skill level in this specific subject or skill?", options=["Novice", "Elementary", "Proficient", "Skilled", "Expert", "Master"]),

                   "learning_style": st.multiselect("What is your learning style?", options=["Visual", "Auditory", "Reading/Writing", "Interactive", "Repetitive", ]).extend(
                       st.text_input("Other learning style: (if not listed above)")), "study_habits": st.multiselect("What are your study habits?",
                                                                                                                     options=["Reading", "Taking notes", "Summarizing", "Mnemonics", "Mind mapping",
                                                                                                                              "Flashcards", "Group study", "Teaching others", "Practice tests",
                                                                                                                              "Time management", "Setting goals", "Taking breaks", "Exercising",
                                                                                                                              "Sleeping well", "Eating well", "Meditating", "Listening to music",
                                                                                                                              "Using apps", "Using online resources", "Via a tutor", ]).extend(
            st.text_input("Other study habits: (if not listed above)")),
                   "concentration_level": st.text_area("How would you describe your ability to concentrate?", placeholder="I can concentrate for ... periods of time."),
                   "preferred_environment": st.text_area("What is your preferred environment for learning?", placeholder="I prefer a... environment with..."),
                   "preferred_feedback": st.text_area("What is your preferred feedback system? For example, do you prefer positive reinforcement or "
                                                      "constructive criticism? How often do you want feedback?", placeholder="I prefer... feedback system. I want feedback every..."), }

    if st.form_submit_button("Submit"):
        preferences = {key: value for key, value in preferences.items() if value}
        preferences["_id"] = username
        preferences["timestamp"] = datetime.datetime.now().isoformat()

        # If the user has already submitted their preferences, update the existing preferences.
        existing_preferences = database.find_one({"_id": username})
        if existing_preferences:
            database.update_one({"_id": username}, {"$set": preferences})
            st.toast("✔ Your preferences have been updated.")
        else:
            database.insert_one(preferences)
            st.toast("✔ Your preferences have been saved.")

        # Clear the form
        st.stop()
