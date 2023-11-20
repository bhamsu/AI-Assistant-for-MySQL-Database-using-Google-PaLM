
import streamlit
from promptExecution import Database_Assistant

db_assistant = Database_Assistant()
streamlit.title("AI-BASED DATABASE ASSISTANT USING GOOGLE PALM")
question = streamlit.text_input("Question: ")

if question:
    # "How many white color Levi's shirt I have?"
    response = db_assistant(question)

    streamlit.header("Answer")
    streamlit.write(response)
