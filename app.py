import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Hello Streamlit")
st.title("Hello Streamlit!")
st.write("Streamlit is running.")