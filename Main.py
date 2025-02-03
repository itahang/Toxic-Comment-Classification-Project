import streamlit as st 

md= open("Contents/MainPage.md").read()

st.markdown(md)