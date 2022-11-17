import streamlit as st
import pandas as pd
import joblib


st.header("University Admit Eligibility Predictor")


GREScore = st.number_input("Gaduate Record Examination Score (250-340) ")
TOEFLScore = st.number_input("Test of English as a Foreign LanguageScore (50-120)")
UniversityRating = st.number_input("University Rating Value (1-5)")
SOP = st.number_input("Statement of Purpose Score (1-5)")
LOR = st.number_input("Letter of Recommendation Score (1-5)")
CGPA = st.number_input("Cumulative Grade Point Average (5-10)")

Research = st.selectbox("Research or Not", ("Research", "Not"))


if st.button("Submit"):
    clf = joblib.load("model.pkl")
    X = pd.DataFrame([[GREScore,TOEFLScore,UniversityRating,SOP,LOR,CGPA,Research]], 
                     columns = ["GREScore","TOEFLScore","UniversityRating","SOP","LOR","CGPA","Research"])
    X = X.replace(["Research", "Not"], [1, 0])
    prediction = clf.predict(X)[0]
    st.subheader(f"The possibility is {prediction}")