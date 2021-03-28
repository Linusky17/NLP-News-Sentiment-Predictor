from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
import re
nltk.download(['punkt', 'wordnet'])


def main():
    st.set_page_config(page_title='Stock Price Prediction Using News Headlines', page_icon = "icon.png")
    st.sidebar.title("Stock Market DJIA Predictor Using News Headlines and Random Forest Classifier")
    st.title("Predicting Dow Jones Industrial Average Based on Headlines")
    predStarted = False
    if not predStarted:
        st.write("Use the sidebar to start the prediction")

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                footer:after {
                    content:'Made by The Birds Team: David Xue, Qasim Ali'; 
                    visibility: visible;
                    display: block;
                    position: relative;
                    #background-color: red;
                    padding: 5px;
                    top: 2px;
                }
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


    def loadCleanData():
        data = pd.read_csv('Combined_DJIA.csv')
        data['Top23'].fillna(data['Top23'].median, inplace=True)
        data['Top24'].fillna(data['Top24'].median, inplace=True)
        data['Top25'].fillna(data['Top25'].median, inplace=True)
        return data

    def mergeToDF(dataset):

        dataset = dataset.drop(columns=['Date', 'Label'])
        dataset.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
        for col in dataset.columns:
            dataset[col] = dataset[col].str.lower()
        headlines = []
        for row in range(0, len(dataset.index)):
            headlines.append(' '.join(str(x) for x in dataset.iloc[row, 0:25]))
        dataset = loadCleanData()
        df = pd.DataFrame(headlines, columns=['headlines'])
        df['label'] = dataset.Label
        df['date'] = dataset.Date
        return df

    def userInputCheck(data):
        data = data.lower()
        converted = pd.DataFrame([data], columns=['headlines'])
        converted.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

        return converted

    def tokenizeString(text):
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        clean_tokens = []
        for token in tokens:
            clean_token = lemmatizer.lemmatize(token).lower().strip()
            clean_tokens.append(clean_token)

        return clean_tokens

    def splitTrainTest(df):
        train = df[df['date'] < '20150101']
        test = df[df['date'] > '20141231']
        x_train = train.headlines
        y_train = train.label
        x_test = test.headlines
        y_test = test.label

        return x_train, x_test, y_train, y_test

    def buildTrainReport():
        st.subheader('Confusion Matrix')
        predictions = model.predict(x_test)
        matrix = confusion_matrix(y_test, predictions)
        #st.write(type(matrix))
        st.write(matrix)
        st.subheader('Classification Report')
        predictions = model.predict(x_test)
        report = classification_report(y_test, predictions)
        st.write(report)
        st.subheader('Model Accuracy Score')
        predictions = model.predict(x_test)
        score = accuracy_score(y_test, predictions)
        st.write("Model Accuracy Score from Test Set: ", score.round(3))

    def PipelineVectorize():
        vectPipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenizeString, stop_words='english')),
            ('tfidf', TfidfTransformer())
        ])
        return vectPipeline

    df = loadCleanData()
    df = mergeToDF(df)
    x_train, x_test, y_train, y_test = splitTrainTest(df)
    vector = PipelineVectorize()

    if st.sidebar.checkbox("show train/test raw data", False):
        st.subheader("Top 25 Headline News from Reddit")
        st.write(df)

    classifier = "Random Forest Classifier"




    if classifier == "Random Forest Classifier":
        st.sidebar.subheader("Random Forest Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "# of estimators", 50, 300, step=50, key='n_estimators')
        trainRes = st.sidebar.checkbox("show model training report", True)

        st.sidebar.subheader("Prediction")	
        userInputLine = st.sidebar.text_area("Enter Your News Headline for Prediction:", "Oklahoma Goodwill Employee Finds $42,000 Hidden in Donated Clothing –And Her Integrity Pays Off")

        fixedInput = "b "+ userInputLine
        h1 = fixedInput
        fixedInput = ""
        for i in range(25):
            fixedInput += h1
        fixedInput = userInputCheck(fixedInput)
        #st.write(f'{userInputLine.head(1)}')


        if st.sidebar.button("Predict", key="Predict"):
            predStarted = True
            st.spinner()
            if trainRes:
                st.header("Model Training Report using Kaggle Dataset")
                st.write("Uncheck Sidebar Option to Hide Report")
            x_train = vector.fit_transform(x_train)
            x_test = vector.transform(x_test)

            transInput = vector.transform(fixedInput)


            model = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy')
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)

            userPred = model.predict(transInput)

            if trainRes:
                buildTrainReport()
                st.markdown('---')
            st.header("Headline Input Prediction Result:")
            st.write("Headline to be used for prediction:")
            st.write(userInputLine)
            st.write(f'User Input Headline Prediction Result: {userPred[0]}')
            if userPred[0] == 1:
                st.markdown("**The Dow Jones Industrial Average will stay the same or more up due to this news headline. Buy!**")
            else:
                st.markdown("**The Dow Jones Industrial Average will decrease due to this news headline. Sell!**")

            st.markdown("*Note: The model gets retrained every time the predict button is pressed. The results may differ slightly when running multiple times*")



if __name__ == '__main__':
    main()

