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
    st.sidebar.title("Stock Market DJIA Predictor News Headlines and Random Forest Classifier")

    def load_data():
        data = pd.read_csv(
            'Combined_DJIA.csv')

        data['Top23'].fillna(data['Top23'].median, inplace=True)
        data['Top24'].fillna(data['Top24'].median, inplace=True)
        data['Top25'].fillna(data['Top25'].median, inplace=True)

        return data

    def create_df(dataset):

        dataset = dataset.drop(columns=['Date', 'Label'])
        dataset.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
        for col in dataset.columns:
            dataset[col] = dataset[col].str.lower()

        headlines = []
        for row in range(0, len(dataset.index)):
            headlines.append(' '.join(str(x) for x in dataset.iloc[row, 0:25]))

        dataset = load_data()

        df = pd.DataFrame(headlines, columns=['headlines'])
        df['label'] = dataset.Label
        df['date'] = dataset.Date

        return df
    def userInputCheck(data):
        data = data.lower()
        converted = pd.DataFrame([data], columns=['headlines'])
        converted.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

        return converted

    def tokenize(text):
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for token in tokens:
            clean_token = lemmatizer.lemmatize(token).lower().strip()
            clean_tokens.append(clean_token)

        return clean_tokens

    def split(df):

        train = df[df['date'] < '20150101']
        test = df[df['date'] > '20141231']
        x_train = train.headlines
        y_train = train.label
        x_test = test.headlines
        y_test = test.label

        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):

        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            predictions = model.predict(x_test)
            matrix = confusion_matrix(y_test, predictions)
            st.write("Confusion Matrix ", matrix)

        if 'Classification_Report' in metrics_list:
            st.subheader('Classification_Report')
            predictions = model.predict(x_test)
            report = classification_report(y_test, predictions)
            st.write("Classification_Report ", report)

        if 'Accuracy_Score' in metrics_list:
            st.subheader('Accuracy_Score')
            predictions = model.predict(x_test)
            score = accuracy_score(y_test, predictions)
            st.write("Accuracy_Score: ", score.round(2))

    def Vectorize():
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, stop_words='english')),
            ('tfidf', TfidfTransformer())
        ])
        return pipeline

    df = load_data()
    df = create_df(df)
    x_train, x_test, y_train, y_test = split(df)
    vector = Vectorize()

    if st.sidebar.checkbox("show train/test raw data", False):
        st.subheader("Top 25 Headline News from Reddit")
        st.write(df)

    classifier = "Random Forest Classifier"

    


    if classifier == "Random Forest Classifier":
        st.sidebar.subheader("Random Forest Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "n_estimators", 50, 300, step=50, key='n_estimators')
        st.sidebar.subheader("Prediction")	
        userInputLine = st.sidebar.text_area("Enter Your News Headline for Prediction:", "Despite 1,300 dead, majority of Palestinians still back Hamas rocket attacks")
        metrics = ["Confusion Matrix", "Classification_Report", "Accuracy_Score"]
        userInputLine = "b "+ userInputLine
        userInputLine = userInputCheck(userInputLine)
        #st.write(f'{userInputLine.head(1)}')


        if st.sidebar.button("Predict", key="Predict"):

            st.header("Results")
            x_train = vector.fit_transform(x_train)
            x_test = vector.transform(x_test)

            transInput = vector.transform(userInputLine)


            model = RandomForestClassifier(
                n_estimators=n_estimators, criterion='entropy')
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)

            userPred = model.predict(transInput)


            matrix = confusion_matrix(y_test, predictions)
            score = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            # st.write("Accuracy_Score: ", score.round(2))
            # st.write("Classification_Report ", report)
            # st.write("Confusion Matrix ", matrix)
            plot_metrics(metrics)
            st.subheader("User Input Headline Prediction:")
            st.write(f'User Input Headline Prediction Result: {userPred[0]}')
            if userPred[0] == 1:
                st.write("The Dow Jones Industrial Average will stay the same or more up today. Buy!")
            else:
                st.write("The Dow Jones Industrial Average will decrease today. Sell!")
            


if __name__ == '__main__':
    main()