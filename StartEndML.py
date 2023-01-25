import streamlit as st
import plotly.express as px
import pycaret
import pandas_profiling
import pandas as pd
import numpy as np
from streamlit_pandas_profiling import st_profile_report
import os
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#regression models
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

#For Login Page
import streamlit_authenticator as stauth
import yaml
from yaml import SafeLoader

import database as db

users = db.fetch_all_users()
usernames = [user['key'] for user in users]
names = [user["name"] for user in users]
hashed_passwords = [user['password'] for user in users]

authenticator = stauth.Authenticate(names, usernames, hashed_passwords, 'startendml', 'abcdef', cookie_expiry_days=30)

def login_page():
    flag = 0
    name, authentication_status, username = authenticator.login('Login', 'main')

    if st.session_state["authentication_status"]:
        authenticator.logout('Logout', 'main')
        # st.write(f'Welcome *{st.session_state["name"]}*')
        # st.title('Some content')
        flag = 1
    elif st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')

    return name, authentication_status, username, flag


name, authentication_status, username, flag = login_page()

if flag == 0:
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Register'):
            try:
                if authenticator.register_user('Register user', preauthorization=False):
                    db.insert_user(st.session_state["username"], st.session_state["name"], st.session_state["hashed_password"])
                    st.success('User registered successfully')
                    login_page()
            except Exception as e:
                st.error(e)

    with col2:
        if st.button('Forgot Password'):
            try:
                username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
                if username_forgot_pw:
                    db.update_user(username_forgot_pw, {"password":random_password})
                    st.success('New password sent securely')
                    # Random password to be transferred to user securely
                elif username_forgot_pw == False:
                    st.error('Username not found')
            except Exception as e:
                st.error(e)


if authentication_status:
    with st.sidebar:
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("Start to End Machine Learning")
        choice = st.radio("Navigation", ["Home", "Upload", "EDA", 'Data Cleaning', "Modelling", "Model Interpretation", "Download", "Account"])
        st.info("This website helps you build and explore your data.")
    if choice == "Home":
        st.write("This is home!")
    if choice == "Upload":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file: 
            df = pd.read_csv(file, index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

    if choice == "EDA":
        try:
            st.title("Exploratory Data Analysis")
            st.write("This may take some time to load")
            df = pd.read_csv('dataset.csv', index_col=None)
            profile_df = df.profile_report()
            st_profile_report(profile_df)
            #Clean the data, use one hot encoding for categorical variables, and scale the data

        except Exception as e:
            st.write(e)
            st.error("Please Upload Your Dataset")

    if choice == 'Data Cleaning':
        # calculate the threshold for non-null values
        df = pd.read_csv('dataset.csv', index_col=None)
        threshold_cols = int(df.shape[0] * 0.5)

        # drop columns with less than threshold non-null values
        df.dropna(axis=1, thresh=threshold_cols, inplace=True)

        for col in df.columns:
            if df[col].dtype not in ['int64', 'float64']:
                # perform one-hot encoding on the column
                df = pd.concat([df,pd.get_dummies(df[col], prefix=col)], axis=1)
                # drop the original column
                df.drop(col, axis=1, inplace=True)
        df.interpolate(method='linear',inplace=True)
        st.dataframe(df)
        df.to_csv('dataset.csv', index=None)

    if choice == "Modelling":
        # try:
        df = pd.read_csv('dataset.csv', index_col=None)
        classoreg = st.radio("Choose the type of problem", ["Regression", "Classification"])
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        X = df.drop(chosen_target, axis=1)
        y = df[chosen_target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        if classoreg == "Regression":
            if st.button('Run Modelling'):
                #REGRESSION MODELS
                models = [('Multiple Linear Regression', LinearRegression()),
                        ('Polynomial Regression', PolynomialFeatures()),
                        ('Robust Regression - RANSAC', RANSACRegressor()),
                        ('Decision Tree', DecisionTreeRegressor()),
                        ('Random Forest', RandomForestRegressor()),
                        ('Gaussian Process Regression', GaussianProcessRegressor()),
                        ('Support Vector Regression', SVR())]

                # Define the hyperparameters for each model
                param_grid = {'Multiple Linear Regression': {},
                            'Polynomial Regression': {'degree': [2, 3, 4]},
                            'Robust Regression - RANSAC': {'max_trials': [100, 200, 500], 'min_samples': [10, 20, 30]},
                            'Decision Tree': {'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10]},
                            'Random Forest': {'n_estimators': [10, 50, 100, 200], 'max_depth': [5, 10, 20, None]},
                            'Gaussian Process Regression': {'kernel': [None, 'RBF']},
                            'Support Vector Regression': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
                            }
                best_model_obj = None
                best_score = -np.inf
                for model_name, model in models:
                    grid = RandomizedSearchCV(estimator=model, param_grid=param_grid[model_name], cv=5, n_jobs=3)
                    grid.fit(X, y)
                    if grid.best_score_ > best_score:
                        best_score = grid.best_score_
                        best_model_obj = grid.best_estimator_
                        pickle.dump(model, open('best_model.pkl', 'wb'))
                print(f'The best model is {best_model_obj} with a score of {best_score} and best parameters {grid.best_params_}')

                best_model_obj.fit(X_train, y_train)
                test_score = best_model_obj.score(X_test, y_test)
                print(f'The test score for best model is {test_score}')
        else:
            if st.button('Run Modelling'):
                #CLASSIFICATION MODELS
                # Define the models to be compared
                st.write("Inside Classification")
                models = [('Logistic Regression', LogisticRegression()),
                        ('Random Forest', RandomForestClassifier()),
                        ('K-Nearest Neighbors', KNeighborsClassifier()),
                        ('Support Vector Machine', SVC()),
                        ('Gaussian Naive Bayes', GaussianNB()),
                        ('XGBoost', XGBClassifier())]
                # st.write("Models defined")

                # Define the hyperparameters for each model
                param_grid = {'Logistic Regression': {'C': np.logspace(-3,3,7), 'penalty': ['l1', 'l2'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']},
                            'Random Forest': {'n_estimators': [10, 50, 100, 200, 500], 'max_depth': [5, 10, 20, 30, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
                            'K-Nearest Neighbors': {'n_neighbors': [1, 3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                            'Support Vector Machine': {'C': np.logspace(-3,3,7), 'kernel': ['linear', 'rbf'], 'degree': [2,3,4], 'gamma': ['scale','auto']},
                            'Gaussian Naive Bayes': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]},
                            'XGBoost': {'learning_rate': [0.1, 0.01, 0.001], 'max_depth': [3, 5, 7, 10], 'n_estimators': [100, 200, 500], 'booster': ['gbtree','dart'], 'subsample':[0.8,0.9,1],'colsample_bytree':[0.8,0.9,1]}
                            }

                # st.write("Params defined")
                best_model_obj = None
                best_score = -np.inf
                for model_name, model in models:
                    # st.write(f"Inside the loop {model}")
                    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid[model_name], cv=5, n_jobs=3)
                    grid.fit(X, y)
                    st.write(f'The model {model_name} gave a score of {grid.best_score_} with parameters {grid.best_params_}')
                    if grid.best_score_ > best_score:
                        best_score = grid.best_score_
                        best_model_obj = grid.best_estimator_
                        pickle.dump(model, open('best_model.pkl', 'wb'))
                st.write(f'The best model is {best_model_obj} with a score of {best_score} and best parameters {grid.best_params_}')

                best_model_obj.fit(X_train, y_train)
                test_score = best_model_obj.score(X_test, y_test)
                st.write(f'The test score for best model is {test_score}')



    if choice == 'Model Interpretation':
        st.write("Inside Model Interpretation")
        pass

    if choice == "Download":
        try:
            with open('best_model.pkl', 'rb') as f:
                st.download_button('Download Model', f, file_name="best_model.pkl")
        except:
            st.error("Please complete the Modelling section first")

    if choice == "Account":
        if authentication_status:
            try:
                if authenticator.reset_password(username, 'Reset password'):
                    st.success('Password modified successfully')
            except Exception as e:
                st.error(e)

        if authentication_status:
            try:
                if authenticator.update_user_details(username, 'Update user details'):
                    st.success('Entries updated successfully')
            except Exception as e:
                st.error(e)

        with open('config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
