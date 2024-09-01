# Data-Science-Projects
collection of solved ML problems

Projects
1. Churn Prediction
- this is a classical classification problem where the intent is to predict customer churn based on shared features via a public dataset by IBM on Telco
- the project is divided into 3 phases : a) Data Analysis b) Training the data on a baseline model c) Comparing different recall values as performance metric for different classification models to choose the best one and Deploy the model
- w.r.t data analysis following checks were made : a) summary statistics b) misiing value c) imbalance case d) outliers e) feature distribution f) correlation between features
- Based on different correlation values and a threshlod of 0.3, 6 features were used for training a Logistic Regression model. This will act as our baseline model
- To evaluate the model performance, Recall value is considered and then it's value is compared across other different models like XGBoost, LIghtGBM, SVC and others
- In this primary analysis, Logistic Regression outperforms out models based on the test recall score.
- Finally the model is saved using joblib and served via FastAPI
- Model Usage:
  create a virtual environment
  clone the repo
  install the dependencies
  type the uvicorn command to do the api call
  query the model with correct feature input values
