
# Boston Housing Price Prediction

Overview

This project is a Boston Housing Price Prediction application built using Streamlit and Ridge Regression. It predicts the median value of owner-occupied homes based on 9 key features of the dataset.
## Features

- User-Friendly Input: Uses sliders for user input of housing features.
- Machine Learning Model: Implements Ridge Regression with Polynomial Features.

- Feature Normalization: Uses StandardScaler for better model performance.

- Performance Metrics: Displays Mean Squared Error (MSE) and R-squared Score (R2).

- Interactive Web App: Built using Streamlit.


## DATASET


The model is trained on the Boston Housing Dataset.
 The target variable (medv) represents the median value of homes in $1000s.

The following 9 features are selected:

- Crime Rate

- Industrial Zone

- Non-Retail Acres

- Charles River Proximity

- Nitrogen Oxides Concentration

- Average Rooms per Dwelling

- Age of Property

- Distance to Employment Centers

- Accessibility to Highways
## 📥 Installation

Prerequisites

Ensure you have Python 3.12.3 installed.
```bash
  pip install pandas numpy matplotlib seaborn streamlit scikit-learn
```
How to Run the App

```bash
 streamlit run app.py
```
After running the command, open the provided URL in a browser to access the web app.

Project Structure

```bash
 |-- BostonHousing.csv   # Dataset file
|-- app.py             # Streamlit application
|-- README.md          # Project documentation
```

## Model Training Steps

- Load Dataset: Reads BostonHousing.csv.

- Feature Selection: Selects 9 important features.

- Data Normalization: Uses StandardScaler to scale features.

- Polynomial Transformation: Enhances feature interactions.

- Train-Test Split: Splits data into 80% training and 20% testing.

- Train Model: Fits Ridge Regression on training data.

- Make Predictions: Uses user input to predict home prices.

- Model Performance: Displays MSE and R-squared scores
## Future Enhancements

- Add more advanced models like Lasso Regression or Gradient Boosting.

- Deploy the app using Streamlit Cloud or Heroku.

- Improve UI with custom styling and graphs.

## Author

Developed by Spoorthi Ramu Biradar
## License

This project is open-source and available for personal and educational use.

