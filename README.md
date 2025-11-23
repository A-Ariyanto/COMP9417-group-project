# Air Quality Prediction using Machine Learning

## 📌 Project Overview

This project develops a machine learning system to forecast urban air pollution levels one hour into the future \(t + 1\) using sensor data from the UCI Machine Learning Repository.

The system predicts concentrations of **Carbon Monoxide (\(CO\))**, **Nitrogen Oxides (\(NO_x, NO_2\))**, and **Benzene (\(C_6H_6\))**.  
The primary objective is to compare the predictive performance of traditional statistical baselines (**Linear Regression**) against modern ensemble learning techniques (**Random Forest** and **XGBoost**) to determine the most effective approach for short-term environmental forecasting.

Besides that, the system also predicts the classes of **Carbon Monoxide (\(CO\))** concentration, which we can divide it into 3 classes: low (CO < 1.5 mg/m^3), mid (1.5 mg/m^3 <= CO < 2.5 mg/m^3), and high (CO >= 2.5 mg/m^3).
The objective is to compare the performance of classification baseline (**Decision Tree**) against **Support Vector Machine** and **Logistic Regression** to determine the approach for short-term CO concentration class.
