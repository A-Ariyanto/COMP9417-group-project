# FORECASTING AIR POLLUTION WITH MACHINE LEARNING

## Group Members:
Abdullah Ariyanto (z5543164), Jeremia Kevin Raja Siahaan (z5493216), Riasat Ahmed Chowdury (z5527294), and Zafran Akhmadery Arif (z5603811)

## 📌 Project Overview

This project develops a machine learning system to forecast urban air pollution levels one hour into the future \(t + 1\) using sensor data from the UCI Machine Learning Repository. The repository contains 9,358 hourly measurements collected at road level in an Italian city (March 2004–February 2005). The dataset includes CO, NMHC, NOx, NO₂, benzene, and meteorological variables such as temperature and humidity. Missing readings (encoded as –200) are treated as missing values and imputed during preprocessing.

The system predicts concentrations of **Carbon Monoxide (\(CO\))**, **Nitrogen Oxides (\(NO_x, NO_2\))**, and **Benzene (\(C_6H_6\))**.  
The primary objective is to compare the predictive performance of traditional statistical baselines (**Linear Regression**) against modern ensemble learning techniques (**Random Forest** and **XGBoost**) to determine the most effective approach for short-term environmental forecasting.

Besides that, the system also predicts the classes of **Carbon Monoxide (\(CO\))** concentration, which we can divide it into 3 classes: low (CO < 1.5 mg/m^3), mid (1.5 mg/m^3 <= CO < 2.5 mg/m^3), and high (CO >= 2.5 mg/m^3).
The objective is to compare the performance of classification baseline (**Decision Tree**) against **Support Vector Machine** and **Logistic Regression** to determine the approach for short-term CO concentration class.

For anomaly detection, we applied a residual-based method that identifies unusual deviations between the model’s predictions and the observed pollutant values. By analyzing the distribution of absolute residuals and selecting the 99th percentile as a threshold, we established a clear boundary between normal variation and abnormal behavior. Any data point exceeding this threshold was flagged as an anomaly, allowing us to highlight potential sensor faults or unexpected pollution spikes.