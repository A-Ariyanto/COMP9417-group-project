# FORECASTING AIR POLLUTION WITH MACHINE LEARNING

## Group Members
**Abdullah Ariyanto** (z5543164), **Jeremia Kevin Raja Siahaan** (z5493216), **Riasat Ahmed Chowdury** (z5527294), and **Zafran Akhmadery Arif** (z5603811)

## 📌 Project Overview

This project develops a machine learning system to forecast urban air pollution levels one hour into the future (t + 1) using sensor data from the UCI Machine Learning Repository. The repository contains 9,358 hourly measurements collected at road level in an Italian city (March 2004–February 2005). The dataset includes CO, NMHC, NOx, NO<sub>2</sub>, benzene, and meteorological variables such as temperature and humidity. Missing readings (originally encoded as –200) are treated as null values and imputed during preprocessing.

### Regression Task
The system predicts concentrations of **Carbon Monoxide (CO)**, **Nitrogen Oxides (NO<sub>x</sub>, NO<sub>2</sub>)**, and **Benzene (C<sub>6</sub>H<sub>6</sub>)**.

The primary objective is to compare the predictive performance of traditional statistical baselines (**Linear Regression**) against modern ensemble learning techniques (**Random Forest** and **XGBoost**) to determine the most effective approach for short-term environmental forecasting.

### Classification Task
Additionally, the system predicts the concentration class of **Carbon Monoxide (CO)**, divided into three categories:
* **Low:** CO < 1.5 mg/m³
* **Mid:** 1.5 mg/m³ ≤ CO < 2.5 mg/m³
* **High:** CO ≥ 2.5 mg/m³

The objective is to compare the performance of a baseline classifier (**Decision Tree**) against **Support Vector Machine (SVM)** and **Logistic Regression** to determine the best approach for classifying short-term CO levels.

### Anomaly Detection
For anomaly detection, the project utilizes a residual-based method that identifies unusual deviations between the model’s predictions and the observed pollutant values. By analyzing the distribution of absolute residuals and selecting the 99th percentile as a threshold, we established a clear boundary between normal variation and abnormal behavior. Any data point exceeding this threshold is flagged as an anomaly, allowing us to highlight potential sensor faults or unexpected pollution spikes.
