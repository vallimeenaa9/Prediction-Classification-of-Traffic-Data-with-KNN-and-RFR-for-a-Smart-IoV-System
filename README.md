# Prediction-Classification-of-Traffic-Data-with-KNN-and-RFR-for-a-Smart-IoV-System

Instituted an algorithm to predict and classify traffic conditions (weather, directions, traffic speed &amp; congestions) via existing traffic data collected throughout the years in Chicago. Exploited KNN and RFR algorithms to fulfil the task. 

Accepted for publication by 4th SMART CITIES SYMPOSIUM, University of Bahrain 

## Introduction

With the massive surge in population, the number of vehicles has also risen exponentially. In recent years, development of Internet of Things (IoT) has led to an increase in connected vehicles or vehicles connected to the internet. This posed an ideal opening to design an Internet of Vehicles (IoV) system that could predict, classify and plot the shortest path between the user’s location and destination. This paper explores machine learning models, K-nearest neighbours (KNN) and Random Forest Regressor (RFR) to predict traffic in a unique way for an Internet of Vehicles ecosystem. The high accuracy of the proposed model and the ability to factor in more features like weather, gives our model great flexibility. Further, the integration of two powerful machine learning models enablesthis method of prediction to be faster, but at the same time more reliable than the existing models. All processes which involve data storage, prediction and computation takes place in dedicated cloud servers. The predicted data is then utilized to formulate the fastest route between all pairs of locations, factoring in the traffic conditions predicted above. This enables the formation of an IoV ecosystem.

## System Architecture

![image](https://user-images.githubusercontent.com/47136906/141652260-1290af00-e530-4f5c-bfdd-9ed8752f16d0.png)

## Readme

1. Pip3 install the necessary libraries required by the algorithm.
2. Make sure that the .csv file and the algorithm is present in the same directory.
3. Run the algorithm to visualize prediction accuracy and classification confusion matrix.
4. On the latest version of Scikitlearn, use the Prediction+Classification_ColumnTransformer file, as the initially used OneHotEncoder has been deprecated in the recent versions.
