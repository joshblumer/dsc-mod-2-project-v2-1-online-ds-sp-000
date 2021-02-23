
# Module 2 Final Project

King County House Sales Analysis 


## Introduction

An exploratory data analysis of the King County Housing database provided by The Flatiron School and found at [kaggle.com](https://www.kaggle.com/harlfoxem/housesalesprediction) My goal in this analysis is to extract meaningful insights from the data and construct them into actionable recommendations using statistical methods, visualizations, and a supervised machine learning algorithm, linear regression. 

## Technologies
* The Jupyter Notebook "student.ipynb" found in this repo contains the data analysis and visualizations used in this project and was created using Python 3.7.6
* There is a powerpoint presentation for non-technical audiences available under the file name "presentation.pdf"

### Necessary libraries to reproduce and run this project are:

* Pandas 
* NumPy
* MatPlotLib
* Seaborn
* SciPy
* DateTime
* Haversine
* SKLearn
* StatsModels

## Objectives

* Explore and analyze dataset using visualizations to identify outliers, distributions, and linear relationships
* Clean data by removing or imputing missing and null values
* Analyze independent variables for the strongest possible correlation with the dependent variable 'price' while avoiding high     correlations between independent variables 
* Engineer features to maximize model performance and interpretability
* Model dependent/independent variable relationships using simple and multiple linear regression
* Validate model performance and ability to infer price given feature coeffecients and verify no regression assumptions are       violated

## Methodology

The strength of linear regression as a supervised machine learning model is interpretability so my primary goal while constructing this model is to achieve as high of an r-squared and as low of a MSE (mean squared error) as possible while maintaining model interpretability. Particular attention has to be paid to the regression assumptions of linearity and multicollinearity among predictor variables and normality, homoscedasticity, and independence of the error terms in order for the feature coefficients to be reliable. 

## Table of Contents

* [Exploratory Data Analysis](#EDA)
* [Feature Engineering](#Features)
* [Modeling](#Models)

<a name="EDA"></a>
### Exploratory Data Analysis

#### Importing the KC dataset
![kcinfo](http://localhost:8888/view/Pics/KC%20Info.png)


![kcinfo](http://localhost:8888/view/Pics/KCInfo.png)





<a name="Features"></a>
### Feature Engineering







<a name="Models"></a>
### Modeling 












## Conclusion

Initial data exploration and baseline modeling revealed high correlation and variance inflation factors among several of the features (independent variables). Thus achieving a high r-squared and low MSE (mean squared error) while avoiding those high correlations and meeting regression assumptions was challenging. I first experimented with separate models based on the two features that had the highest correlation with price, "grade" and "sqft_living" and found that "grade" was the best candidate due to lower levels of correlation among a greater number of features. "Grade" and "sqft_living" were not compatible for use in a model together due to high correlation. My approach then consisted of experimenting with top-down (starting with all predictor variables and removing features with high correlation or vif one at a time) and bottom-up (starting with the feature that had the highest correlation with price and adding features one at a time until correlation among features was introduced) model constructions and I found top-down to be more efficient in this particular use case so that approach was retained. My final model uses the features grade, view, waterfront, home age,  and school districts binned into zones as price predictors and has an r-squared of 0.67 and MSE of $113,835. The coefficient for grade is $119,500, so for every increase in one unit of grade on the scale of 3-13, with a mean of 7, there is a corresponding price increase of $119,500. The coefficient for view is $38,870, view is a measure of how many times the home was viewed before selling on a scale of 0-4 with a mean of 0.23. The coefficient for waterfront is $130,400 so a property with waterfront access sells for an average of $130,400 more than the same home without waterfront access. 






#### Visualizations & EDA

* Your project contains at least 4 meaningful data visualizations, with corresponding interpretations. All visualizations are well labeled with axes labels, a title, and a legend (when appropriate)  
* You pose at least 3 meaningful questions and answer them through EDA.  These questions should be well labeled and easy to identify inside the notebook.
    * **Level Up**: Each question is clearly answered with a visualization that makes the answer easy to understand.   
* Your notebook should contain 1 - 2 paragraphs briefly explaining your approach to this project.

#### Model Quality/Approach

* Your model should not include any predictors with p-values greater than .05.  
* Your notebook shows an iterative approach to modeling, and details the parameters and results of the model at each iteration.  
    * **Level Up**: Whenever necessary, you briefly explain the changes made from one iteration to the next, and why you made these choices.  
* You provide at least 1 paragraph explaining your final model.   
* You pick at least 3 coefficients from your final model and explain their impact on the price of a house in this dataset.   