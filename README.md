
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



The first step in any data analysis is orienting yourself to an overview of the data you'll be working with. This summary will show us the number of observations in the dataset, as well as the number in each column, and the type of data used in each column. Our data has two columns of the 'object' type that we will have to convert in order to be processed visually and modeled. 

After converting data types we will check to see if there are any null values in our dataset.
![null](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/KC%20Null.png)


![price](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/KC%20Price.png)


![pricefiltered](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/KC%20Price%20Filtered.png)


![model](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Model%20Ex..png)


![heatmap](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Heatmap.png)


![pricecorr](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Price%20Corr.png)


![gradeex](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/KC%20Grade%20Ex..png)


![gradebox](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Grade%20BoxPlot.png)


![baseline](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Baseline%20OLS.png)



<a name="Features"></a>
### Feature Engineering

![pricemap](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/KC%20Lat%3ALong%20Map.png)


![zipbar](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Zips.png)


![sdmap](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/KC%20School%20Districts.gif)


![sdbox](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SD%20PreBinned.png)


![sdbinned](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SD%20Binned.png)


<a name="Models"></a>
### Modeling 

![simple](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Simple%20OLS.png)


![topdownstart](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Top%20Down%20Start%20OLS.png)


![topdownend](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Top%20Down%20End%20OLS.png)


![topdownvif](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Top%20Down%20End%20VIF.png)


![sqftstart](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SQFT%20Start%20OLS.png)


![sqftend](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SQFT%20End%20OLS.png)


![sdstart](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SD%20Start%20OLS.png)


![sdend](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SD%20End%20OLS.png)


![finalp](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Final%20Model%20P-Values.png)


![finalvif](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Final%20VIF.png)


![resid](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Residuals.png)



## Conclusion

![sdsum](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SD%20Summary.png)
![sddpsq](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SD%20dpsf.png)

Initial data exploration and baseline modeling revealed high correlation and variance inflation factors among several of the features (independent variables). Thus achieving a high r-squared and low MSE (mean squared error) while avoiding those high correlations and meeting regression assumptions was challenging. I first experimented with separate models based on the two features that had the highest correlation with price, "grade" and "sqft_living" and found that "grade" was the best candidate due to lower levels of correlation among a greater number of features. "Grade" and "sqft_living" were not compatible for use in a model together due to high correlation. My approach then consisted of experimenting with top-down (starting with all predictor variables and removing features with high correlation or vif one at a time) and bottom-up (starting with the feature that had the highest correlation with price and adding features one at a time until correlation among features was introduced) model constructions and I found top-down to be more efficient in this particular use case so that approach was retained. My final model uses the features grade, view, waterfront, home age,  and school districts binned into zones as price predictors and has an r-squared of 0.67 and MSE of $113,835. The coefficient for grade is $119,500, so for every increase in one unit of grade on the scale of 3-13, with a mean of 7, there is a corresponding price increase of $119,500. The coefficient for view is $38,870, view is a measure of how many times the home was viewed before selling on a scale of 0-4 with a mean of 0.23. The coefficient for waterfront is $130,400 so a property with waterfront access sells for an average of $130,400 more than the same home without waterfront access. 

