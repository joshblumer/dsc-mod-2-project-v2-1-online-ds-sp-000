
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

#### Importing and Reviewing the KC dataset

The first step in any data analysis is orienting yourself to an overview of the data you'll be working with. This summary will show us the number of observations in the dataset, as well as the number in each column, and the type of data used in each column. Our data has two columns of the 'object' type that we will have to convert in order to be processed visually and modeled. 

![info](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Screen%20Shot%202021-02-24%20at%209.19.19%20PM.png)

#### Data Preprocessing

After converting data types using string methods we will check to see if there are any null values in our dataset.

![null](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/KC%20Null.png)

Due to the number of null values in our data and what they represented I will impute them with the mode value for each column that contained null values. 

#### Filtering Values

![model](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Model%20Ex..png)

A normally distributed target (dependent) variable is not required for the model to function properly, but the more normal the distribution is the more acceptable the normality and constant variance of the error terms will be during model validation. Our target distribution is highly skewed so normalizing it will be beneficial. The most common way to normalize a distribution in machine learning applications is to apply a logarithmic transformation, but that will also change the scale and magnitude of the intercept and coefficients making them much harder to interpret. Since the strength of linear regression relies on its interpretability we will avoid any processes that make the model diffifcult to interpret. 

![price](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/KC%20Price.png)

One way we can attempt to normalize our target distribution without changing the scale is to filter the range of values manually. By placing an upper limit on 'price' we can remove several outliers contributing to the level of skewness in the distribution. 

![pricefiltered](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/KC%20Price%20Filtered.png)

The result is not a perfectly normal distribution but it is an improvement that will allow the model to retain intuitive interpretation. 

#### Correlation Among Variables

We will use a correlation heatmap to begin to assess what features have a strong relationship with our target 'price' and also what features have high correlations with each other. 

![heatmap](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Heatmap.png)

The feature with the strongest relationship with price is sqft_living, but sqft_living also has high correlation with several other features. We will remove sqft_living from consideration due to its high correlation with many other variables that share proportional increases and decreases with square footage of a home such as number of bedrooms and bathrooms. The second highest correlation with price is the grade feature so we will continue to examine if it is a suitable candidate to build our model around. 

![gradebox](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Grade%20BoxPlot.png)

One of the regression assumptions that cannot be violated if we want to have a successful model is the target and feature variables having a linear relationship. We can check linearity assumptions by plotting features against the target. Histograms work well with continuous data but since grade is an ordinal (following a natural order) categorical  variable we will visualize it using a boxplot. There is clearly a linear relationship between grade and price, but we are not provided with much background information on what grade represents so we will research its significance. 

![gradeex](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/KC%20Grade%20Ex..png)

A quick search through King County's website returned a summary of what the different levels of grade represent. Now that the meaning and range of grade is no longer ambiguous and the linearity assumption has been satisfied, we will move forward with model construction. 

![pricecorr](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Price%20Corr.png)

With features relating to square footage being removed and grade being chosen as our strongest feature we will take a closer look at the remaining features to analyze their relationship with price and get an idea of how they could be engineered and optimized to enhance our model. Our second strongest remaining feature after grade is latitude, but latitude and longitude are not currently represented in an intuitive format. Location is very important to buyers and sellers in real estate markets so we will focus on a solution to incorporate location into our model using the data we've been given.  


![baseline](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Baseline%20OLS.png)

Our data has been preprocessed and we are ready to move on to engineering features and tuning our model. Before we begin that process we will run a baseline regression that can be used as a performance metric going forward. 

<a name="Features"></a>
### Feature Engineering

#### Latitude, Longitude, and Zipcode

Location is often one of if not the most important factors taken into consideration when buying a home, and is also a known contributor of how the value of a property is assessed when selling a home. The location features we are given in the dataset are latitude, longitude, and zip code. There is useful information in those features but not in the format given because the model interprets them as a continuous range of values. In order to incorporate those features into the model, engineering them into an intuitive and interpretible format was required. 

If you examine the map below you can see there are several areas with common groupings of prices. The city areas of 'Seattle' and 'Bellevue' as well as the areas along waterfronts appear to contain a much higher proportion of higher priced homes. 

![pricemap](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/KC%20Lat%3ALong%20Map.png)

My first attempt at feature engineering was using the Haversine library to pinoint a houses location by combining their latitude and longitude, and then with that information I calculated their distance to the cities 'Seattle' and 'Bellevue' due to the grouping of home prices around that area. My hypothesis was that homes close to those cities would be priced higher and prices would decrease as distance from those cities increased. This new feature had a moderate correlation with price, but became problematic during the modeling phase due to high VIF with the engineered zipcode variable so it was not retained.

![zipbar](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Screen%20Shot%202021-02-27%20at%203.11.17%20PM.png)

My second attempt with feature engineering was with zipcode. By plotting zipcodes against their average prices you can determine that there is a linear relationship with price but zipcode is a category so it will need to be transformed in order for the model to be able to process it. One way you can transform categorical variables to be used in regression modeling is to "one-hot encode" them. This process takes each individual value or 'category', turns it into its own feature, and binarizes it so the model interprets its presence as a 1 or 0 instead of the given value. One-hot encoding the zipcode feature was very effective and increased the r-squared, but adding the 70 different zipcodes each as their own independent variable also complicates the model and makes it much harder to interpret. 

![sdmap](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/KC%20School%20Districts.gif)

One way to reduce the complexity of one hot encoding each individual zip code was to group the zip codes together into bins based on a given value range and then one hot encode the bins. I wanted the groupings of zip codes to be naturally occurring and geographically significant and based on the clustering of home prices I assumed that to be possible if I could find the right mechanism. After reflecting on the challenge, the question of how King Counties school district borders were determined came to mind so I googled my question and found the following map. After visually comparing the school district and price by location maps it was very apparent that there was a relationship between the two.

![sdbox](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SD%20PreBinned.png)

I manually created a dictionary of school districts that contained their corresponding zip codes and used a box plot to assess if there was a linear relationship between school districts and price which there was. By binning zip codes into school districts I was able to reduce zip code from 70 entries down to 18, but there were multiple school districts with similarly distributed price ranges so I assessed the map again and grouped those similar school districts together for a final number of 9 school district zones.

![sdbinned](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SD%20Binned.png)

Reducing the number of zip codes nearly ten fold from 70 to 9 reduced the r-squared but greatly improved the ability to use the model intuitively and interpret the results.  

<a name="Models"></a>
### Modeling 

I took an iterative approach to modeling that included experimenting with bottom up ( adding one feature at a time to the model until multicollinearity was introduced), top down (starting with all available features and removing one at a time until there was no multicollinearity present), and models built around different ‘anchor’ features (sqft_living and grade) that had high correlation with the target variable but couldn’t be included in a model together due to high correlation among each other. 

#### Bottom Up

![simple](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Simple%20OLS.png)

I began the bottom up approach with a simple regression on the strongest retained feature 'grade' which is shown above and added features iteratively checking the VIF with each addition. This process seemed to be a less efficient workflow due to having to add a new cell to rerun the regression during each iteration.   

#### Top Down Grade Centric Model with Individual School Districts

![topdownstart](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Top%20Down%20Start%20OLS.png)

My next approach was top down feature removal with the original school district list before being combined into zones with similar price ranges. The r-squared began at 0.696

![topdownend](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Top%20Down%20End%20OLS.png)

Once I removed the features with a p-value over 0.05 and VIF over 30 I was left with an r-squared of 0.649

![topdownvif](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Top%20Down%20End%20VIF.png)

Due to the original school district features having multiple columns with similar price mean and range, this model had multiple features with VIF over 10. This model met validation assumptions but crossed the VIF threshold that I was comfortable with. 

#### Sqft_Living Centric Model

![sqftstart](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SQFT%20Start%20OLS.png)

Another iteration of model experimentation I attemped was using sqft_living as the primary feature to measure VIF against instead of grade. It began with an r-squared of 0.690

![sqftend](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SQFT%20End%20OLS.png)

Throughout the process of removing features with high p-values and VIF the r-squared of this model approach diminished to 0.411

![sdstart](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SD%20Start%20OLS.png)

#### Top Down Grade Centric Model with Zoned School Districts 

![sdend](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SD%20End%20OLS.png)

This is my final model which was built using a grade centric approach and zoned school districts. By grouping the school districts into zones with similar price means and ranges the initial r-squared increased from 0.696 to 0.714

![finalp](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Final%20Model%20P-Values.png)

After removing the features with high p-values and VIF I was left with an r-squared of 0.665, an increase from 0.649

![finalvif](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Final%20VIF.png)

The final model has the lowest measures of skew, kurtosis, and condition number. All of the p-values are less than 0.05

![resid](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/Residuals.png)

The r-squared improvement in the final model is marginal but by grouping all school districts into zones the VIF for each feature was lowered to less than 5 which is a great improvement. A high level of VIF indicates that the model can't distinguish which correlated features are contributing predictive power to the the models results so lowering the VIF makes the model much more reliable. 

## Summary

### Insights

![sdsum](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SD%20Summary.png)
![sddpsq](https://raw.githubusercontent.com/joshblumer/dsc-mod-2-project-v2-1-online-ds-sp-000/master/Photos/SD%20dpsf.png)

### Conclusion

Initial data exploration and baseline modeling revealed high correlation and variance inflation factors among several of the features (independent variables). Thus achieving a high r-squared and low MSE (mean squared error) while avoiding those high correlations and meeting regression assumptions was challenging. I first experimented with separate models based on the two features that had the highest correlation with price, "grade" and "sqft_living" and found that "grade" was the best candidate due to lower levels of correlation among a greater number of features. "Grade" and "sqft_living" were not compatible for use in a model together due to high correlation. My approach then consisted of experimenting with top-down (starting with all predictor variables and removing features with high correlation or vif one at a time) and bottom-up (starting with the feature that had the highest correlation with price and adding features one at a time until correlation among features was introduced) model constructions and I found top-down to be more efficient in this particular use case so that approach was retained. My final model uses the features grade, view, waterfront, home age,  and school districts binned into zones as price predictors and has an r-squared of 0.67 and MSE of $113,835. The coefficient for grade is $119,500, so for every increase in one unit of grade on the scale of 3-13, with a mean of 7, there is a corresponding price increase of $119,500. The coefficient for view is $38,870, view is a measure of how many times the home was viewed before selling on a scale of 0-4 with a mean of 0.23. There was not reliable information about what each interval of view represented but it is assumed each interval is a range of numbers. The coefficient for waterfront is $130,400 so a property with waterfront access sells for an average of $130,400 more than the same home without waterfront access. 

