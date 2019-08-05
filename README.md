# Thinkful--Projects

## [6.6.5-Keras for Neural Net-CNN.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/6.6.5-Keras%20for%20Neural%20Net-CNN.ipynb)
Implementation of Neural net CNN on keras built on Tensorflow. Used several configurations of CNN to improve accuracy of the model in classification of cifar-10 data set. Cifar-10 dataset is a collection of animals and vehicles as groups of 10 classes.
Here aim is to classify the images to these 10 classes after training on the labelled dataset.

Implementation 1: Multi Layer Perceptron

Implementation 2: Convolutional Neural Networks

Implementation 3: Convolutional Neural Networks with additional Convl 2D layer and pooling layer

Implementation 4: Convolutional Neural Networks with different number of neurons and parameters

Implementation 5: Convolutional Neural Networks with Batch Normalization

Implementation 6: Convolutional Neural Network with Data Augmentation and additional parameters

**Conclusions:** The best implementation is 6 even the computational time has increased, since accuracy is around 88% it is worth the tradeoff.

## [Challenge 4.4.5-Build your own NLP model-vf.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Challenge%204.4.5-Build%20your%20own%20NLP%20model-vf.ipynb)

**Challenge: Is to Build an NLP model to classify reviews or texts from Amazon as positive or negative.**

Data cleaning / processing / language parsing

Create features using two different NLP methods: For example, BOW and TF-IDF.

BOW feature generation

Use the features to fit supervised learning models for each feature set to predict the category outcomes.

TF-IDF

Use the features to fit supervised learning models for each feature set to predict the category outcomes.

With TFIDF feature set : The Logistic regression gave the best AUC of 0.88. 84% test accuracy.

With BOW feature set : The Gradient Boost classifier gave the best AUC of 0.67. 64% test accuracy.

When compared with the BOW features and TF_IDF features, models generated from TF_IDF feature set gave much better results.

Picked one of the models and tried to increase accuracy by at least 5 percentage points.

**Conclusion: 
I have changed the following: and got 5% increase in accuracy from 82% to 87%.**

**The model can be enhanced even more using Neural networks and implement word2vec or sentence2vec or doc2vec embedding to generalize more.**

## [Challenge-4.4.2 Supervised NLP.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Challenge-4.4.2%20Supervised%20NLP.ipynb)
This project involves developing a new NLP model that is accurate in classifying or identifying "Alice in Wonderland" novel vs any other work, Persuasion vs any other work, or Austen vs any other work.  This will involve pulling a new book from the Project Gutenberg corpus (print(gutenberg.fileids()) for a list) and processing it.

**Conclusion: 
When added common pos, common phrases and common entities the accuracy increased from 83% to 87% i.e. a 4% increase.
The model does well with 88% (same as got before) when used to train another text document edge along with alice txt. It has got the same accuracy as alice and persuasion.**

## [Challenge-BC-5.2.4 APIs Movies.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Challenge-BC-5.2.4%20APIs%20Movies.ipynb)
This is web-scraping project. Initiated scrapy crawler to web-scrape data from movie ranking website for gross revenue in year 2018. Using the response query and xpath could successfully scrape the desired revenue details from the website. the data downloaded is used in calculating the percentages and ranking them and visualizing the gross revenue of the movies in 2018 as percentages in pie graph.

**Conclusion: 
The data from webscraping allows us to see what movies got what percentage of gross collections in 2018. From this we can conclude that Black Panther bagged around 1/4th of the total revenue of that year and the other 1/4 was bagged by Avengers: Infinity war
Challenge-BC-5.5.4-What test to use.ipynb** 

## [Challenge-BC-5.5.4-What test to use.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Challenge-BC-5.5.4-What%20test%20to%20use.ipynb)
The dataset here is about ESS dataset @ https://thinkful-ed.github.io/data-201-resources/ESS_practice_data/ESS_codebook.html.
In this dataset, the same participants answered questions in 2012 and again 2014 about happiness, trust and living single or with a partner.

- 1) Did people become less trusting from 2012 to 2014? Compute results for each country in the sample.
First, we need to decide we should go for parametric or non-parametric test. This can be decided based on normality test by using Shapiro-Wilk test statistic, W and visualization by histograms.
**Conclusion: Based on the p values except for countries 'ES' and 'SE' there is no difference in people trust between 2012 and 2014 years. Even those there is border line significance. Now let us see if it has decreased or increased. But here there is no decrease and actually there is an increase (borderline) in the people trust in 'SE' from 2012 to 2014**

- 2) Did people become happier from 2012 to 2014? Compute results for each country in the sample. -- Similar analysis as above was conducted.

- 3) Who reported watching more TV in 2012, men or women?
**Conclusion: No Significant change observed in the TV watching between men and women in 2012.**

- 4) Who was more likely to believe people were fair in 2012, people living with a partner or people living alone?
**Since, the means for people living alone was higher in the people fair they are the ones who believe that people are fair than the other group. Conclusion: People living alone are more likely to believe that people were fair in 2012 than people living with partners**

- 5) Pick three or four of the countries in the sample and compare how often people met socially in 2014. Are there differences, and if so, which countries stand out?
**The pair wise comparisons of the countries we see almost all countries comparisons the null hypothesis is not rejected expect two country "SE" and "CH". Not Rejecting null hypothesis means there is no significant effect between the groups in people who meet socially in 2014.**

**The countries that stands out are the "SE" when compared to 'CH' there is significant difference in people in how often people meet socially in 2014 between these two countries.**

## [Project 3.2.6-Challenge- If a tree falls in the forest....ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Project%203.2.6-Challenge-%20If%20a%20tree%20falls%20in%20the%20forest....ipynb)

This is a small dataset of admission to a college suing GRE and GPA and rank of the students. The challenge of the dataset is to predict the admission yes or no or 1 or 0 binary if the student gets admitted based on the GRE, GPA and rank provided.

There is good relationship (as seen by the slope of the line) between the outcome variable 'admit' and the predictor variables such as GPA, GRE and rank.

**Conclusion: It can be concluded that RFC is much better in terms of reducing errors drastically at least in this dataset. The accuracy is not improved much here in this dataset. The AUC is also improved drastically which is a measure of again the errors.**

## [Project 3.6.1-Credit card Fraud.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Project%203.6.1-Credit%20card%20Fraud.ipynb)
 The dataset is about credit card fraud detection.  Most of the variables  have been anonymized by doing a PCA. This is a classifier problem where we are trying to classify Fraud, or no Fraud based on predictor variables. 
 
 Some algorithms that do well here are 
 
-Naives Bayes Classifier

-kNN Classifier

-Vanilla Logistic Regression

-Random Forest Classifier

-Gradient Boost Classifier

Lot of class imbalance. One way to overcome this problem is by under sampling the majority class and synthetic oversampling the minority class i.e., SMOTE.

Before trying all models, that let us first solve class imbalance problem. This is done using SMOTE.

The heat map shows that there is very less or no correlation between all the PCA components which is expected and shows they are independent variables.

The outcome variable class however shows varied correlation with predicted variables.

**Conclusion: The models which performed better in terms of both the accuracy and error rate especially, lower false negatives rate are Random Forest==Gradient Boost Classifier > Vanilla Logistic Regression > Naive Bayes > KNN Classifier**

## [Project 3.6.2-Airline Arrivals.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Project%203.6.2-Airline%20Arrivals.ipynb)

This is a regression problem where we are trying to predict Flight delay based on various features. Some algorithms that do well here are Random Forest Regression, Gradient Boost regression etc.

Reset the outliers by an optimized process called winsorizing.

**Conclusion: Random Forest regressor and Gradient Boost Regressor appear to be having low on errors with approx. equal accuracies around 99%.**

## [Project 3.6.3-Amazon Reviews.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Project%203.6.3-Amazon%20Reviews.ipynb)

The dataset is about the amazon Instant video reviews which has the review text and a overall rating on a scale of 1-5. Based on the overall rating score the review texts are labelled as positive or negative i.e. above 3 it is positive and 3 and below is negative.

The solution was supposed to be restricted to use only positive words or negative words found in the positive or negative reviews respectively. So this solution looks at only primitive method to classify. 

**Conclusion: Gradient boost classifier shows better accuracy scores than other models. The accuracy scores are low because a very primitive method of classification used. More advanced methods include using TF-IDF, word2vec, sen2vec, doc2vec methods to add features for the classification.**

## [Project 3.6.4-Housing Prices.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Project%203.6.4-Housing%20Prices.ipynb)

The data set is a Melbourne housing dataset with various columns related to house. The goal is to predict the house prices based on these features.

Feature selection methods like SelectKBest was used to predict the most important features or columns. Gradient boosting regression model has a built-in feature importance method which is shown here how to use to predict the most important features to predict the prices of houses.

As suggested by the Gradient boost model top 12 features needed for a real estate developer are shown to add value to a house and also predictive of the prices of house.

**Conclusion: The Gradient Boost model has the highest accuracy of 81% compared to all other models.**


## [Project BC-3.1.4.Challenge Model Comparison.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Project%20BC-3.1.4.Challenge%20Model%20Comparison.ipynb)

The Dataset used here is a housing data set. Different models of regression are compared with only one X variable and multi-variate X i.e. more than one X variable.

OLS-Linear regression (Simple) i.e. only one X variable

KNN regression (Simple) i.e. only one X variable

OLS-Linear regression (Multi-variate) i.e. more than one X variable

KNN regression (Multi) i.e. more than one X variable

Using all the 19 features/Variables in the data to perform a Multiple LSR

Using all the 19 features/Variables in the data to perform a Multiple KNN Regression

Using Feature selections methods to see which features are worthwhile to consider for regression. And see if the accuracy of the model is any better using the best three features from the 'SelectKBest' selection

Using SelectKBest selector

Using Recursive Feature Elimination

PCA to reduce dimensions

## [Project-3.3.4-BC-Challenge.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Project-3.3.4-BC-Challenge.ipynb)

The dataset used here is Crime-FBI-2013-Washington state. The goal or the challenge here is to predict which cities are safe based on the crime or type of crime committed i.e. binary classification.

First of all the Lasso regression is being done on Binary outcome variable i.e. categorical variable. That is why the model would have performed so poorly.

Second of all even the Ridge regression was done with Binary outcome variable and the accuracy was very low. Therefore, I changed the ridge to RidgeClassifier so that it performs better.

Based on accuracy scores Vanilla Logistic regression performed much better than rest of the two models with accuracy score of 98% compared to others. The feature selection and adding features was helpful in improving the accuracy of Vanilla Logistic Regression.

Based on AUC by plotting ROC found that Vanilla Logistic Regression performed extremely well with AUC of 1, whereas Ridge Regression and Ridge Classifier performed low with 0.82/0.85 AUC.

The Lasso I did not attempt to do the ROC since the accuracy was utterly poor. It would have been better if I could use Lasso for Categorical variables too.

**Conclusion: Lastly, it felt better to test the Vanilla Logistic regression for overfitting, and it came out with flying colors of accuracy with minimal error as evidenced by an AUC of 0.97 or 1.**

## [Project-3.4.4-BC-Guided example plus challenge.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Project-3.4.4-BC-Guided%20example%20plus%20challenge.ipynb)

The dataset used here is Epicurious i.e. a recipe dataset. The challenge here is to predict the rating based on the recipe features.

-Trial 1 Modelling SVR with all the features but without null values

Removing the null values but keeping the nutritional information features helped a little bit in the accuracy of the model when compared to the model in the unit lesson which was will null values. But still the model is very poor.

-Trial 2 - Reducing the number of features by PCA
The PCA could help to quite an extent in improving the model from 0.02% accuracy to 0.05%. Basically it could equal the accuracy of modelling of 676 features with as little as 60 features i.e reduction or cutdown by 100 times in the number of features.

-Trial 3 - Changing the model from Regression to Classifier. (We are supposed to work with Regression Model before by Design)
We can change the SV Regression to classifier since the rating is a classification problem.
Converting the model from Regression to Classifier improved model from 5% to 87%.
We can change the outcome variable, rating into a binary classifier rather than multiple classifier to make the model simple and accurate.
Converting the model from Regression to Classifier improved model from 5% to 87%.

-Trial 4 - Feature selection brings down to 30 most valuable features
By bringing the features down to 30 from 676 features the accuracy of the model improved from 78% when it was 676 features to 88% with 30 features......and the also the run time and memory burden has reduced quite a bit.

-Trial 5 - Since PCA and SV Classifier separately increased accuracy....tried them in combination
With as less as PCA components of even one component an accuracy of 87% could be achieved i.e. reduction of features from 676 to one component with higher accuracy.

-Trial 6 - Since feature selection and PCA both had a positive effect, combined the PCA on selected features by method.
The accuracy could improve by very less % i.e. 1% to 88% by feature selection and PCA.

## [Project-3.6.5-Cancer Diagnostics.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Project-3.6.5-Cancer%20Diagnostics.ipynb)

The Dataset is about breast cancer diagnosis. It has various markers or diagnostic features that are categorical and are scored 1-10. the Class features is the Y variable or dependent variable which is a binary classification i.e. 2 or 0 for benign or no cancer and 4 or 1 or malignant or cancer.

Various models that are tested are:

kNN Classifier

Random Forest Classifier

Gradient Boost Classifier

Support Vector Machine

Vanilla Logistic Regression

**Logistic Regression and SVC both have the lowest amount of error (around 1%) compared to other models and also the highest accuracy of 97%. Therefore, these two models with the above parameters tuned can be used for cancer diagnosis in predicting cancer from the above parameters.**

## [Project-4.2.5-BC Challenge Marathon-Clustering.ipynb](https://github.com/aradhagan/Thinkful--Projects/blob/master/Project-4.2.5-BC%20Challenge%20Marathon-Clustering.ipynb)

The dataset is for Boston Marathon challenge, where the marathon runners’ details are given. The challenge here is to find clusters in this dataset and teach the reader some useful information from this dataset.

Used to mean shift clustering method to predict the number of clusters the data forms and it is 4 clusters.

Used the below clustering methods and to verify or validate the similarity of the datapoints in the cluster Silhouette Coefficient was measured. Values range from -1 for very bad clusters to +1 for very dense clusters.

Below is the score for the datapoints.

{'K Mean': 0.74974375327410137,
 'MeanShift': 0.74824606966003016,
 'Spectral': 0.7414239037162158}
 
Used Gradient boost classifier to classify the feature importance from the inbuilt method.

 **Conclusion: We can conclude from this exercise that the all the Boston marathon data can be divided into four clusters based majorly on the run time of 25k mile time or age or division or 10k mile time in that order.**
 

## [Project-4.3.6-BC-Challenge-Make Your Network.ipynb](https://github.com/aradhagan/Thinkful--Projects/blob/master/Project-4.3.6-BC-Challenge-Make%20Your%20Network.ipynb)

The data set is a Melbourne housing dataset with various columns related to house. The goal is to predict the house prices based on these features. But here we need to compare models other than neural networks to MLP (MultiLayer Perceptrons) the basic Neural net. The question here is whether it is worth to have the complexity and achieve accuracy.

Channing alpha to higher penalty made the accuracy and error better but not by too much. It is still better to have little higher accuracy and lower error.

Tried more than 10 but causes the accuracy to go down.

So In conclusion the Neural network supposed to be superior than all other models...in this particular case did not perform to that extent with increase by 1% in accuracy compared to supervised models.

**Conclusion: I think the tradeoff of letting go on the complexity is not worth it because the accuracy did not increase that well in this particular case or dataset.**

## [Project-BC-2.5.2-Challenge Validating a linear regression.ipynb](https://github.com/aradhagan/Thinkful--Projects/blob/master/Project-BC-2.5.2-Challenge%20Validating%20a%20linear%20regression.ipynb)

The dataset is a crime statistics set of Offenses_Known_to_Law_Enforcement_by_State_by_City_2013. We need to predict the property crime in the cities based on the features such as burglary and other crimes.

The linear regression formula used here is linear_formula = 'Property_crime ~ Larceny_theft+Burglary+Rape_1'

**Conclusion: Testing the accuracy of the model with Linear regression gave an accuracy of around 99%.**

## [Thinkful-Project-2.1.8.ipynb](https://github.com/aradhagan/Thinkful--Projects/blob/master/Thinkful-Project-2.1.8.ipynb)
Data Source: United States Department of Transportation.

Department: Bureau of transportation statistics

Data set name: Airline-On-time-data.csv

Subset of Data : From Date January 2016 to January 2019 and at all major airports

https://www.transtats.bts.gov/OT_Delay/ot_delaycause1.asp?display=data&pn=1

Outcome variable: Arrival delay in flights (arr_delay)
Data analysis question: What is the major factor for delay in flights ?

Performed Feature engineering of several variables to see which feature is the major factor in delay of flights.

**Conclusion: Implemented SelectKBest feature extraction/filtering method and selected the top 3 major factors for arrival delays in the flights. They are ['late_aircraft_delay', 'nas_delay', 'carrier_delay']**

## [Unit 1-My first Data plot.ipynb](https://github.com/aradhagan/Thinkful--Projects/blob/master/Unit%201-My%20first%20Data%20plot.ipynb)

The dataset is from kaggle website about black-Friday sales and the customer details and the shopping they do. The challenge here is to do data analysis and answer some basic questions or patterns of customer shopping behavior through visuals or tables or explanations. 

There are 4 Visuals total and each of them answer questions about the shopping behavior of the customer based on the black Friday dataset.

## [Unit 2 DS Fundaments Capstone project-Nagadhara.ipynb](https://github.com/aradhagan/Thinkful--Projects/blob/master/Unit%202%20DS%20Fundaments%20Capstone%20project-Nagadhara.ipynb)

The data set here is Traffic violations. Dataset is Open Parking and Camera Violations. Source: NYC OpenDATA issued by City of New York. https://data.cityofnewyork.us/City-Government/Open-Parking-and-Camera-Violations/nc67-uf89/data

The dataset is hosted by NYC OpenData and created on Jan 4th, 2016 and updated every day. It was made public on 01/11/2016. The dataset used here was last updated on Dec 19, 2018. This dataset contains Open Parking and Camera Violations issued by the City of New York. The Agency that issued the dataset is Department of Finance (DOF).

It has 1048575 Rows and 16 Columns.

- Question 1: Does the time of the day or month of year or Day of the week have any effect on Violations?
  - Step 1: Clean the data of NaNs
  - Step 2: Formatting the 'Violation_Time' column to 24-hour date format for efficient date extractions and readability
  - Step 3: Now plotting the desired analysis in Matplotlib using bar charts
  
- Question 2(a): How far are the Violations given by the Law enforcement officer really lawful from the cases that came to the court?
  - Step 1: We need to calculate how many are not guilty which will give a percentage of Violations that are not really lawful or incorrectly charged.
  
- Question 2(b): How far are the amount of fees charged by Officer correctly charged based on the cases that were charged a fine amount for the violation?
  - Step 1: Find the number of non-null cases where there is a charge or the Fine amount is greater than 0
  - Step 2: From those cases find the number of cases where the charge was correct i.e. where the reduction amount was zero.
  
- Question 3: Which county has the highest number of Violations and does the number of Violations correlate with the population/Number of vehicles in those counties?

- Question 4: Of all the cases appealed to the court, what is the category of violations
  - a) To answer this question
    - Second, we need to check the categories of Violations and see from the cases that appealed is there any category which has the highest number of appeals.
  - b) What is the category of violations that have the highest cases of not guilty?
  
### RESEARCH PROPOSAL
When a person is ticketted and he feels he is not guilty, then it will be highly beneficial for a person to know his chances of getting proven not guilty before deciding to appeal to the court.

To answer this question, first one needs to calculate what the chances of are guilty or not guilty for all the major categories, based on the past/existing data.

Then, using a prediction model, one needs to train the model using the existing data and then predict the chances of not guilty based on a category.


## [Unit 3-Capstone-Supervised learning-3.7.1.ipynb](https://github.com/aradhagan/Thinkful--Projects/blob/master/Unit%203-Capstone-Supervised%20learning-3.7.1.ipynb)
 Power point presentation [Unit 3 Capstone-Supervised learning.pptx](https://github.com/aradhagan/Projects-Thinkful/blob/master/Unit%203%20Capstone-Supervised%20learning.pptx)

This is a regression problem where we are trying to predict House prices based on various features. We also want to know some explanatory power of these features in determining the value of the houses. Some algorithms that do well in predicting here are Random Forest Regression, Gradient Boost regression etc.

The residual plot appears to do well expect that very high/low values are not predicted very well with this model.

The residual histogram plot shows if there is multivariate normality of the error. The outlier or skewness in error must be fixed for better performance.

The Gradient boost model also suggests a top 12 features for a real estate developer to add value to a house and also predictive of the prices of house.

Random Forest Regression has the highest accuracy of 93% and Random Forest Regression model has the next highest with 92% accuracy

In terms of the lowest error Random Forest Regression model has the lowest error compared to all other models.
## [Unit 4-Capstone -NLP Text clustering Modelling-final.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Unit%204-Capstone%20-NLP%20Text%20clustering%20Modelling-final.ipynb)

Challenge: Compare the stability or accuracy of text classification between clustering and model classification
Overview:
1) Data preprocessing
Data cleaning / processing / language parsing
2) Generation of TF_IDF features (Supervised technique) and dimensionality reduction by SVD
TF-IDF Feature generation and selection/reduction with SVD
3) Evaluation of models with TF_IDF features
Model assessment and evaluation with other models
4) Generation of word2vec features (Unsupervised technique)
Word2Vec Unsupervised Feature generation
5) Evaluation of models with Word2vec features
Model assessment and evaluation with other models
With word2vec alone an accuracy of 75% is got. Now how about if we add the TFIDF features.
6) Clustering of data and Evaluation
All the data used for TF-IDF have undergone dimension reduction by SVD and so the first three components are used here to visualize
Before going with the clustering lets us first define some functions to make to easier to build and analyses the clusters
K Means clustering
Mean shift clustering
Affinity Propagation clustering
Affinity propagation is known for predicting absurdly high clusters and we can see here that there lot of cluster predicted.
Spectral clustering

**Conclusions: why Modelling is better than clustering.
Future Scope: The model can be further enhanced even more using sentence2vec or doc2vec embedding to generalize more.**

## [Unit 7 Capstone Ver Final -Nag -submitted.ipynb](https://github.com/aradhagan/Projects-Thinkful/blob/master/Unit%207%20Capstone%20Ver%20Final%20-Nag%20-submitted.ipynb)

 [PROPOSAL FOR FINAL CAPSTONE-Nagadhara.docx](https://github.com/aradhagan/Projects-Thinkful/blob/master/PROPOSAL%20FOR%20FINAL%20CAPSTONE-Nagadhara.docx)
 
Proposal document for the Final capstone project on CNN X-ray image classification using neural-networks implementing keras on tensorflow. [Project code HERE](https://github.com/aradhagan/Projects-Thinkful/blob/master/Unit%207%20Capstone%20Ver%20Final%20-Nag%20-submitted.ipynb)

 [Unit 7 Capstone-Final presentation.pptx (Power point)](https://github.com/aradhagan/Projects-Thinkful/blob/master/Unit%207%20Capstone-Final%20presentation.pptx)

Dataset is NIH Chest X-ray Dataset from Kaggle website.

Overview of Final Capstone project
1. Introduction to the problem
2. Challenge I Classification - Solution: Binary
3. Implementation of Binary classification with RFC, GBC, SVC and MLP
4. Challenge II Improve accuracy - Solution: Hyper parameter tuning by GSCV
5. Implementation of Neural net CNN
6. Challenge III Improve CNN accuracy - Solution: Hyper parameter turning by GSCV
7. Challenge IV Overfitting - Solution: Image Augmentation.
8. Challenge V Overfitting - Solution: Regularization.
9. Future scope
Multilabel Classification with solving class imbalance by Under sampling/Oversampling and Class weights
and Softmax activation

   Solutions:

      1) Object Detection with Bounding boxes.
      2) Probable model change: Use of Structure Correcting Adversarial Network (SCAN) framework (similar to conditional GAN)

**Conclusion:**

**The accuracy increased drastically after hyper paramter tuning and solving the overfitting problem by regularixation.**

**Neural networks need to be tuned to derive maximum benefit/accuracy. The quality of the data input can never be overstated i.e. garbage in garbage out, which determines the success of any modelling technique.
The medical imaging datasets are especially challenging to handle given great variation which becomes a problem since it is hard to decipher patterns in the data.**
