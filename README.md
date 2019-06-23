# AirFare-Prediction-Model
An ensemble model designed to predict the airfare with an r2_score of 0.83

Three algorithms are considered:
### XGBoost:
XGBoost is one of the most popular machine learning algorithm these days. Regardless of the type of prediction task at hand, regression or classification. It is well known to provide better solutions than other machine learning algorithms. In fact, since its inception, it has become the "state-of-the-art” machine learning algorithm to deal with structured data.
XGBoost is popular for following reasons,

##### Speed and performance: 
Originally written in C++, it is comparatively faster than other ensemble classifiers.
##### Core algorithm is parallelizable: 
Because the core XGBoost algorithm is parallelizable it can harness the power of multi-core computers. It is also parallelizable onto GPU’s and across networks of computers making it feasible to train on very large datasets as well.
##### Consistently outperforms other algorithm methods: 
It has shown better performance on a variety of machine learning benchmark datasets.
##### Wide variety of tuning parameters: 
XGBoost internally has parameters for cross-validation, regularization, user-defined objective functions, missing values, tree parameters, scikit-learn compatible API etc.
XGBoost (Extreme Gradient Boosting) belongs to a family of boosting algorithms and uses the gradient boosting (GBM) framework at its core. It is an optimized distributed gradient boosting library.
### Boosting
Boosting is a sequential technique which works on the principle of an ensemble. It combines a set of weak learners and delivers improved prediction accuracy. At any instant t, the model outcomes are weighed based on the outcomes of previous instant t-1. The outcomes predicted correctly are given a lower weight and the ones miss-classified are weighted higher. Note that a weak learner is one which is slightly better than random guessing. 

### LGBM Regressor:
Light GBM is a gradient boosting framework that uses tree based learning algorithm. Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm.

The size of data is increasing day by day and it is becoming difficult for traditional data science algorithms to give faster results. Light GBM is prefixed as ‘Light’ because of its high speed. Light GBM can handle the large size of data and takes lower memory to run. Another reason of why Light GBM is popular is because it focuses on accuracy of results. LGBM also supports GPU learning and thus data scientists are widely using LGBM for data science application development.
Implementation of Light GBM is easy, the only complicated thing is parameter tuning. Light GBM covers more than 100 parameters. Below are some of the parameters

#### Parameters
max_depth: It describes the maximum depth of tree. This parameter is used to handle model overfitting. Any time we feel that your model is overfitted, my first advice will be to lower max_depth.
min_data_in_leaf: It is the minimum number of the records a leaf may have. The default value is 20, optimum value. It is also used to deal over fitting
feature_fraction: Used when your boosting(discussed later) is random forest. 0.8 feature fraction means LightGBM will select 80% of parameters randomly in each iteration for building trees.

##### bagging_fraction: 
specifies the fraction of data to be used for each iteration and is generally used to speed up the training and avoid overfitting.
##### lambda: 
lambda specifies regularization. Typical value ranges from 0 to 1.
##### min_gain_to_split: 
This parameter will describe the minimum gain to make a split. It can used to control number of useful splits in tree.
##### Task: 
It specifies the task you want to perform on data. It may be either train or predict.
##### application: 
This is the most important parameter and specifies the application of your model, whether it is a regression problem or classification problem. LightGBM will by default consider model as a regression model.
##### boosting: 
defines the type of algorithm you want to run, default=gdbt
##### learning_rate: 
This determines the impact of each tree on the final outcome. GBM works by starting with an initial estimate which is updated using the output of each tree. The learning parameter controls the magnitude of this change in the estimates. Typical values: 0.1, 0.001, 0.003…
##### num_leaves: 
number of leaves in full tree, default: 31
#### Disadvantage:
Light GBM is sensitive to overfitting and can easily overfit small data. Their is no threshold on the number of rows and is to be used it only for data with 10,000+ rows.
### Randomforest Regressor:
A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap Aggregation, commonly known as bagging. Bagging, in the Random Forest method, involves training each decision tree on a different data sample where sampling is done with replacement.
The basic idea behind this is to combine multiple decision trees in determining the final output rather than relying on individual decision trees.
First the features(X) and the dependent(y) variable values of the data set, are passed to the method created for the random forest regression model. Grid search cross validation method from the sklearn library is used to determine the optimal values to be used for the hyperparameters of the model from a specified range of values. Here, we have chosen the two hyperparameters; max_depth and n_estimators, to be optimized. 
max_depth refers to the maximum depth of the tree and n_estimators, the number of trees in the forest. Ideally, we can expect a better performance from the model when there are more trees. However, we must be cautious of the value ranges we specify and experiment using different values to see how the model performs.
After creating a random forest regressor object, we pass it to the cross_val_score() function which performs K-Fold cross validation on the given data and provides as an output, an error metric value, which can be used to determine the model performance.
After creating a random forest regressor object, we pass it to the cross_val_score() function which performs K-Fold cross validation on the given data and provides as an output, an error metric value, which can be used to determine the model performance.
## Prediction Model
An Ensemble learning approach is used in this proposed approach. We’ve used supervised learning algorithms. A single algorithm may classify the objects poorly. But if we combine multiple classifiers with selection of training set at every iteration and assigning right amount of weight in final voting, we can have good accuracy score for overall classifier. 
We’ve ensembled three algorithms, RandomForest Regressor, XGBosst, LGBM Regressor by stacking one upon the other using StackingCVRegressor.
### StackingCVRegressor
Stacking is an ensemble learning technique to combine multiple regression models via a meta-regressor. The StackingCVRegressor extends the standard stacking algorithm (implemented as StackingRegressor) using out-of-fold predictions to prepare the input data for the level-2 regressor.
In the standard stacking procedure, the first-level regressors are fit to the same training set that is used prepare the inputs for the second-level regressor, which may lead to overfitting. The StackingCVRegressor, however, uses the concept of out-of-fold predictions: the dataset is split into k folds, and in k successive rounds, k-1 folds are used to fit the first level regressor. In each round, the first-level regressors are then applied to the remaining 1 subset that was not used for model fitting in each iteration. The resulting predictions are then stacked and provided -- as input data -- to the second-level regressor. After the training of the StackingCVRegressor, the first-level regressors are fit to the entire dataset for optimal predicitons.
 	
Now we will split the dataset into train and test dataset where Train dataset will be used for model training. We were able to build an ensemble learning based classifier that can recognize the flight fare will vary or not. Finally the model is saved once it satisfies a certain performance criteria.
