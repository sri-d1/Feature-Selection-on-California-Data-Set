
# California Housing Price Predictor Using Feature Selection and Dimensionality Reduction



This repository is designed to examine California housing data and forecast housing prices through the application of linear regression and feature selection and Dimensionality reduction as a machine learning technique. The implementation leverages the widely-used scikit-learn library to handle linear regression and manage the machine learning components of the analysis.

## DataSet Details

This dataset was obtained from the StatLib repository. https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

It can be downloaded/loaded using the sklearn.datasets.fetch_california_housing function.

California Housing Dataset in Sklearn Documentation
20640 samples
8 Input Features:
MedInc median income in block group
HouseAge median house age in block group
AveRooms average number of rooms per household
AveBedrms average number of bedrooms per household
Population block group population
AveOccup average number of household members
Latitude block group latitude
Longitude block group longitude
Target: Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000
## Language / Libraries

Language: Python

Packages: Sklearn, Matplotlib, Seaborn

## Usage

The code is built on Google Colab on an iPython Notebook.

```bash
Download the repository, upload the notebook and dataset on colab, and execute!
```

## Lessons Learned / Feature Selection Techniques Used



Lasso Regularization
```bash
Lasso regularization, also known as L1 regularization, is a technique in machine learning that adds a penalty term to the objective function during model training, encouraging the model to prefer sparse solutions by promoting some of the model's coefficients to exactly zero. This helps prevent overfitting and can lead to more interpretable and efficient models by selecting only the most relevant features for prediction.
```
Mutual Information
 
```bash
Mutual information is a measure in information theory that quantifies the degree of dependence between two random variables by assessing how much knowing the value of one variable reduces uncertainty about the other. It is commonly used in various fields, including machine learning and signal processing, to capture the statistical dependence and information shared between variables.
```
Pearson Correlation
 
```bash
Pearson correlation is a statistical measure that quantifies the linear relationship between two continuous variables, providing a coefficient that ranges from -1 to 1. A correlation of 1 indicates a perfect positive linear relationship, -1 denotes a perfect negative linear relationship, and 0 suggests no linear correlation between the variables.
```
Recurive Feature Elimination (RFE)
 
```bash
Recursive Feature Elimination (RFE) is a feature selection technique used in machine learning to identify and retain the most relevant features for model training. It works by recursively removing the least important features based on model performance until the desired number of features is reached, helping to improve model efficiency and reduce overfitting.
```
Sequential Feature Selection
 
```bash
Sequential Feature Selection is a technique in machine learning where features are added or removed iteratively based on their impact on model performance. This process involves evaluating different subsets of features sequentially to identify the most informative combination, enhancing model interpretability and potentially improving predictive accuracy.
```
PCA 
```bash
Principal Component Analysis (PCA) is a dimensionality reduction technique used in machine learning and statistics to transform a dataset into a new coordinate system, capturing the most significant variations in the data. By representing the data in terms of principal components, which are orthogonal and ordered by their variance, PCA helps simplify the dataset and can be instrumental in reducing computational complexity and noise in subsequent analyses.
```
    
    
## Observations
Below are the modal evaluation values observed by implementing filter and dimensionality reduction using linear regression on the dataset.

| Technique        | R2 Score         | MSE error  |
| ------------- | ------------- |------------- |
| Linear Regression without feature selection     |0.577         | 0.524              |
| Lasso Regularization           | 0.589        |       0.553           |
| Mutual Information Using Percentile          | 0.609       |      0.522          |
| Mutual Information Using K Number           | 0.597       |       0.530           |
| Pearson Correlation Using Individual Values           | 0.615        |       0.522          |
| Pearson Correlation Using Mutiple Correlation            | 0.619      |      0.532          |
| Recurive Feature Elimination (RFE)            | 0.607     |     0.495          |
| Sequential Feature Selection            |0.604      |      0.535          |
| PCA            | 0.611    |     0.507         |


Conclusion from Observations:

Feature Selection using Pearson Correlation Using Mutiple Correlation predicts the best r2score of 0.619


## License

[MIT](https://choosealicense.com/licenses/mit/)

