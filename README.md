# EDA
Exploratory Data Analysis with Python

  1- Data Inspection
What is the story of the dataset?
What do the columns represent?

df.head(): Displays the first few rows of the dataset.
df.info(): Displays information about the dataset, data types and missing values.
df.shape: Displays dimensions (rows, columns)
df.columns: Returns the column names.
df.dtypes: Displays data types of each column.
df.describe().T: Displays summary statistics for numerical non-null columns.
df.value_counts(): Displays count of unique values
df[”column_name”].nunique() / df[”column_name”].unique()
df.drop(”column_name”, axis=1): Removes useless columns

  2- Data Type Conversions
df.dtypes: Displays column data types
df["column_name"].astype(): Converts a column to a specific data type.
df.column_name = pd.Categorical(df.column_name)
pd.to_datetime(): Converts a column to a datetime format.

  3- Handling Missing Values
represented as NaN, NULL, or other placeholders. 

df.fillna(value): Replace all null values with a specific value, using domain knowledge to replace missing values with appropriate values.
df.fillna(df[”column_name”].mean()): For numerical variables, imputing missing values using techniques like mean, median, or interpolation.
df.fillna(method=”bfill”/ “ffill”): For categorical data replace with mode, forward fill, or backward fill.
df.isnull().sum() / df.notnull().sum(): Displays percentage of null values in each column
df.dropna(how="all"): Removes rows or columns with too many missing values.
df[df.isnull().any(axis=1)]: Displays rows with at least one null value
df[df.notnull().all(axis=1)]: Displays rows with no null values

Techniques for assessing the randomness of missing values: heatmaps, barcharts, histograms, correlation analysis, MCAR test

  Missing value visualization: 
import missingno as msno
msno.bar()
msno.matrix()
msno.heatmap()

4- Handling Duplicates:
df.duplicated(): Checks for duplicate rows
df.drop_duplicates(): Removes duplicate rows

5- Outlier Detection and Handling 
Identify with domain knowledge, visualization techniques (box plot, scatter plots, histograms) or  statistical methods

Z-Score or Standard Deviation Method: Data points beyond a certain threshold
from scipy import stats
z_scores = np.abs(stats.zscore(data))
threshold = 3  # Set your threshold for what constitutes an outlier
outliers = np.where(z_scores > threshold)[0]

IQR (Interquartile Range) Method: 
q1 = df[“data”].quantile(0.25)
q3 = df[“data”].quantile(0.75)
IQR = q3 - q1
lower_bound = q1 - 1.5 * IQR
upper_bound = q3 + 1.5 * IQR
outliers = df[(df[“data”] < lower_bound) | (df[“data”] > upper_bound)]

The Local Outlier Factor (LOF): used for multiple outliers observation. 
from sklearn.neighbors import LocalOutlierFactor

Decide whether to remove outliers or transform them to be less extreme, depending on the nature of your analysis/ imputation.

6- Standardization and Normalization:
Depending on the analysis and the models you plan to use, you may need to standardize (mean = 0, standard deviation = 1) or normalize the data (scale to a specified range, e.g., [0, 1]).
from sklearn import preprocessing
preprocessing.scale(df)
preprocessing.normalize(df)
preprocessing.MinMaxScaler(feature_range=(10,20))

7- Handling Categorical Data:

[df[col].value_counts() for col in df.columns if df[col].dtypes in ["category"]]: Displays counts of unique values for categorical columns
cat_cols = df.select_dtypes(include=[“category”]): Selects categorical columns
df[“column_name”].value_counts().plot.barh: Plots horizontal bar chart for value counts of a specific categorical column
df[cat_cols].nunique(): Displays the number of unique values for each categorical column

Identify and categorize columns
cat_cols = [col for col in df.columns if df[col].dtypes in ["object", "category", "bool"]]: Select categorical columns
num_but_cat = [col for col in df.columns if df[col].dtypes in ["int", "float"] and df[col].nunique() < 10]: extract numerically represented but actually categorical columns
cat_cols += num_but_cat: Update current categorical variables
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and df[col].dtypes in ["category", "object"]]: identify high cardinality categorical variables
(Variables with a high number of classes, e.g., "Name" with too many classes for meaningful interpretability)
cat_cols = [col for col in cat_cols if col not in cat_but_car]: Categorical variables
others = [col for col in df.columns if col not in cat_cols]: Displays columns that are not categorical

Convert categorical variables into a numerical format suitable for analysis. One-hot encoding (binary columns for each category in seperate rows) and label encoding (assigning a unique numerical label to each category for ordinal data). 
import pandas as pd
df_one_hot = pd.get_dummies(df, columns=cat_cols): One-hot encoding

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["encoded_cat"] = le.fit_transform(df["categorical_col"])

8- Handling Numerical Data:
pd.cut(): Converts numerical data into categorical by specifying bin numbers and known category intervals.
pd.qcut(): Divides data based on quartiles, useful when category boundaries need to be determined by quantiles.

Feature Scaling:
If your analysis involves algorithms sensitive to feature scales (e.g., distance-based algorithms), apply feature scaling techniques such as Min-Max scaling or Z-score standardization.

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

9- Data Integrity Checks: 
Check for noisy data that are inconsistent data entries, contradictory information, or values that violate business rules.

10- Text Data Cleaning:
str.strip()
df[“column_name”].str.replace()
apply(lambda x: str(x))
regex

11- Handling Imbalanced Data:
In classification tasks, if the classes are imbalanced, consider techniques like oversampling the minority class or undersampling the majority class to balance the dataset.

12- Documenting Changes:
Keep a record of all changes, reasons for each change made, especially if it involves data imputation or outlier handling, during the data cleaning.

13- Validation and Quality Checks:
ensure that the dataset is free from errors, may include statistical summaries and data visualization.

14- Data Backup:
It's a good practice to create a backup or copy of the cleaned data before proceeding with analysis.

15- Reproducibility:
Ensure that your data cleaning process is reproducible by documenting the code and steps used for cleaning. This helps if you need to clean new data in the future or share your work with others.

