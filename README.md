# Waiter Tips Prediction

## Project Overview

This project predicts waiter tips based on features from a dataset. It involves data preprocessing, encoding categorical variables, scaling features, and training a linear regression model to predict the `tip` variable.

## Data Preprocessing

### Import Libraries

The following libraries are used:
- **pandas**: For data manipulation
- **numpy**: For numerical operations
- **matplotlib**: For visualization
- **seaborn**: For statistical visualization
- **sklearn**: For machine learning tasks

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
```

### Load the Data

Load the dataset from a CSV file:

```python
Tips_Data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/tips.csv")
```

### Exploratory Data Analysis (EDA)

1. **Describe the Data**

   ```python
   Tips_Data.head()
   Tips_Data.tail()
   Tips_Data.shape
   Tips_Data.info()
   Tips_Data.describe()
   Tips_Data.duplicated()
   ```

2. **Visualizations**

   Visualize the distribution and boxplots of features:

   ```python
   fig, axes = plt.subplots(3, 2, figsize=(15, 25))
   # Plot code here
   plt.tight_layout()
   plt.show()

   fig, axes = plt.subplots(1, 2, figsize=(15, 5))
   # Plot code here
   plt.tight_layout()
   plt.show()
   ```

### Data Cleaning

**Removing Outliers**

Outliers are removed based on the interquartile range (IQR) method:

```python
Q1_total_bill = Tips_Data['total_bill'].quantile(0.25)
Q3_total_bill = Tips_Data['total_bill'].quantile(0.75)
IQR_total_bill = Q3_total_bill - Q1_total_bill

Q1_size = Tips_Data['size'].quantile(0.25)
Q3_size = Tips_Data['size'].quantile(0.75)
IQR_size = Q3_size - Q1_size

Q1_tip = Tips_Data['tip'].quantile(0.25)
Q3_tip = Tips_Data['tip'].quantile(0.75)
IQR_tip = Q3_tip - Q1_tip

# Calculate bounds
lower_bound_total_bill = Q1_total_bill - 1.5 * IQR_total_bill
upper_bound_total_bill = Q3_total_bill + 1.5 * IQR_total_bill

lower_bound_size = Q1_size - 1.5 * IQR_size
upper_bound_size = Q3_size + 1.5 * IQR_size

lower_bound_tip = Q1_tip - 1.5 * IQR_tip
upper_bound_tip = Q3_tip + 1.5 * IQR_tip

# Remove outliers
Tips_Data_no_outliers = Tips_Data[(Tips_Data['total_bill'] >= lower_bound_total_bill) & 
                                  (Tips_Data['total_bill'] <= upper_bound_total_bill) &
                                  (Tips_Data['size'] >= lower_bound_size) & 
                                  (Tips_Data['size'] <= upper_bound_size) & 
                                  (Tips_Data['tip'] >= lower_bound_tip) & 
                                  (Tips_Data['tip'] <= upper_bound_tip)]
```

**Visualize Cleaned Data**

```python
fig, axes = plt.subplots(3, 2, figsize=(15, 25))
# Plot code here
plt.tight_layout()
plt.show()
```

### Data Transformation

**Encoding Categorical Variables**

Categorical variables are converted using one-hot encoding:

```python
Tips_Data_Encoded = pd.get_dummies(Tips_Data_no_outliers, columns=['sex', 'smoker', 'day', 'time'])
```

**Scaling Features**

Standardize features to improve model performance:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(Tips_Data_Encoded.drop(columns=['tip']))
```

### Train-Test Split

Split the data into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Tips_Data_Encoded['tip'], test_size=0.2, random_state=42)
```

## Model Training and Evaluation

**Training the Model**

Train a Linear Regression model and evaluate it:

```python
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

**Performance Metrics**

Evaluate model performance on training and testing sets:

```python
print("Training Mean Squared Error:", mean_squared_error(y_train, y_train_pred))
print("Testing Mean Squared Error:", mean_squared_error(y_test, y_test_pred))
print("Training R^2 Score:", model.score(X_train, y_train))
print("Testing R^2 Score:", model.score(X_test, y_test))
```

## Conclusion

The project demonstrates the end-to-end process of predicting waiter tips using Linear Regression, including data preprocessing, outlier removal, feature encoding, and model evaluation.
