```python
import os
```


```python
os.chdir("/home/bennour/Pywd/")
```

# **PREPARING DATA FOR GENERAL USE**


```python
import numpy as np
import pandas as pd
import lightgbm as lgb
```


```python
np.random.seed(123)
```


```python
# Load the data in the environment.
training = pd.read_csv("/home/bennour/Rprojects/tax_fraud/ahawa/tmp_train.csv")
testing = pd.read_csv("/home/bennour/Rprojects/tax_fraud/ahawa/tmp_test.csv")
```


```python
training = training.drop('id', 1)
testing = testing.drop('id', 1)
```


```python
y_train = training.target
y_test = testing.target
```


```python
x_train1 = lgb.Dataset(training, label = y_train)
y_train1 = lgb.Dataset(testing, label = y_train)
```

# **example of what I want to do**


```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
```

### No grid search included.


```python
model = lgb.LGBMRegressor(objective='regression',
                          num_leaves=5,
                          learning_rate=0.05, 
                          n_estimators=720,
                          max_bin = 55, 
                          bagging_fraction = 0.8,
                          bagging_freq = 5, 
                          feature_fraction = 0.2319,
                          feature_fraction_seed=9, 
                          bagging_seed=9,
                          min_data_in_leaf =6, 
                          min_sum_hessian_in_leaf = 11)
```


```python
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
```


```python
model.fit(training, y_train)
train_prediction = model.predict(training)
prediction = model.predict(testing)
rmse(y_train, train_prediction)
```

# **Essays**


```python
from sklearn.model_selection import GridSearchCV
import seaborn as sns
```

### we want to do regression on the data, using random forest, we specify this below, and we will build on it


```python
m_1 = lgb.LGBMRegressor(objective='regression',
                        boosting_type = 'rf')
```


```python
para_1 = {
    'num_leaves': [25,75],
    'learning_rate': [0.025,0.075],
    'n_estimators': [25,75],
    'max_bin': [25,75],
    'bagging_fraction': [0.025,0.075],
    'bagging_freq': [25,75],
    'min_data_in_leaf': [25,75],
    'min_sum_hessian_in_leaf': [25,75]
}
```


```python
folds = 2

gkf = KFold(n_splits = folds)

g_1 = GridSearchCV(m_1, 
                  param_grid = para_1, 
                  scoring = "neg_mean_squared_error", 
                  cv = gkf.split(training, y_train), 
                  verbose=5)

g_1.fit(training, y_train)
```


```python
g_1.best_params_
```


```python
para_2 = {
 'bagging_fraction':[0.05,0.1],
 'bagging_freq': [1, 50],
 'learning_rate': [0.01, 0.05],
 'max_bin': [50, 100],
 'min_data_in_leaf':[1, 50],
 'min_sum_hessian_in_leaf': [1, 50],
 'n_estimators': [50,100],
 'num_leaves': [10, 50]
}
```


```python
g_2 = GridSearchCV(m_1, 
                  param_grid = para_2, 
                  scoring = "neg_mean_squared_error", 
                  cv = gkf.split(training, y_train), 
                  verbose=5)

g_2.fit(training, y_train)
```


```python
g_2.best_params_
```

### **work still in progress, feedback welcomed**
