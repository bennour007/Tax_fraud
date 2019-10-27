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

    Fitting 2 folds for each of 256 candidates, totalling 512 fits
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.02007815174215965, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.008398399269772141, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.3s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.4s remaining:    0.0s


    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.02007815174215965, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.008398399269772141, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 


    [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:    0.6s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   4 out of   4 | elapsed:    0.8s remaining:    0.0s


    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.005935863836278545, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.005967975748223111, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.005935863836278545, total=   0.3s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.005967975748223111, total=   0.4s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.1992910256715702, total=   0.3s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.2587374492785147, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.1992910256715702, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.2587374492785147, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.31718106872798646, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.16475530563667062, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.31718106872798646, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.1992910256715702, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.2587374492785147, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.1992910256715702, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.2587374492785147, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.1992910256715702, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.2587374492785147, total=   0.3s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.1992910256715702, total=   0.3s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.2587374492785147, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.027492829269545167, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.008217158010093583, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.027492829269545167, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.008217158010093583, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.005477445247665569, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.005095827646922143, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.005477445247665569, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.005095827646922143, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.24687354624124444, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.24687354624124444, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.18459959875794937, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.22977537214506208, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.18459959875794937, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.22977537214506208, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.1638207818585409, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.1638207818585409, total=   0.3s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.24687354624124444, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.18459959875794937, total=   0.3s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.22977537214506208, total=   0.4s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.18459959875794937, total=   0.3s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.22977537214506208, total=   0.4s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.1638207818585409, total=   0.3s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.1638207818585409, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.18459959875794937, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.22977537214506208, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.18459959875794937, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.22977537214506208, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.02007815174215965, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.008398399269772141, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.02007815174215965, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.008398399269772141, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.005935863836278545, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.005967975748223111, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.005935863836278545, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.005967975748223111, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.1992910256715702, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.2587374492785147, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.1992910256715702, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.2587374492785147, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.1992910256715702, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.2587374492785147, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.1992910256715702, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.2587374492785147, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.1992910256715702, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.2587374492785147, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.1992910256715702, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.2587374492785147, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.027492829269545167, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.008217158010093583, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.027492829269545167, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.008217158010093583, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.005477445247665569, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.005095827646922143, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.005477445247665569, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.005095827646922143, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.24687354624124444, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.24687354624124444, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.18459959875794937, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.22977537214506208, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.18459959875794937, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.22977537214506208, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.24687354624124444, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.18459959875794937, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.22977537214506208, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.18459959875794937, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.22977537214506208, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.24687354624124444, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.24687354624124444, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.18459959875794937, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.22977537214506208, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.18459959875794937, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.22977537214506208, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.02007815174215965, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.008398399269772141, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.02007815174215965, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.008398399269772141, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.020078151742159652, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.008398399269772138, total=   0.3s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.020078151742159652, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.008398399269772138, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.16475530563667073, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.3171810687279865, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.16475530563667073, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.3171810687279865, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.16475530563667073, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.3171810687279865, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.16475530563667073, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.3171810687279865, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.31718106872798646, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.16475530563667073, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.3171810687279865, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.16475530563667073, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.3171810687279865, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.027492829269545167, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.008217158010093583, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.027492829269545167, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.008217158010093583, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.027492829269545174, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.008217158010093576, total=   0.3s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.027492829269545174, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.008217158010093576, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.16382078185854093, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.24687354624124438, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.16382078185854093, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.24687354624124438, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.24687354624124444, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.1638207818585409, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.24687354624124444, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.16382078185854093, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.24687354624124438, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.16382078185854093, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.24687354624124438, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.16382078185854093, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.24687354624124438, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.16382078185854093, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.24687354624124438, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.02007815174215965, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.008398399269772141, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.02007815174215965, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.008398399269772141, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.020078151742159652, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.008398399269772138, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.020078151742159652, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.008398399269772138, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.31718106872798646, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.31718106872798646, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.16475530563667073, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.3171810687279865, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.16475530563667073, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.3171810687279865, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.31718106872798646, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.31718106872798646, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.16475530563667073, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.3171810687279865, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.16475530563667073, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.3171810687279865, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.31718106872798646, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.16475530563667062, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.31718106872798646, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.16475530563667073, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.3171810687279865, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.16475530563667073, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.3171810687279865, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.027492829269545167, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.008217158010093583, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.027492829269545167, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.008217158010093583, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.027492829269545174, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.008217158010093576, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.027492829269545174, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.008217158010093576, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.16382078185854093, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.24687354624124438, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.16382078185854093, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.24687354624124438, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.16382078185854093, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.24687354624124438, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.16382078185854093, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.24687354624124438, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.1638207818585409, total=   0.1s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.24687354624124444, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.16382078185854093, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.24687354624124438, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.16382078185854093, total=   0.2s
    [CV] bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.025, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.24687354624124438, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.016281762129150258, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.004814349982952767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.016281762129150258, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.004814349982952767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.007061149028294354, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.0021769374032576685, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.007061149028294354, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.0021769374032576685, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.009307386020901932, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.006031224618290272, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.009307386020901932, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.006031224618290272, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.009307386020901932, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.006031224618290272, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.009307386020901932, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.006031224618290272, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.009307386020901932, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.006031224618290272, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.009307386020901932, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.006031224618290272, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.014801904859399539, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.004595050493305648, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.014801904859399539, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.004595050493305648, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.0025562396701293956, total=   0.3s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.0013833695789899968, total=   0.3s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.0025562396701293956, total=   0.4s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.0013833695789899968, total=   0.3s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.018105899395325736, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.018105899395325736, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.0051677338013637365, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.004301297724244005, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.0051677338013637365, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.004301297724244005, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.018105899395325736, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.018105899395325736, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.0051677338013637365, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.004301297724244005, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.0051677338013637365, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.004301297724244005, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.018105899395325736, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.018105899395325736, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.0051677338013637365, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.004301297724244005, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.0051677338013637365, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.004301297724244005, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.016281762129150258, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.004814349982952767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.016281762129150258, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.004814349982952767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.007061149028294354, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.0021769374032576685, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.007061149028294354, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.0021769374032576685, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.009307386020901932, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.006031224618290272, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.009307386020901932, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.006031224618290272, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.009307386020901932, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.006031224618290272, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.009307386020901932, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.006031224618290272, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.009307386020901932, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.006031224618290272, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.009307386020901932, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.006031224618290272, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.014801904859399539, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.004595050493305648, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.014801904859399539, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.004595050493305648, total=   0.4s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.0025562396701293956, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.0013833695789899968, total=   0.3s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.0025562396701293956, total=   0.3s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.0013833695789899968, total=   0.3s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.018105899395325736, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.018105899395325736, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.0051677338013637365, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.004301297724244005, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.0051677338013637365, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.004301297724244005, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.018105899395325736, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.018105899395325736, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.0051677338013637365, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.004301297724244005, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.0051677338013637365, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.004301297724244005, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.018105899395325736, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.018105899395325736, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.0051677338013637365, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.004301297724244005, total=   0.4s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.0051677338013637365, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=25, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.004301297724244005, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.016281762129150258, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.004814349982952767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.016281762129150258, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.004814349982952767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.016281762129150255, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.004814349982952763, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.016281762129150255, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.004814349982952763, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.0090460659743873, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.0090460659743873, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.0090460659743873, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.0090460659743873, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.0090460659743873, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.0090460659743873, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.014801904859399539, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.004595050493305648, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.014801904859399539, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.004595050493305648, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.01480190485939953, total=   0.4s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.004595050493305648, total=   0.4s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.01480190485939953, total=   0.4s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.004595050493305648, total=   0.3s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.018105899395325736, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.018105899395325736, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.018105899395325767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.007773265147972424, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.018105899395325767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.007773265147972424, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.018105899395325736, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.018105899395325736, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.018105899395325767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.007773265147972424, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.018105899395325767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.007773265147972424, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.018105899395325736, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.018105899395325736, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.018105899395325767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.007773265147972424, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.018105899395325767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.025, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.007773265147972424, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.016281762129150258, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.004814349982952767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.016281762129150258, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.004814349982952767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.016281762129150255, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.004814349982952763, total=   0.3s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.016281762129150255, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.004814349982952763, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.0090460659743873, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.0090460659743873, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.0090460659743873, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.0090460659743873, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.020120925537810804, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.009046065974387308, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.0090460659743873, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.020120925537810804, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=25, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.0090460659743873, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.014801904859399539, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.004595050493305648, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.014801904859399539, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.004595050493305648, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.01480190485939953, total=   0.4s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.004595050493305648, total=   0.4s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.01480190485939953, total=   0.4s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.004595050493305648, total=   0.4s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.018105899395325736, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.018105899395325736, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.018105899395325767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.007773265147972424, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.018105899395325767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=25, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.007773265147972424, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.018105899395325736, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=25, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.018105899395325736, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=25, num_leaves=75, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.018105899395325767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=25, score=-0.007773265147972424, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.018105899395325767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=25, n_estimators=75, num_leaves=75, score=-0.007773265147972424, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.018105899395325736, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=25, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.018105899395325736, total=   0.1s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=25, num_leaves=75, score=-0.007773265147972428, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.018105899395325767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=25, score=-0.007773265147972424, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.018105899395325767, total=   0.2s
    [CV] bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75 
    [CV]  bagging_fraction=0.075, bagging_freq=75, learning_rate=0.075, max_bin=75, min_data_in_leaf=75, min_sum_hessian_in_leaf=75, n_estimators=75, num_leaves=75, score=-0.007773265147972424, total=   0.2s


    [Parallel(n_jobs=1)]: Done 512 out of 512 | elapsed:  1.9min finished





    GridSearchCV(cv=<generator object _BaseKFold.split at 0x7f355beb5cd0>,
           error_score='raise-deprecating',
           estimator=LGBMRegressor(boosting_type='rf', class_weight=None, colsample_bytree=1.0,
           importance_type='split', learning_rate=0.1, max_depth=-1,
           min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
           n_estimators=100, n_jobs=-1, num_leaves=31, objective='regression',
           random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
           subsample=1.0, subsample_for_bin=200000, subsample_freq=0),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'num_leaves': [25, 75], 'learning_rate': [0.025, 0.075], 'n_estimators': [25, 75], 'max_bin': [25, 75], 'bagging_fraction': [0.025, 0.075], 'bagging_freq': [25, 75], 'min_data_in_leaf': [25, 75], 'min_sum_hessian_in_leaf': [25, 75]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='neg_mean_squared_error', verbose=5)




```python
g_1.best_params_
```




    {'bagging_fraction': 0.075,
     'bagging_freq': 25,
     'learning_rate': 0.025,
     'max_bin': 75,
     'min_data_in_leaf': 25,
     'min_sum_hessian_in_leaf': 25,
     'n_estimators': 75,
     'num_leaves': 25}




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

    Fitting 2 folds for each of 256 candidates, totalling 512 fits
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0013234152825796024, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.3s remaining:    0.0s


    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0015947138142030015, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 


    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.6s remaining:    0.0s


    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.001243320368503708, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 


    [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:    1.1s remaining:    0.0s


    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0015026498352583445, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 


    [Parallel(n_jobs=1)]: Done   4 out of   4 | elapsed:    1.7s remaining:    0.0s


    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0012436849888731866, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0013717160894356627, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0011544637269719391, total=   1.0s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0013100362856133462, total=   0.8s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0038108345517271175, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.004279790545672445, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0038108345517271175, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.004279790545672445, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0038217367906338198, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004109635395098459, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.0038217367906338198, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004109635395098459, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0038108345517271175, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.004279790545672445, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0038108345517271175, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.004279790545672445, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0038217367906338198, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.004109635395098459, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0038217367906338198, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.004109635395098459, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0038108345517271175, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.004279790545672445, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0038108345517271175, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.004279790545672445, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0038217367906338198, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004109635395098459, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.0038217367906338198, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004109635395098459, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0012953491690495351, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0016781376139774198, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0012106920927475835, total=   0.8s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0015168489730428133, total=   0.8s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0012196456142500166, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.001471763737204454, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0011426047384754565, total=   1.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0013414762168382174, total=   1.7s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0039336167508134045, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0043412016051430375, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0039336167508134045, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0043412016051430375, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.00383451189470291, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004106160059284946, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.00383451189470291, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004106160059284946, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0039336167508134045, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0043412016051430375, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0039336167508134045, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0043412016051430375, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.00383451189470291, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.004106160059284946, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.00383451189470291, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.004106160059284946, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0039336167508134045, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0043412016051430375, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0039336167508134045, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0043412016051430375, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.00383451189470291, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004106160059284946, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.00383451189470291, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004106160059284946, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0013234152825796024, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0015947138142030015, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.001243320368503708, total=   0.6s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0015026498352583445, total=   0.7s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0012436849888731866, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0013717160894356627, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0011544637269719391, total=   0.8s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0013100362856133462, total=   1.0s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0038108345517271175, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.004279790545672445, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0038108345517271175, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.004279790545672445, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0038217367906338198, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004109635395098459, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.0038217367906338198, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004109635395098459, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0038108345517271175, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.004279790545672445, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0038108345517271175, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.004279790545672445, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0038217367906338198, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.004109635395098459, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0038217367906338198, total=   1.5s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.004109635395098459, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0038108345517271175, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.004279790545672445, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0038108345517271175, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.004279790545672445, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0038217367906338198, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004109635395098459, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.0038217367906338198, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004109635395098459, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0012953491690495351, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0016781376139774198, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0012106920927475835, total=   0.7s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0015168489730428133, total=   2.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0012196456142500166, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.001471763737204454, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0011426047384754565, total=   1.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0013414762168382174, total=   1.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0039336167508134045, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0043412016051430375, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0039336167508134045, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0043412016051430375, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.00383451189470291, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004106160059284946, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.00383451189470291, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004106160059284946, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0039336167508134045, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0043412016051430375, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0039336167508134045, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0043412016051430375, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.00383451189470291, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.004106160059284946, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.00383451189470291, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.004106160059284946, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0039336167508134045, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0043412016051430375, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0039336167508134045, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0043412016051430375, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.00383451189470291, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004106160059284946, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.00383451189470291, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004106160059284946, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.003850245615315417, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.005091890872158679, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0026470248812174657, total=   0.7s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.003950831972535073, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0022453406883513848, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.002777604457025956, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0010892182916792843, total=   0.9s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0021861929978006563, total=   1.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.017039628839850624, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.00915564127597065, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.017039628839850624, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.00915564127597065, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.006036819593054191, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.005684689185884134, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.006036819593054191, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.005684689185884134, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.017039628839850624, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.00915564127597065, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.017039628839850624, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.00915564127597065, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.006036819593054191, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.005684689185884134, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.006036819593054191, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.005684689185884134, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.017039628839850624, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.00915564127597065, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.017039628839850624, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.00915564127597065, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.006036819593054191, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.005684689185884134, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.006036819593054191, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.005684689185884134, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.004011233486139941, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0051509533214351986, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0027850069082448374, total=   0.6s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.003652105917237085, total=   0.7s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0026706709553847383, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.001619654172643649, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0014406124131458229, total=   1.4s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0011876258101866175, total=   1.7s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.017560199693118447, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.008197694164833761, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.017560199693118447, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.008197694164833761, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.006932783382639954, total=   0.7s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004743140292599827, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.006932783382639954, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004743140292599827, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.017560199693118447, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.008197694164833761, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.017560199693118447, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.008197694164833761, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.006932783382639954, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.004743140292599827, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.006932783382639954, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.004743140292599827, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.017560199693118447, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.008197694164833761, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.017560199693118447, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.008197694164833761, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.006932783382639954, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004743140292599827, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.006932783382639954, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004743140292599827, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.003850245615315417, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.005091890872158679, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0026470248812174657, total=   0.5s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.003950831972535073, total=   0.9s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0022453406883513848, total=   0.6s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.002777604457025956, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0010892182916792843, total=   0.8s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0021861929978006563, total=   0.8s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.017039628839850624, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.00915564127597065, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.017039628839850624, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.00915564127597065, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.006036819593054191, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.005684689185884134, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.006036819593054191, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.005684689185884134, total=   2.1s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.017039628839850624, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.00915564127597065, total=   1.0s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.017039628839850624, total=   1.1s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.00915564127597065, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.006036819593054191, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.005684689185884134, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.006036819593054191, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.005684689185884134, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.017039628839850624, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.00915564127597065, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.017039628839850624, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.00915564127597065, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.006036819593054191, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.005684689185884134, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.006036819593054191, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.005684689185884134, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.004011233486139941, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0051509533214351986, total=   0.6s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0027850069082448374, total=   0.6s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.003652105917237085, total=   0.7s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0026706709553847383, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.001619654172643649, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0014406124131458229, total=   1.0s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0011876258101866175, total=   1.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.017560199693118447, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.008197694164833761, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.017560199693118447, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.008197694164833761, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.006932783382639954, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004743140292599827, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.006932783382639954, total=   0.4s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004743140292599827, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.017560199693118447, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.008197694164833761, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.017560199693118447, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.008197694164833761, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.006932783382639954, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.004743140292599827, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.006932783382639954, total=   0.3s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.004743140292599827, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.017560199693118447, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.008197694164833761, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.017560199693118447, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.008197694164833761, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.006932783382639954, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004743140292599827, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.006932783382639954, total=   0.2s
    [CV] bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.05, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004743140292599827, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0011856374298919552, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0011503132390667963, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0007231743935033088, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0007342742704887496, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0010768242604917786, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0011932691430285541, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0006542278934915854, total=   0.8s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0007613490478455825, total=   0.9s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0025098341060209566, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.002642704216989693, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.002446202950297998, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0026211935488858473, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.002583363975081797, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0027296408429028363, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.0025271062875435613, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.002705024954826798, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0025098341060209566, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.002642704216989693, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.002446202950297998, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0026211935488858473, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.002583363975081797, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0027296408429028363, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0025271062875435613, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.002705024954826798, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0025098341060209566, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.002642704216989693, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.002446202950297998, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0026211935488858473, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.002583363975081797, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0027296408429028363, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.0025271062875435613, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.002705024954826798, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0009215641257718491, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0009055141576048998, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.00042199782096329855, total=   0.8s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0004317395563288196, total=   1.1s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0008625137661378203, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0009183979595278731, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0003848344456086995, total=   1.5s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.00044580058301508846, total=   2.7s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.002324249419551901, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.002093278799367755, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.00225967817784204, total=   0.6s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0020440795080794447, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0022559404571946147, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.002133955886831353, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.0021954972203479728, total=   0.8s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.002088900234914333, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.002324249419551901, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.002093278799367755, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.00225967817784204, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0020440795080794447, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0022559404571946147, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.002133955886831353, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0021954972203479728, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.002088900234914333, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.002324249419551901, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.002093278799367755, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.00225967817784204, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0020440795080794447, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0022559404571946147, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.002133955886831353, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.0021954972203479728, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.002088900234914333, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0011856374298919552, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0011503132390667963, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0007231743935033088, total=   0.6s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0007342742704887496, total=   0.6s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0010768242604917786, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0011932691430285541, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0006542278934915854, total=   0.9s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0007613490478455825, total=   0.9s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0025098341060209566, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.002642704216989693, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.002446202950297998, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0026211935488858473, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.002583363975081797, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0027296408429028363, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.0025271062875435613, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.002705024954826798, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0025098341060209566, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.002642704216989693, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.002446202950297998, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0026211935488858473, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.002583363975081797, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0027296408429028363, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0025271062875435613, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.002705024954826798, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.0025098341060209566, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.002642704216989693, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.002446202950297998, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0026211935488858473, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.002583363975081797, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0027296408429028363, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.0025271062875435613, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.002705024954826798, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0009215641257718491, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0009055141576048998, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.00042199782096329855, total=   0.7s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0004317395563288196, total=   0.8s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0008625137661378203, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0009183979595278731, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0003848344456086995, total=   1.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.00044580058301508846, total=   1.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.002324249419551901, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.002093278799367755, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.00225967817784204, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0020440795080794447, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0022559404571946147, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.002133955886831353, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.0021954972203479728, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.002088900234914333, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.002324249419551901, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.002093278799367755, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.00225967817784204, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0020440795080794447, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0022559404571946147, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.002133955886831353, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0021954972203479728, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.002088900234914333, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.002324249419551901, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.002093278799367755, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.00225967817784204, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0020440795080794447, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0022559404571946147, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.002133955886831353, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.0021954972203479728, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=1, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.002088900234914333, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0051538250954270325, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.002659820637582263, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0064056468007057535, total=   0.6s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0010349153340826248, total=   0.6s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0018646007618433452, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0019460192832040262, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.002023143966838846, total=   0.9s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0011769115715121604, total=   0.9s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.01453917286389659, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.004443367307315774, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.01426918546686476, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0043079875861918885, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004366130152837284, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004279354160820148, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004241719915779936, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004162516561192145, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.01453917286389659, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.004443367307315774, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.01426918546686476, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0043079875861918885, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.004366130152837284, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.004279354160820148, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.004241719915779936, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.004162516561192145, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.01453917286389659, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.004443367307315774, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.01426918546686476, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0043079875861918885, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004366130152837284, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004279354160820148, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004241719915779936, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004162516561192145, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.005535531237693028, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0023975415805085488, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0069989281397521445, total=   0.7s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0006733006868422495, total=   0.8s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.002307090230034152, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.001498591145487616, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.00267078847315621, total=   1.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0003966448374631348, total=   1.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.015984992922885365, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.004252299095251082, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.015543834888017315, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.00398041519724539, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.00564268890303075, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0027185246446748998, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.00543873524435638, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.00257116612491814, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.015984992922885365, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.004252299095251082, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.015543834888017315, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.00398041519724539, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.00564268890303075, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0027185246446748998, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.00543873524435638, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.00257116612491814, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.015984992922885365, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.004252299095251082, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.015543834888017315, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.00398041519724539, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.00564268890303075, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0027185246446748998, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.00543873524435638, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.01, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.00257116612491814, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0051538250954270325, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.002659820637582263, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0064056468007057535, total=   0.6s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0010349153340826248, total=   0.6s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0018646007618433452, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0019460192832040262, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.002023143966838846, total=   0.9s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0011769115715121604, total=   1.0s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.01453917286389659, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.004443367307315774, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.01426918546686476, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0043079875861918885, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004366130152837284, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004279354160820148, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004241719915779936, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004162516561192145, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.01453917286389659, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.004443367307315774, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.01426918546686476, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0043079875861918885, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.004366130152837284, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.004279354160820148, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.004241719915779936, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.004162516561192145, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.01453917286389659, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.004443367307315774, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.01426918546686476, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.0043079875861918885, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004366130152837284, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.004279354160820148, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004241719915779936, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=50, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.004162516561192145, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.005535531237693028, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.0023975415805085488, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0069989281397521445, total=   0.7s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.0006733006868422495, total=   0.8s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.002307090230034152, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.001498591145487616, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.00267078847315621, total=   1.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.0003966448374631348, total=   1.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.015984992922885365, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.004252299095251082, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.015543834888017315, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.00398041519724539, total=   0.3s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.00564268890303075, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0027185246446748998, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.00543873524435638, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=1, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.00257116612491814, total=   0.5s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.015984992922885365, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=10, score=-0.004252299095251082, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.015543834888017315, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=50, num_leaves=50, score=-0.00398041519724539, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.00564268890303075, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=10, score=-0.0027185246446748998, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.00543873524435638, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=1, n_estimators=100, num_leaves=50, score=-0.00257116612491814, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.015984992922885365, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=10, score=-0.004252299095251082, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.015543834888017315, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=50, num_leaves=50, score=-0.00398041519724539, total=   0.2s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.00564268890303075, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=10, score=-0.0027185246446748998, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.00543873524435638, total=   0.4s
    [CV] bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50 
    [CV]  bagging_fraction=0.1, bagging_freq=50, learning_rate=0.05, max_bin=100, min_data_in_leaf=50, min_sum_hessian_in_leaf=50, n_estimators=100, num_leaves=50, score=-0.00257116612491814, total=   0.4s


    [Parallel(n_jobs=1)]: Done 512 out of 512 | elapsed:  3.6min finished





    GridSearchCV(cv=<generator object _BaseKFold.split at 0x7f355ab229d0>,
           error_score='raise-deprecating',
           estimator=LGBMRegressor(boosting_type='rf', class_weight=None, colsample_bytree=1.0,
           importance_type='split', learning_rate=0.1, max_depth=-1,
           min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
           n_estimators=100, n_jobs=-1, num_leaves=31, objective='regression',
           random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
           subsample=1.0, subsample_for_bin=200000, subsample_freq=0),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'bagging_fraction': [0.05, 0.1], 'bagging_freq': [1, 50], 'learning_rate': [0.01, 0.05], 'max_bin': [50, 100], 'min_data_in_leaf': [1, 50], 'min_sum_hessian_in_leaf': [1, 50], 'n_estimators': [50, 100], 'num_leaves': [10, 50]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='neg_mean_squared_error', verbose=5)




```python
g_2.best_params_
```




    {'bagging_fraction': 0.1,
     'bagging_freq': 1,
     'learning_rate': 0.01,
     'max_bin': 100,
     'min_data_in_leaf': 1,
     'min_sum_hessian_in_leaf': 1,
     'n_estimators': 100,
     'num_leaves': 50}



### **work still in progress, feedback welcomed**
