
# checking files in my project's directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# This is to supress the warning messages (if any) generated in our code
import warnings
warnings.filterwarnings('ignore')

# Import of fundamental libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as sts
%matplotlib inline

import pandas as pd
full_data = pd.read_csv('full_data.csv',index_col=0)

####################################
## spliting train and test datasets 
###################################
import numpy as np

X = full_data.iloc[:1451,]
y = X['SalePrice']
test = full_data.iloc[1451:,]

X.drop('SalePrice',axis=1,inplace=True)
test.drop('SalePrice',axis=1,inplace=True)

##############################################
## Scaling the data
#############################################
from sklearn.preprocessing import RobustScaler

cols = X.select_dtypes(np.number).columns
transformer = RobustScaler().fit(X[cols])
X[cols] = transformer.transform(X[cols])
test[cols] = transformer.transform(test[cols])

##############################################
## Split data in train and valuation datasetes
##############################################
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=21)


##############################################
# XGBoost
#############################################
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# Instanciate the model
xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror')

#list of parameters to optimize
param_lst = {
    'learning_rate' : [0.01, 0.1, 0.15, 0.3, 0.5],
    'n_estimators' : [100, 500, 1000, 2000, 3000],
    'max_depth' : [3, 6, 9],
    'min_child_weight' : [1, 5, 10, 20],
    'reg_alpha' : [0.001, 0.01, 0.1],
    'reg_lambda' : [0.001, 0.01, 0.1]
}

# Randomizedd search instance
xgb_reg = RandomizedSearchCV(estimator = xgb, 
                             param_distributions = param_lst,
                             n_iter = 100,
                             scoring = 'neg_root_mean_squared_error',
                             cv = 5)

# Looking for the best parametes and timing the search
import time
start = time.time()
xgb_search = xgb_reg.fit(X_train, y_train)
stop = time.time()
ttime = (stop-start)/60
print(f'Tuning XGBoost hyperparameters:{ttime:.2f} minutes')

best_param = xgb_search.best_params_
xgb = XGBRegressor(**best_param)

#####################################
## function to calculate the mean score of cross_val
#####################################
def mean_cross_val(model, X, y):
    score = cross_val_score(model, X, y, cv=5)
    mean = score.mean()
    return mean

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

start = time.time()
xgb.fit(X_train, y_train)   
stop = time.time()

preds = xgb.predict(X_val) 
preds_test_xgb = xgb.predict(test)
mae_xgb = mean_absolute_error(y_val, preds)
rmse_xgb = np.sqrt(mean_squared_error(y_val, preds))
score_xgb = xgb.score(X_val, y_val)
cv_xgb = mean_cross_val(xgb, X, y)

print(f'Mean Absolute Error: {mae_xgb:.4f}')
print(f'Root of Mean Squared Error: {rmse_xgb:.4f}')
print(f'Score (R^2): {score_xgb:.4f}')
print(f'Mean of cross_val score: {cv_xgb:.4f}')
print(f'Time to train the model:{stop-start:.3f} seconds')

### Submission to kaggle.com
subm = np.exp(preds_test_xgb)
submission = pd.DataFrame({'Id': test.index,
                           'SalePrice': subm})

submission.to_csv("submission_xgb.csv", index=False)