### Import Library

import numpy as np
import pandas as pd
import scipy as sp
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import matplotlib

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))
def rmspe_xgb(y, yhat):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)
### Data Preprocessing

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
store = pd.read_csv("store.csv")

# assume store open
train.fillna(1., inplace=True)
train.loc[train.Open.isnull(), 'Open'] = 1.
test.fillna(1., inplace=True)

# remove closed store
train = train[train["Open"] != 0]
# remove sales <= 0
train = train[train["Sales"] > 0]

#train.replace(['a', 'b', 'c', 'd'], [1., 2., 3., 4.])

# join with Store.csv
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

feat = ['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth', \
                'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', \
                'Promo2SinceYear', 'SchoolHoliday', 'StateHoliday', 'DayOfWeek', 'year', 'month', 'day', 'StoreType', 'Assortment']

### feature
def build_feat(feat, data):
    # School Holiday
    #feat.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)
    
    train.fillna(0, inplace=True)
    # State Holiday
    #feat.append('StateHoliday')
    #data.loc[data['StateHoliday'] == "a", 'StateHoliday'] = '1'
    #data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    #data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    #data['StateHoliday'] = data['StateHoliday'].astype(float)
    
    # Date
    #feat.append('DayOfWeek')
    #feat.append('year')
    data['year'] = data.Date.apply(lambda x: x.split('-')[0]).astype(float)
    #feat.append('month')
    data['month'] = data.Date.apply(lambda x: x.split('-')[1]).astype(float)
    #feat.append('day')
    data['day'] = data.Date.apply(lambda x: x.split('-')[2]).astype(float)
    
    # Store Type
    #feat.append('StoreType')
    #data.loc[data['StoreType'] == 'a', 'StoreType'] = '1'
    #data.loc[data['StoreType'] == 'b', 'StoreType'] = '2'
    #data.loc[data['StoreType'] == 'c', 'StoreType'] = '3'
    #data.loc[data['StoreType'] == 'd', 'StoreType'] = '4'
    #data['StoreType'] = data['StoreType'].astype(float)
    
    # Assortment
    #feat.append('Assortment')
    #data.loc[data['Assortment'] == 'a', 'Assortment'] = '1'
    #data.loc[data['Assortment'] == 'b', 'Assortment'] = '2'
    #data.loc[data['Assortment'] == 'c', 'Assortment'] = '3'
    #data['Assortment'] = data['Assortment'].astype(float)
    data.replace(['0','a', 'b', 'c', 'd'], [0. ,1., 2., 3., 4.], inplace=True)
    
build_feat(feat, train)
build_feat(feat, test)
test.fillna(0., inplace=True)

### Parameter
param = {"objective": "reg:linear",
        "eta": 0.3, # 0.3 
        "max_depth": 20, # 10
        "silent": 1,
        "seed": 10,
         "verbose":2
        }
num_tree = 50

### Train Model
x_train, x_valid = train_test_split(train, test_size = 0.01)
y_train = np.log1p(x_train.Sales) # np.log(x_train["Sales"] + 1)
y_valid = np.log1p(x_valid.Sales) # np.log(x_valid["Sales"] + 1)
d_train = xgb.DMatrix(x_train[feat], y_train)
d_valid = xgb.DMatrix(x_valid[feat], y_valid)

watch = [(d_valid, 'eval'), (d_train, 'train')]
bst = xgb.train(param, d_train, num_tree, evals=watch)

yhat = bst.predict(xgb.DMatrix(x_valid[feat]))
error = rmspe(x_valid.Sales.values, np.expm1(yhat))

d_test = xgb.DMatrix(test[feat])
y_pred = bst.predict(d_test)

result = pd.DataFrame({"ID": test["Id"], "Sales": np.expm1(y_pred)})
result.to_csv("submission.csv", index=False)
