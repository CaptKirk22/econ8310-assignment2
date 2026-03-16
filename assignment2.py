import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("assignment2train.csv")

data

data['DateTime'] = pd.to_datetime(data['DateTime'])

data['hour'] = data['DateTime'].dt.hour

data = data.dropna(subset=['meal'])

data

# Upper case before split, lower case after
Y = data['meal']
# make sure you drop a column with the axis=1 argument
X = data.drop(['meal', 'DateTime', 'id'], axis=1)

(X)

x, xt, y, yt = train_test_split(X, Y, test_size=0.1,
     random_state=42)


from xgboost import XGBClassifier


xgb = XGBClassifier(n_estimators=82, max_depth=3, #this just creates the model, not put any data into it
 learning_rate=0.5, objective='binary:logistic') # learning rate means how much does it assume based on data ex: a high learning rate is like assuming all people are like one citizen of that country
# multi means multi class problem
# softmax means highest prob across all classes

xgb

xgb.fit(x, y)

pred = xgb.predict(xt)

pred

print(accuracy_score(yt, pred)*100)