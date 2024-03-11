import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import catboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split

#выгружаем данные
data = pd.read_csv('test/train_df.csv')
test = pd.read_csv('test/test_df.csv')

#заменяем не обрабатываемые символы и проверим, что лишнего не осталось
data.replace(" ", np.nan, inplace=True)
test.replace(" ", np.nan, inplace=True)

data["target"] = data.target.fillna(0).astype(float)
test["target"] = test.target.fillna(0).astype(float)

off_cols = ['target', 'search_id', 'feature_0', 'feature_3', 'feature_4', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_14', 'feature_73', 'feature_74', 'feature_75']
target_cols = ['target']

X_train, X_val, y_train, y_val = train_test_split(data.drop(columns=off_cols, axis=1).values, data[target_cols].values,
                                                       train_size=0.9,
                                                       random_state=4)
X_test, _, y_test, _ = train_test_split(test.drop(columns=off_cols, axis=1).values, test[target_cols].values,
                                                       train_size=0.99999,
                                                       random_state=41)
# X_train, y_train = data[feature_cols].drop(columns=['search_id', 'feature_0'], axis=1).values, data[target_cols].values
# X_test, y_test = test[feature_cols].drop('search_id', axis=1).values, test[target_cols].values
X_train.shape, y_train.shape

boosting_model = catboost.CatBoostClassifier(n_estimators=120)

boosting_model.fit(X_train, y_train, eval_set=(X_val, y_val))

y_train_predicted = boosting_model.predict_proba(X_train)[:, 1]
y_test_predicted = boosting_model.predict_proba(X_test)[:, 1]

train_ndcg = ndcg_score(y_test.reshape(1, len(y_test)), y_test_predicted.reshape(1, len(y_test)))
print(f'best NDCG metric: {train_ndcg}')