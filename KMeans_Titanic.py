# Austin Priest
# CS473
#
# The purpose of this program is to implement an unsupervised clustering algorithm to
# evaluate the dataset and predict the survival of the passengers on the Titanic cruise ship.
# For this program I will be using K-Means and the Titanic dataset
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing


df = pd.read_excel('CS473_Titanic_data.xls')

df.drop(['body', 'name'], 1, inplace=True)
df.fillna(0, inplace=True)
# I am dropping these columns because if they have a body id then that assumes they
# did or did not survive the name is not necessary to predict survival.

def format_non_numeric_data(df):
    columns = df.columns.values

    for column in columns:
        values = {}

        def convert_to_int(val):
            return values[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in values:
                    values[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df


df = format_non_numeric_data(df)

df.drop(['boat'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])
# I am dropping these columns because if they have a boat number then that assumes they did or did not survive.

Kmean = KMeans(n_clusters=2)
Kmean.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = Kmean.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct / len(X))
