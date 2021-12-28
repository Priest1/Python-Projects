# Austin Priest
# CS379
#
# The purpose of this program is to implement a supervised machine learning algorithm to
# evaluate the dataset and predict the survival of the passengers on the Titanic cruise ship.
# For this program I will be using the Decision Tree algorithm and the Titanic dataset


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


dataset_source = pd.read_excel('CS379T-Week-1-IP.xls')
dataset_source = dataset_source.drop(['body', 'cabin', 'boat'], axis=1)
# I am dropping these columns because if they have a body id or boat number then that assumes they
# did or did not survive also the cabin will already be covered by which class they were in.

dataset_source["home.dest"] = dataset_source["home.dest"].fillna("NA")
dataset_source = dataset_source.dropna()  # This will replace any missing fields with NA


# The below function will format the rest of the data so that the Decision Tree algorithm can accept it.
# Both the sex & embarked fields are string values that will be run through a preprocessing function.
# This will convert all of the keys for these columns from string to integer which will make it able to classify them
def preprocess_dataset_source(df):
    processed_dataset_source = df.copy()
    column_encoding = preprocessing.LabelEncoder()
    processed_dataset_source.sex = column_encoding.fit_transform(processed_dataset_source.sex)
    processed_dataset_source.embarked = column_encoding.fit_transform(processed_dataset_source.embarked)
    processed_df = processed_dataset_source.drop(['name', 'ticket', 'home.dest'], axis=1)
    return processed_df


processed_dataset = preprocess_dataset_source(dataset_source)

# Now X will contain all of the columns except for the survived column and Y will contain only the survived column.
X = processed_dataset.drop(['survived'], axis=1).values
y = processed_dataset['survived'].values

# X & y will now be further moved into training & testing sets. The training set will be processed by the Decision Tree
# algorithm to form the model for training. After we have formed the model the program will classify the testing
# set to determine the accuracy of the model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

dTree = DecisionTreeClassifier()
dTree.fit(X_train, y_train)
dTree_report = round(dTree.score(X_train, y_train) * 100, 2)

report = pd.DataFrame({
    'Algorithm': ['Decision Tree'],
    'Accuracy Score': [dTree_report],
    'F1 Score': f1_score(y_test, dTree.predict(X_test))})
print(report)
