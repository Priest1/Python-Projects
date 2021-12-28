# CS379
# Austin Priest
#
# The purpose of this program is to implement a machine learning algorithm to
# evaluate the data and predict credit card of the customers.
# For this program I will be using the Random Forest algorithm and the German Credit Fraud dataset

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

data_source = pd.read_csv("german_credit_fraud.csv")

data_source = data_source.drop(
    ['id', 'credit_history', 'employment', 'location', 'other_parties', 'residence_since', 'property_magnitude',
     'other_payment_plans', 'existing_credits', 'num_dependents', 'own_telephone', 'foreign_worker'], axis=1)


# I am dropping these columns because if they are not needed for the determination of fraud


# The below function will format the checking column in the data so that the Random Forest algorithm can accept it.
# This will convert all of the keys for this columns from string to integer which will make it able to classify them
def checking_LabelEncoder(text):
    if text == "no checking":
        return 1
    elif text == "x<0":
        return 2
    elif text == "0<=x<200":
        return 3
    elif text == "x>=200":
        return 4
    else:
        return 0


# The below function will format the savings column in the data so that the Random Forest algorithm can accept it
# the same way the checking function encoded them.
def savings_LabelEncoder(text):
    if text == "no known savings":
        return 1
    elif text == "x<100":
        return 2
    elif text == "100=<x<500":
        return 3
    elif text == "500=<x<1000":
        return 4
    elif text == "x>=1000":
        return 5
    else:
        return 0


# The below function will format the job column in the data so that the Random Forest algorithm can accept it
# the same way the checking & savings functions encoded them.
def job_LabelEncoder(text):
    if text == "unemp/unskilled":
        return 1
    elif text == "unskilled resident":
        return 2
    elif text == "skilled":
        return 3
    elif text == "high qualif/self emp/mgmt":
        return 4
    else:
        return 0


def housing_LabelEncoder(text):
    if text == "for free":
        return 0
    elif text == "rent":
        return 1
    elif text == "own":
        return 2


# The below section of code is applying the encoding to the corresponding columns.
data_source["savings"] = data_source["savings"].apply(savings_LabelEncoder)
data_source["checking"] = data_source["checking"].apply(checking_LabelEncoder)
data_source["job"] = data_source["job"].apply(job_LabelEncoder)
data_source["housing"] = data_source["housing"].apply(housing_LabelEncoder)

# The below section is encoding the remaining columns similar to the functions above.
for col in ["personal_status", "purpose", "class"]:
    le = LabelEncoder()
    le.fit(data_source[col])
    data_source[col] = le.transform(data_source[col])


# Now X will contain all of the columns except for the class column and Y will contain only the class column.
X, y = data_source.drop("class", axis=1), data_source["class"]

# The X & y variables will now be further moved into training & testing sets. The training set will be processed by the
# Random Forest algorithm to form the model for training. After we have formed the model the program will classify the
# testing set to determine the accuracy of the model.
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)

ranForest = RandomForestClassifier()
ranForest.fit(X_train, y_train)
ranForest_Report = round(ranForest.score(X_train, y_train) * 100, 2)
report = pd.DataFrame({
    'Model': ['Random Forest'],
    'Accuracy': [ranForest_Report],
    'F1 Score': f1_score(y_test, ranForest.predict(X_test))})
print(report)
