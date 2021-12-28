# Austin Priest 8/11/2021
# CS379 - Machine Learning
# This program is to display the contrast between unsupervised & supervised learning.
# It is going to read the data from the Titanic accident & predict who is most likely to survive & who will most likely not survive.

import os
from collections import OrderedDict
import numpy
import pandas
import seaborn as sns

sns.set()

import matplotlib.pyplot as plot
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

pandas.options.display.width = None
pandas.options.display.max_rows = None
pandas.options.display.max_columns = None
pandas.options.display.max_colwidth = None
pandas.options.display.expand_frame_repr = True
pandas.options.mode.chained_assignment = None


def line(n=80): return '_' * n


def wipe_screen(title=' '):
    _ = os.system('cls')
    if title: print(f'\n{title}')


def clear_input(prompt=' '):
    _ = os.system('cls')
    if prompt: print(f'\n{prompt}')


def pause(n=33):  # n == (Len ('Pres any key to continue))
    print(f'{line(n)}\n')
    os.system('pause')


def message(text, clear=False):
    if clear: wipe_screen()
    print(f'\n{text}')
    pause()


def message_block(text, clear=False):
    if clear: wipe_screen()
    print()
    for line in text:
        print(line)
    pause()


def format_categories(data, columns):
    if len(columns) == 0: columns = list(data)
    for column in columns:
        if not is_numeric_dtype(data[column]):
            category_map, category = {}, 1
            for key in set(data[column].values.tolist()):
                if key not in category_map:
                    category_map[key] = category
                    category += 1
                data[column] = list(map(lambda x: category_map[x], data[column]))


def format_indicators(data, columns):
    if columns:
        for column in columns:
            indicator = pandas.get_dummies(data[column], prefix=column, prefix_sep=' ')
            for column in indicator.columns:
                data[column] = indicator[column]
            data.drop(columns, axis=1, inplace=True)


def prepare(data, indicators, categories, fx=None, pre=True):
    if not data.empty:
        prepare_fx = not fx is None

        if prepare_fx and pre: fx(data)

        if indicators:
            format_indicators(data, indicators)
            if categories:
                format_categories(data, list(set(categories) - set(indicators)))
            else:
                format_categories(data, categories if categories else list(data))

                if prepare_fx and not pre: fx(data)
                data.fillna(0, inplace=True)


def load(datapath, include):
    data = pandas.read_excel(datapath)
    return data[include]


def show(data, plots=True):
    wipe_screen()
    print('\Data.head(10)\n', data.head(20))
    print('\Available Data')
    print(f'{pandas.DataFrame({"count": data.count(), "dtype": data.dtypes})}\n')

    nulls = data.columns[data.isnull().any()]
    print('\nMissing Values')
    print(f'{"(None)" if nulls.empty else data[nulls].isnull().sum()}\n')

    if plots:
        sns.catplot(x="pclass", y="survived", kind='point', data=data).set(ylim=(0, 1))  # .set_title(Survivors x class
        plot.show()

        result = data.groupby(['pclass', 'sex']).size()
        print(result.to_string(), '\n')

        sns.catplot(x="sex", y="survived", hue="pclass", kind="point", data=data)
        plot.show()

        sns.catplot(x="pclass", y="survived", hue="sex", kind="point", data=data).set(ylim=(0, 1)).despine(
            left=True)  # .set_title('Survivors x Class x Gender')
        plot.show()

        ranges = numpy.arange(0, 90, 10)
        data['age_range'] = pandas.cut(data['age'], numpy.arange(0, 90, 10))
        sns.catplot(data=data, x="age_range", y="survived", hue="sex", col='pclass', kind="point")
        plot.show()
        data.drop('age_range', axis=1, inplace=True)
        pause()


'''This is the important part. Use the algorithm to fit the data'''


def test_kmeans_model(model, x, y):  # Unsupervised Model
    def score_km(model, x, y):
        n_correct = 0
        for i in range(len(x)):
            v = numpy.array(x[i].astype(float))
            v = v.reshape(-1, len(v))
            if model.predict(v)[0] == y[i]:
                n_correct += 1
            return n_correct / len(x)
        return score_km(model, x, y)


def fit_kmeans_model(x_train):
    model = KMeans(n_clusters=2, init='k-means++', max_iter=400, n_init=10, random_state=0)
    model.fit(x_train)
    return model


def test_model(model, x, y):
    shuffle_validator = ShuffleSplit(test_size=0.25, random_state=0)
    scores = cross_val_score(model, x, y, cv=shuffle_validator, n_jobs=-1)
    return scores.mean()


def fit_dtree_model(x_train, y_train):  # Supervised Model
    model = tree.DecisionTreeClassifier(max_depth=None)
    model.fit(x_train, y_train)
    return model


def report(results):
    wipe_screen(f'Accuracy Report')
    print(f'{"Indicators":<11} {"Categories":<11} {"KMeans":>10} {"DTree":>10}')
    for (i, c, k, d) in zip(results['Indicators'], results['Categories'], results['KMeans'], results['DTree']):
        print(f'{i:^11} {c:^11} {k:>10.2} {d:>10.2}')
    print()


def execute(x, y, split, results):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split)
    results['KMeans'].append(test_kmeans_model(fit_kmeans_model(x_train), x_test, y_test))
    results['DTree'].append(test_model(fit_dtree_model(x_train, y_train), x_test, y_test))


def process(raw_data, indicators, categories, results, split=0.25, **f):
    results['Indicators'].append('Y' if indicators else '_')
    results['Categories'].append('Y' if categories else '_')
    data = pandas.DataFrame(raw_data)
    prepare(data, indicators, categories, f['x'])
    show(data)
    x, y = f['xy'](data)
    execute(x, y, split, results)


def main(source, include, indicators, categories, **f):
    data = load(source, include)
    show(data, True)
    results = OrderedDict({'Indicators': [], 'Categories': [], 'KMeans': [], 'DTree': []})
    process(data, [],                       [], results, **f)
    process(data, [],               categories, results, **f)
    process(data, indicators,               [], results, **f)
    process(data, ['sex', 'embarked'], ['pclass'], results, **f)
    report(results)


def fx(data):  # Replaces the empty values in the fare & age columns with the mean of the pclass & sex columns.
    def mean(data, column, group): return data.groupby(group)[column].transform(lambda g: g.fillna(g.mean()))
    if not data.empty:
        data['fare'] = mean(data, 'fare', ['pclass'])
        data['age'] = mean(data, 'age', ['sex', 'pclass'])
    else:
        raise Exception('fx(data): No data. ')


def fxy(data):  # Returns the x & the y independent & dependent variables.
    if not data.empty:
        x = data.drop(['survived'], axis=1).astype('float64').values
        y = data['survived'].astype('float64').values
        return x, y
    else:
        raise Exception('fxy(data): No data. ')


source = 'CS379T-Week-1-IP.xls'
include = ['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'fare',
           'embarked']  # Columns to be used by the program.
indicators = ['pclass', 'sex', 'embarked']
categories = ['pclass', 'sex', 'embarked']

main(source, include, indicators, categories, x=fx, xy=fxy, summarize=False, visualize=False)
