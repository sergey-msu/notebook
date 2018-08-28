import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, Imputer, StandardScaler
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from statsmodels.stats.proportion import proportion_confint
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from scipy import stats
from xgboost import XGBClassifier


def title():
    return 'TITANIC DATA SET: example data analysis'


def main():

    # 1. Load data, EDA

    df = pd.read_csv(r'F:\Work\My\Python\examples\titanic_example\titanic3.csv', decimal=',')
    print(df.head())
    print(df.info())
    print(df.dtypes)
    print(df.describe())
    print(df.describe(include=['O']))
    print(df.isnull().sum()/len(df))

    df = df.drop(['name'], axis=1)

    target_column = 'survived'
    num_columns = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'body', 'survived']
    cat_columns = ['sex', 'ticket', 'cabin', 'embarked', 'boat', 'home.dest', 'survived']

    #pclass         int64    0.00
    #survived       int64    0.00
    #name          object    0.00
    #sex           object    0.00
    #age          float64    0.20
    #sibsp          int64    0.00
    #parch          int64    0.00
    #ticket        object    0.00
    #fare         float64    0.00
    #cabin         object    0.77
    #embarked      object    0.00
    #boat          object    0.62
    #body         float64    0.90
    #home.dest     object    0.43


    # 2. Visual Analysis

    ## 2.1 numerical features

    visual_num_df = df[num_columns].drop(['body'], axis=1).dropna()
    sns.pairplot(visual_num_df, hue=target_column)
    plt.show()

    num_corrs = visual_num_df.corr()
    sns.heatmap(num_corrs, square=True, annot=True)
    plt.show()

    ## 2.2 categorical features

    visual_cat_df = df[cat_columns].drop(['cabin', 'home.dest', 'boat'], axis=1).fillna('__none__')

    for column in visual_cat_df.columns:
        enc = LabelEncoder()
        visual_cat_df[column] = enc.fit_transform(visual_cat_df[column])

    sns.pairplot(visual_cat_df, hue=target_column)
    plt.show()

    cat_corrs = visual_cat_df.apply(lambda col: matthews_corrcoef(col.values, visual_cat_df[target_column].values))
    print(cat_corrs)


    # 3. Hypotheses testing

    ## 3.1. Women are most likely to survive

    XW = df[df['sex'] == 'female'][target_column]
    XM = df[df['sex'] == 'male'][target_column]

    print('Woman survived mean:', XW.mean())
    print('Man survived mean:',   XM.mean())

    woman_confint = proportion_confint(XW.sum(), len(XW), alpha=0.05)
    man_confint = proportion_confint(XM.sum(), len(XM), alpha=0.05)

    print('Woman survived proportion confint:', woman_confint)
    print('Man survived proportion confint:', man_confint)

    diff_int = proportions_diff_confint_ind(XW, XM)
    print('Confidence interval for difference:', diff_int)

    p_value = proportions_diff_z_test(XW, XM)
    print('z-test for proportions, p_value:', p_value)

    ## 3.2. passengers with bigger fare are most likely to survive

    XS = df[df[target_column] == 1]['fare'].dropna()
    XD = df[df[target_column] == 0]['fare'].dropna()

    print('Mean fare row survived:', XS.mean())
    print('Mean fare row died:', XD.mean())

    plt.hist([XS, XD], bins=40, stacked=True)
    plt.show()

    print(stats.mannwhitneyu(XS, XD))
    print(stats.ttest_ind(XS, XD))


    # 4. Modeling

    num_columns.remove(target_column)
    cat_columns.remove(target_column)

    ## 4.1 Preprocessing: filter outliers

    clean_df = df.copy()

    q_low  = 0.05
    q_high = 0.95
    q_df = df[num_columns].quantile([q_low, q_high])

    for column in q_df.columns:
        x_low  = q_df.loc[q_low, column]
        x_high = q_df.loc[q_high, column]
        clean_df = clean_df[clean_df[column].isnull() |
                            ((x_low <= clean_df[column]) & (clean_df[column] <= x_high))]

    print(clean_df.shape)

    ## 4.2 Split data

    X = clean_df.drop([target_column], axis=1)
    y = clean_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, shuffle=True, stratify=y,
                                                        random_state=9)

    ## 4.3 Construct pipeline, fitting

    model = XGBClassifier()
    pipeline = get_pipeline(model, num_columns, cat_columns)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=9)
    params = { 'model__n_estimators': [50, 100, 150], 'model__learning_rate': [ 0.01, 0.1, 1 ] }
    grid = GridSearchCV(pipeline, params, cv=cv, scoring='roc_auc', verbose=10)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print(grid.best_params_)
    print(grid.best_score_)

    ## 4.4 Holdout validation

    y_pred = best_model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred)
    print(score)

    return


def proportions_diff_confint_ind(sample1, sample2, alpha = 0.05):
    z = stats.norm.ppf(1 - alpha / 2.)

    p1 = float(sum(sample1)) / len(sample1)
    p2 = float(sum(sample2)) / len(sample2)

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1)/ len(sample1) + p2 * (1 - p2)/ len(sample2))
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1)/ len(sample1) + p2 * (1 - p2)/ len(sample2))

    return (left_boundary, right_boundary)


def proportions_diff_z_test(sample1, sample2, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    n1 = len(sample1)
    n2 = len(sample2)

    p1 = float(sum(sample1)) / n1
    p2 = float(sum(sample2)) / n2
    P = float(p1*n1 + p2*n2) / (n1 + n2)

    z_stat = (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))


    if alternative == 'two-sided':
        return 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        return stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - stats.norm.cdf(z_stat)


class DummyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dummy_columns_ = None

    def fit(self, X, y=None, **kwargs):
        self.dummy_columns_ = pd.get_dummies(X, sparse=True, dummy_na=True).columns.values
        return self

    def transform(self, X, y=None, **kwargs):
        check_is_fitted(self, ['dummy_columns_'])

        dummy_df = pd.get_dummies(X, dummy_na=True)
        res_df = pd.DataFrame()


        for column in self.dummy_columns_:
            res_df[column] = dummy_df[column] if column in dummy_df.columns.values else np.zeros((len(X),), dtype=int)

        return res_df


def get_pipeline(model, num_columns, cat_columns):

    pipeline = Pipeline(steps=[
            ('processing', FeatureUnion([
                ('numeric', Pipeline(steps=[
                    ('selecting', FunctionTransformer(lambda data: data.loc[:, num_columns], validate=False)),
                    ('imputing',  Imputer(strategy='mean')),
                    ('scaling',   StandardScaler())
                    ])),

                ('categorical', Pipeline(steps=[
                    ('selecting', FunctionTransformer(lambda data: data.loc[:, cat_columns], validate=False)),
                    ('encoding',  DummyEncoder())
                    ]))
                ])),

            ('model', model)
        ])

    return pipeline