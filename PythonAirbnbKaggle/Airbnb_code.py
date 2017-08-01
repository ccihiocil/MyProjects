
# coding: utf-8

# Machine Learning Final Project
# `Airbnb` Bookings
# ===

# # Abstract

# New users on Airbnb can book a place to stay in 34,000+ cities across 190+ countries. By accurately predicting where a new user will book their first travel experience, Airbnb can share more personalized content with their community, decrease the average time to first booking, and better forecast demand.

# ## Key Questions

# - Predict which country a new user’s first booking destination will be. (Supervised)
#     - Constraints on users?
#     - Target Marketing.
# - Predict the time between creating the account and booking (Outlier)

# # Introduction

# ## Why this project?

# Why interesting? How relevant? 
# This project is real-life applicable. 

# ## Data Set 

# - train_users.csv - the training set of users 
# - test_users.csv - the test set of users
# - sessions.csv - web sessions log for users
# - countries.csv - summary statistics of destination countries in this dataset and their locations 
# - age_gender_bkts.csv - summary statistics of users' age group, gender, country of destination 
# - sample_submission.csv - correct format for submitting your predictions

# We have a list of users along with their demographics, web session records, and some summary statistics, a total of 16 variables. The majority of variables are categorical variables. A total of 213466 users in the training set and 62096 in the test set. 

# The training and test sets are split by dates. In the test set, you will predict all the new users with first activities after 7/1/2014 (note: this is updated on 12/5/15 when the competition restarted).

# ## Setups 

# In[11]:

import os
import random
from collections import Counter

import numpy as np
import pandas as pd

# Plotting
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# import category_encoders as ce


# In[12]:

sns.set_context("paper")


# # Data Preparation

# ## Work Flow

# 0. Preprocessing 
#     - Formatting
#     - Cleaning 
# 0. Transformation
#     - Scaling
#     - Decomposition (date, time)
# 0. Feature Selection

# some question I may be insterested in asking:
# (2, 3 questions are more interesting to answer)       
# 
# 1. predict the first destination of a new user. 
# 2. predict the time between creating the account and booking 
# 3. look into what signup method I want to invest more money in? (What signup method is the most popular or has the highest rate of booking a hotel) 

# ## Helper Functions

# In[185]:

def get_summary(df, all=True):
    get_shape(df)
    print('\nData Types')
    get_dtypes(df)
    print('\nNA Values')
    get_NAs(df, all=all)


# In[186]:

def get_shape(df):
    num_obs, num_features = df.shape
    print('There are {} observations with {} features'
          .format(num_obs, num_features))


# In[15]:

def get_dtypes(df):
    _, num_features = df.shape
    dtype_counts = Counter(df.dtypes).most_common()
    print(df.dtypes, '\n')
    for dtype, counts in dtype_counts:
        print('{}: {} {:.2f}%'
              .format(dtype, counts, counts / num_features * 100))


# In[16]:

def get_NAs(df, all=True):
    num_obs, _ = df.shape
    for feature, num_na in df.isnull().sum().items():
        if all:
            print("{}: {} {:.2f}%"
                  .format(feature, num_na, num_na / num_obs * 100))
        if num_na > 0:
            print("{}: {} {:.2f}%"
                  .format(feature, num_na, num_na / num_obs * 100))


# In[17]:

def get_categorical(df):
    return [feature for feature, dtype in df.dtypes.items()
            if dtype == 'O']


# In[18]:

def get_numerical(df):
    return [feature for feature, dtype in df.dtypes.items()
            if feature not in get_categorical(df)]


# ## Data Cleaning

# Load the data into DataFrames

# In[19]:

get_ipython().magic('ls data')


# In[20]:

files = [f for f in os.listdir('./data/') if f.endswith('csv')]
files


# In[21]:

na_values = ['-unknown-', ' ']


# ## Users

# In[22]:

users_train_file, users_test_file = files[4], files[3]


# In[23]:

users = pd.read_csv('./data/{}'.format(users_train_file),
                    na_values=na_values)


# In[24]:

users.sample(10)


# In[25]:

get_summary(users)


# > 58.35% of the users have never booked. Let's explore why.

# In[26]:

get_numerical(users)


# In[27]:

get_categorical(users)


# Some of these are not categorical, let's fix that.

# ### Cleaning DateTime

# 0. change data types to datetime
# 0. created, first booking - extract month
# 0. time difference in days between created and first booking
# 0. first active - extract hour of the day

# In[28]:

users_cols_date = [feature for feature in users.keys()
                   if 'date' in feature]
users_cols_time = [feature for feature in users.keys()
                   if 'time' in feature]
users_cols_dt = users_cols_date + users_cols_time


# We also record the columns after change

# In[29]:

users_cols_dt_after = list(users_cols_dt)


# In[30]:

users[users_cols_dt].sample(5)


# #### change data types to datetime

# In[31]:

users[users_cols_date].sample(5)


# In[32]:

users[users_cols_date].dtypes


# In[33]:

for col in users_cols_date:
    users[col] = pd.to_datetime(users[col])


# In[34]:

users[users_cols_date].dtypes


# In[35]:

users[users_cols_time].head(5)


# In[36]:

users[users_cols_time].dtypes


# In[37]:

for col in users_cols_time:
    users[col] = pd.to_datetime(users[col], format="%Y%m%d%H%M%S")


# In[38]:

users[users_cols_time].dtypes


# #### created, first booking - extract month

# In[39]:

users[users_cols_dt].sample(5)


# In[40]:

for col in users_cols_dt:
    users['{}_month'.format(col)] = users[col].apply(lambda dt: dt.month)
    users_cols_dt_after += ['{}_month'.format(col)]


# In[41]:

users_cols_dt_month = [col + '_month' for col in users_cols_dt]


# In[42]:

users[users_cols_dt_month].sample(5)


# #### time difference in days between created and first booking

# In[43]:

users_cols_date


# In[44]:

users['time_delta_bc'] = users[users_cols_date[1]] - users[users_cols_date[0]]
users['time_delta_bc'] = (
    users['time_delta_bc'].dropna() / np.timedelta64(1, 'D')).astype(int)
users_cols_dt_after += ['time_delta_bc']


# In[45]:

users[['time_delta_bc']].sample(5)


# #### first active hour of the day

# In[46]:

users[users_cols_time].sample(5)


# In[47]:

for col in users_cols_time:
    users['{}_hour'.format(col)] = users[col].apply(lambda dt: dt.hour)
    users_cols_dt_after += ['{}_hour'.format(col)]


# In[48]:

users_cols_time_hour = [col + '_hour' for col in users_cols_time]


# In[49]:

users[users_cols_time_hour].sample(5)


# ### Cleaning `age`

# In[50]:

users['age'].describe()


# In[51]:

sns.distplot(users['age'].dropna())


# In[52]:

users.shape


# In[53]:

users = users[users['age'].fillna(0) < 100]


# In[54]:

users.shape


# In[55]:

users[['age']].sample(10)


# In[56]:

sns.distplot(users['age'].dropna())


# ### Convert Categorical Data

# In[57]:

users.drop(users_cols_dt_after, axis=1).sample(5)


# In[58]:

users_cols_categorical = get_categorical(users)
users_cols_categorical


# We don't need id since that's just randomly generated index

# In[59]:

users_cols_categorical.remove('id')
users_cols_categorical


# Some of the datetime (hour and month) can also be categorical (ordinal)

# In[60]:

users_cols_dt_categorical = [col for col in users_cols_dt_after
                             if 'month' in col or 'hour' in col]
users_cols_dt_categorical


# In[61]:

users_cols_categorical += users_cols_dt_categorical


# In[62]:

users_cols_categorical


# In[63]:

for col in users_cols_categorical:
    users[col] = users[col].astype('category')


# In[64]:

get_dtypes(users)


# ### Fill NAs

# In[65]:

get_NAs(users, all=False)


# We will try out different methods for dealing with NAs

# #### drop all NAs

# Conditions:
# 1. Have sufficient data points, so the model doesn't lose power.
# 2. Not to introduce bias (meaning, disproportionate or non-representation of classes).

# In[66]:

users_dropna = users.dropna().reset_index(drop=True)


# In[67]:

get_summary(users_dropna, all=False)


# In[68]:

users['gender'].value_counts().plot(kind='bar')


# In[69]:

users_dropna['gender'].value_counts().plot(kind='bar')


# In[70]:

sns.distplot(users['age'].dropna())


# In[71]:

sns.distplot(users_dropna['age'])


# # Machine Learning

# ## Helper Functions

# In[187]:

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


# In[188]:

def encode_features(features, cols):
    """Encode all nominal features to oridinal"""
    features_encoded = features.copy(deep=True)
    feature_encoding_map = {col: LabelEncoder() for col in cols}

    for col in cols:
        features_encoded[col] = feature_encoding_map[col].fit_transform(
            features_encoded[col])

    return features_encoded, feature_encoding_map


# In[189]:

def eval_on_data(features, target, model, seed=523):
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        random_state=seed,
                                                        test_size=.15)
    model.fit(X_train, y_train)
    print("Train Score with all features: {:.3f}%"
          .format(model.score(X_train, y_train) * 100))
    print("Test Score with all features: {:.3f}%"
          .format(model.score(X_test, y_test) * 100))


# Select feature by best K features univariate regressions

# In[190]:

def select_features(features, target, clf, selector=SelectKBest, verbose=False, max_k=None):
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        random_state=523,
                                                        test_size=.15)
    num_features = X_train.shape[1]
    max_k = max_k if max_k else num_features
    K = [i for i in range(1, max_k + 1)]
    scores_train, scores_test = [], []
    for k in K:
        select = selector(k=k)
        select.fit(X_train, y_train)
        X_train_selected = select.transform(X_train)
        X_test_selected = select.transform(X_test)

        # Fit Classifier with training data
        clf.fit(X_train_selected, y_train)
        print("k={}".format(k))

        # Show training score
        score_train = clf.score(X_train_selected, y_train)
        scores_train += [score_train]
        print("Train Score with all features: {:.3f}%"
              .format(score_train * 100))

        # Show test score
        score_test = clf.score(X_test_selected, y_test)
        scores_test += [score_test]
        print("Test Score with all features: {:.3f}%"
              .format(score_test * 100))

        # Show selected features
        mask = select.get_support()
        if verbose:
            # visualize the mask -- black is True, white is False
            plt.matshow(mask.reshape(1, -1), cmap='gray_r')
            plt.xlabel('best {} features'.format(k))
        print([c for c, s in zip(features.columns, mask) if s])
        print('\n')
    plt.plot(K, scores_train, label='training score')
    plt.plot(K, scores_test, label='test score')
    plt.legend()
    return K, scores_train, scores_test


# In[191]:

def select_features_perc(features, target, clf, max_perc=50, inc=5, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        random_state=523,
                                                        test_size=.15)

    percs = list(range(1, max_perc + 1, inc))
    scores_train, scores_test = [], []
    for perc in percs:
        select = SelectPercentile(percentile=perc)
        select.fit(X_train, y_train)
        X_train_selected = select.transform(X_train)
        X_test_selected = select.transform(X_test)

        # Fit Classifier with training data
        clf.fit(X_train_selected, y_train)
        print("k={}".format(k))

        # Show training score
        score_train = clf.score(X_train_selected, y_train)
        scores_train += [score_train]
        print("Train Score with all features: {:.3f}%"
              .format(score_train * 100))

        # Show test score
        score_test = clf.score(X_test_selected, y_test)
        scores_test += [score_test]
        print("Test Score with all features: {:.3f}%"
              .format(score_test * 100))

        # Show selected features
        mask = select.get_support()
        if verbose:
            # visualize the mask -- black is True, white is False
            plt.matshow(mask.reshape(1, -1), cmap='gray_r')
            plt.xlabel('best {} features'.format(perc))
        print([c for c, s in zip(features.columns, mask) if s])
        print('\n')
    plt.plot(percs, scores_train, label='training score')
    plt.plot(percs, scores_test, label='test score')
    plt.legend()
    return percs, scores_train, scores_test


# In[192]:

def select_features_model(features, target, clf, threshold='median', verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        random_state=523,
                                                        test_size=.15)
    select = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=523),threshold=threshold)
    select.fit(X_train, y_train)
    X_train_selected = select.transform(X_train)
    X_test_selected = select.transform(X_test)

    # Fit Classifier with training data
    clf.fit(X_train_selected, y_train)

    # Show training score
    score_train = clf.score(X_train_selected, y_train)
    print("Train Score with all features: {:.3f}%"
          .format(score_train * 100))

    # Show test score
    score_test = clf.score(X_test_selected, y_test)
    print("Test Score with all features: {:.3f}%"
          .format(score_test * 100))

    # Show selected features
    mask = select.get_support()
    if verbose:
        # visualize the mask -- black is True, white is False
        plt.matshow(mask.reshape(1, -1), cmap='gray_r')
        plt.xlabel('best {} features'.format(perc))
        print([c for c, s in zip(features.columns, mask) if s])
        print('\n')
    return select


# In[193]:

def select_features_RFE(features, target, clf, numX = 3,verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        random_state=523,
                                                        test_size=.15)
    select = RFE(
        RandomForestClassifier(n_estimators=100, random_state=523),
        n_features_to_select=numX)
    select.fit(X_train, y_train)
    X_train_selected = select.transform(X_train)
    X_test_selected = select.transform(X_test)

    # Fit Classifier with training data
    clf.fit(X_train_selected, y_train)

    # Show training score
    score_train = clf.score(X_train_selected, y_train)
    print("Train Score with all features: {:.3f}%"
          .format(score_train * 100))

    # Show test score
    score_test = clf.score(X_test_selected, y_test)
    print("Test Score with all features: {:.3f}%"
          .format(score_test * 100))

    # Show selected features
    mask = select.get_support()
    if verbose:
        # visualize the mask -- black is True, white is False
        plt.matshow(mask.reshape(1, -1), cmap='gray_r')
        plt.xlabel('best {} features'.format(perc))
        print([c for c, s in zip(features.columns, mask) if s])
        print('\n')
    return select


# In[79]:

def eval_selected_feature(features, target, clf, k):
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        random_state=523,
                                                        test_size=.15)
    select = SelectKBest(k=k)
    select.fit(X_train, y_train)
    X_train_selected = select.transform(X_train)
    X_test_selected = select.transform(X_test)

    # Fit Classifier with training data
    clf.fit(X_train_selected, y_train)
    print("k={}".format(k))

    # Show training score
    score_train = clf.score(X_train_selected, y_train)
    print("Train Score with all features: {:.3f}%"
          .format(score_train * 100))

    # Show test score
    score_test = clf.score(X_test_selected, y_test)
    print("Test Score with all features: {:.3f}%"
          .format(score_test * 100))

    # Show selected features
    mask = select.get_support()

    # visualize the mask -- black is True, white is False
    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.xlabel('best {} features'.format(k))
    selected_features = [c for c, s in zip(features.columns, mask) if s]
    print(selected_features)
    print('\n')
    return score_train, score_test, selected_features


# In[80]:

def print_result(train_R2, test_R2, support_vector):
    print('Final train R2: {:.4f}'.format(train_R2))
    print('Final test R2: {:.4f}'.format(test_R2))
    print('Final select_features:')
    for feature in support_vector:
        print(feature)


# ## Preprocessing

# ### Drop Useless Features

# In[81]:

users_dropna.sample(5)


# In[82]:

cols = list(users.columns)
cols


# In[83]:

cols_to_drop = ['id', 'date_account_created',
                'timestamp_first_active', 'date_first_booking']


# In[84]:

users_dropna = users_dropna.drop(cols_to_drop, axis=1)


# In[85]:

users_dropna.sample(5)


# ## Supervised Learning (Regression)

# Predict the time between creating the account and booking

# In[86]:

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


# - Linear Regression
# - Ridge Regression
# - Lasso Regression
# - Decision Tree Regressor
# - Support Vector Machine Regressor

# ### features, target split

# In[74]:

users_dropna.sample(5)


# In[91]:

target_name_rgs = ['time_delta_bc']


# In[92]:

cols_to_drop_rgs = target_name_rgs + ['date_first_booking_month']


# In[93]:

features_rgs, target_rgs = users_dropna.drop(
    cols_to_drop_rgs, axis=1), users_dropna[target_name_rgs]


# In[94]:

features_rgs.sample(5)


# In[95]:

target_rgs.sample(5)


# ### Encoding Categorical Data

# In[96]:

features_rgs.dtypes


# In[97]:

features_num_rgs = ['age', 'signup_flow']
features_num_rgs


# In[98]:

features_cat_rgs = list(features_rgs.columns.difference(features_num_rgs))
features_cat_rgs


# In[153]:

features_rgs, fe_cat_map_rgs = encode_features(features_rgs, features_cat_rgs)


# In[154]:

features_rgs.sample(5)


# In[155]:

fe_cat_map_rgs


# ### Feature Scaling

# In[156]:

def scale_features(features, cols=None):
    """Scale selected features to oridinal"""
    features_scaled = features.copy(deep=True)

    cols = cols if cols else list(features.columns)

    feature_scaling_map = {col: StandardScaler() for col in cols}

    for col in cols:
        features_scaled[col] = feature_scaling_map[col].fit_transform(
            features_scaled[col])

    return features_scaled, feature_scaling_map


# In[157]:

features_rgs, fs_map_rgr = scale_features(features_rgs)


# In[158]:

features_rgs.sample(5)


# In[159]:

fs_map_rgr


# ### flatten target dimensions

# In[160]:

target_rgs.sample(5)


# In[161]:

target_rgs = np.ravel(target_rgs)
target_rgs


# ### Linear Regression

# In[162]:

lin = LinearRegression()
eval_on_data(features_rgs, target_rgs, lin)


# ### Lasso

# In[94]:

lasso = Lasso()
eval_on_data(features_rgs, target_rgs, lasso)


# In[95]:

for alpha in [0.2, 0.4, 0.6, 0.8, 1]:
    print('alpha {}'.format(alpha))
    lasso = Lasso(alpha=alpha)
    select_features(features_rgs, target_rgs, lasso)


# ### SVM

# In[96]:

# svr = SVR()
# select_features(features_rgs, target_rgs, svr, max_k=3)


# ### Decision Tree

# #### tuning max depth

# In[97]:

max_test_scores = dict()
for max_depth in range(2, 16 + 1):
    print('max depth: {}'.format(max_depth))
    dtr = DecisionTreeRegressor(max_depth=max_depth)
    k, _, scores_test = select_features(features_rgs, target_rgs, dtr)
    max_test_scores[max_depth] = max(
        list(zip(scores_test, k)), key=lambda sk: sk[0])


# In[98]:

max_test_scores


# #### tuning max leaf nodes

# In[99]:

max_test_scores = dict()
for min_samples_leaf in range(38, 48, 1):
    print('min samples leaf: {}'.format(min_samples_leaf))
    dtr = DecisionTreeRegressor(
        max_depth=11, min_samples_leaf=min_samples_leaf)
    k, _, scores_test = select_features(features_rgs, target_rgs, dtr)
    max_test_scores[min_samples_leaf] = max(
        list(zip(scores_test, k)), key=lambda sk: sk[0])


# In[100]:

max_test_scores


# ## Supervised Learning (Classification) Time Diff 

# #### transform the time_delta_bc to a categorical data 

# In[183]:

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd 


# In[163]:

users_dropna['time_delta_bc'].describe()


# In[164]:

def group(c):
  if c['time_delta_bc'] < 1:
    return '0'
  elif 1 <= c['time_delta_bc'] < 4:
    return '1'
  elif 4 <= c['time_delta_bc'] < 46:
    return '2'
  else:
    return '3'


# In[165]:

users_dropna['time_delta_c'] = users_dropna.apply(group, axis=1)


# In[166]:

users_dropna.dtypes


# In[167]:

# convert it to categorical variable 
users_dropna['time_delta_c'] = users_dropna['time_delta_c'].astype('category')


# ### prepare

# In[168]:

target_name_clf = ['time_delta_c'] 


# In[169]:

# drop the useless cols 
cols_to_drop_clf = ['time_delta_bc',
                    'date_first_booking_month',
                    'country_destination'] + target_name_clf
cols_to_drop_clf


# In[170]:

features_clf, target_clf = users_dropna.drop(
    cols_to_drop_clf, axis=1), users_dropna[target_name_clf]


# In[171]:

# numeric variables 
features_num_clf = ['age', 'signup_flow']
features_num_clf


# In[172]:

# what categorical variables I used 
features_cat_clf = list(features_clf.columns.difference(features_num_clf))
features_cat_clf


# In[173]:

features_clf, fe_cat_map_clf = encode_features(features_clf, features_cat_clf)


# In[174]:

# convert targers  
target_clf, te_cat_map_clf = encode_features(target_clf, target_name_clf)


# In[175]:

te_cat_map_clf


# In[176]:

target_clf = np.ravel(target_clf)
target_clf


# In[177]:

# feature scaling 
features_clf, fs_map_clf = scale_features(features_clf)


# In[178]:

features_clf.sample(5)


# In[179]:

fs_map_clf


# ### Decision Trees 

# In[180]:

dtc = DecisionTreeClassifier(random_state=523)
select_features(features_clf, target_clf, dtc)


# In[196]:

select_features_model(features_clf, target_clf, dtc, selector=RFE)


# In[197]:

select_features_model(features_clf, target_clf, dtc, selector=RFE, numX=2)


# ### Logistic Regression 

# In[198]:

lrc = LogisticRegression(random_state=523)
select_features(features_clf, target_clf, lrc)


# ### SVM

# In[199]:

svc = SVC()
select_features(features_clf, target_clf, svc, max_k=3)


# ### Random Forest 

# In[200]:

rfc = RandomForestClassifier()
select_features(features_clf, target_clf, rfc)


# ### Boosting 

# In[201]:

adbc = AdaBoostClassifier(DecisionTreeClassifier())
select_features(features_clf, target_clf, adbc)


# ## Supervised Learning (Classification)

# Predict which country a new user's first booking destination will be. 

# - Decision Tree
# - Logistic regression 
# - SVM 
# - Ensemble
#     - Boosting
#     - Random Forest

# In[102]:

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ### features, target split

# We need to get rid of anything related to first booking. Since we would know the country destination had we known the first booking.

# In[136]:

users_dropna.sample(5)


# In[137]:

users_dropna.columns


# In[138]:

target_name_clf = ['country_destination']


# In[139]:

cols_to_drop_clf = ['time_delta_bc',
                    'date_first_booking_month'] + target_name_clf
cols_to_drop_clf


# In[140]:

features_clf, target_clf = users_dropna.drop(
    cols_to_drop_clf, axis=1), users_dropna[target_name_clf]


# In[141]:

features_clf.sample(5)


# In[142]:

target_clf.sample(5)


# ### Encoding Categorical Data

# #### Convert Features

# In[143]:

features_clf.dtypes


# In[144]:

features_num_clf = ['age', 'signup_flow']
features_num_clf


# In[145]:

features_cat_clf = list(features_clf.columns.difference(features_num_clf))
features_cat_clf


# In[146]:

features_clf, fe_cat_map_clf = encode_features(features_clf, features_cat_clf)


# In[148]:

features_clf.sample(5)


# In[147]:

fe_cat_map_clf


# #### Convert Targets

# In[149]:

target_clf, te_cat_map_clf = encode_features(target_clf, target_name_clf)


# In[150]:

target_clf.sample(5)


# In[151]:

te_cat_map_clf


# In[153]:

target_clf = np.ravel(target_clf)
target_clf


# ### Feature Scaling

# In[155]:

features_clf.sample(5)


# In[152]:

features_clf, fs_map_clf = scale_features(features_clf)


# In[157]:

features_clf.sample(5)


# for classification

# In[158]:

fs_map_clf


# ### Decision Trees

# In[ ]:

dtc = DecisionTreeClassifier(random_state=523)
select_features(features_clf, target_clf, dtc)


# In[175]:

select_features_perc(features_clf, target_clf, dtc, max_perc=35, inc=5)


# In[194]:

select_features_model(features_clf, target_clf, dtc, selector=RFE)


# In[193]:

select_features_model(features_clf, target_clf, dtc, selector=RFE, numX=2)


# In[192]:

select_features_model(features_clf, target_clf, dtc, selector=RFE, numX=4)


# ### Logistic Regression

# In[160]:

lrc = LogisticRegression(random_state=523)
select_features(features_clf, target_clf, lrc)


# ### SVM

# In[162]:

svc = SVC()
select_features(features_clf, target_clf, svc, max_k=3)


# ### Random Forest

# In[161]:

rfc = RandomForestClassifier()
select_features(features_clf, target_clf, rfc)


# ### Boosting

# In[163]:

adbc = AdaBoostClassifier(DecisionTreeClassifier())
select_features(features_clf, target_clf, adbc)


# # Reflection

# ## Conclusion

# ## Issues
