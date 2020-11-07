#!/usr/bin/env python
# coding: utf-8

# In[15]:


#Time: 13 seconds

#Student name: Samara Kalaoun
#Cohort: 4

################################################################################
#Import Packages
################################################################################

import pandas as pd
import statsmodels.formula.api as smf
import random as rand
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

################################################################################
# Load Data
################################################################################

file = "Apprentice_Chef_Dataset.xlsx"
original_df = pd.read_excel(file)

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# Setting specified seed
rand.seed(a = 222)

# Dropping columns
original_df = original_df.drop(columns=['FAMILY_NAME', 'FIRST_NAME', 'NAME', 'TOTAL_MEALS_ORDERED',
                          'UNIQUE_MEALS_PURCH', 'WEEKLY_PLAN', 'LARGEST_ORDER_SIZE', 'EMAIL'])
#Setting thresholds
CONTACTS_W_CUSTOMER_SERVICE = 10
TOTAL_PHOTOS_VIEWED = 0
AVG_CLICKS_PER_VISIT = 10

# Creating 2 more variables for CONTACTS_W_CUSTOMER_SERVICE, 
# one for when the value is lower or equal to 10 and one for when it is higher

original_df['low_CONTACTS_W_CUSTOMER_SERVICE'] = original_df['CONTACTS_W_CUSTOMER_SERVICE']
condition = original_df.loc[0:,'low_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > 
                                                           CONTACTS_W_CUSTOMER_SERVICE]

original_df['low_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition, value = 0, inplace = True)

original_df['high_CONTACTS_W_CUSTOMER_SERVICE'] = original_df['CONTACTS_W_CUSTOMER_SERVICE']
condition = original_df.loc[0:,'high_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] <= 
                                                            CONTACTS_W_CUSTOMER_SERVICE]

original_df['high_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition, value = 0, inplace = True)

# Creating a dummy variable for TOTAL_PHOTOS_VIEWED (1 --> higher than 0, 0 --> equal to 0)
original_df['dv_TOTAL_PHOTOS_VIEWED'] = 0
condition = original_df.loc[0:,'dv_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED]

original_df['dv_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition, value = 1, inplace = True)

# Creating 2 more variables for AVG_CLICKS_PER_VISIT, 
# one for when the value is lower or equal to 10 and one for when it is higher

original_df['low_AVG_CLICKS_PER_VISIT'] = original_df['AVG_CLICKS_PER_VISIT']
condition = original_df.loc[0:,'low_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] > 
                                                           AVG_CLICKS_PER_VISIT]

original_df['low_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition, value = 0, inplace = True)

original_df['high_AVG_CLICKS_PER_VISIT'] = original_df['AVG_CLICKS_PER_VISIT']
condition = original_df.loc[0:,'high_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] <= 
                                                            AVG_CLICKS_PER_VISIT]

original_df['high_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition, value = 0, inplace = True)


# Dropping response variable
new_df = original_df.drop(columns=['REVENUE'])

################################################################################
# Train/Test Split
################################################################################

# Preparing the response variable
df_target = original_df.loc[:,'REVENUE']

# Preparing training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
            new_df,
            df_target,
            test_size = 0.25,
            random_state = 222)

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# Merging X_train and y_train so that they can be used in statsmodels
df_train = pd.concat([X_train, y_train], axis = 1)

# Building a model
lm_best = smf.ols(formula = """REVENUE ~df_train['CROSS_SELL_SUCCESS'] +
                                df_train['PRODUCT_CATEGORIES_VIEWED'] +
                                df_train['AVG_TIME_PER_SITE_VISIT'] +
                                df_train['MOBILE_NUMBER'] +
                                df_train['CANCELLATIONS_BEFORE_NOON'] +
                                df_train['CANCELLATIONS_AFTER_NOON'] +
                                df_train['TASTES_AND_PREFERENCES'] +
                                df_train['MOBILE_LOGINS'] +
                                df_train['PC_LOGINS'] +
                                df_train['EARLY_DELIVERIES'] +
                                df_train['LATE_DELIVERIES'] +
                                df_train['PACKAGE_LOCKER'] +
                                df_train['REFRIGERATED_LOCKER'] +
                                df_train['FOLLOWED_RECOMMENDATIONS_PCT'] +
                                df_train['AVG_PREP_VID_TIME'] +
                                df_train['TOTAL_PHOTOS_VIEWED'] +
                                df_train['MASTER_CLASSES_ATTENDED'] +
                                df_train['MEDIAN_MEAL_RATING'] +
                                df_train['low_CONTACTS_W_CUSTOMER_SERVICE'] +
                                df_train['high_CONTACTS_W_CUSTOMER_SERVICE'] +
                                df_train['dv_TOTAL_PHOTOS_VIEWED'] +
                                df_train['low_AVG_CLICKS_PER_VISIT'] +
                                df_train['high_AVG_CLICKS_PER_VISIT']""",
                                  data = df_train)

# Fitting the model based on the data
results = lm_best.fit()
print(results.summary())

df_train = df_train.drop(columns = ['CROSS_SELL_SUCCESS', 'PRODUCT_CATEGORIES_VIEWED', 
                                    'MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON',
                                    'CANCELLATIONS_AFTER_NOON', 'TASTES_AND_PREFERENCES',
                                    'MOBILE_LOGINS', 'PC_LOGINS','EARLY_DELIVERIES','LATE_DELIVERIES',
                                    'PACKAGE_LOCKER','REFRIGERATED_LOCKER',
                                    'FOLLOWED_RECOMMENDATIONS_PCT',
                                    'dv_TOTAL_PHOTOS_VIEWED', 'high_AVG_CLICKS_PER_VISIT','AVG_TIME_PER_SITE_VISIT'])

lm_best = smf.ols(formula = """REVENUE ~df_train['AVG_PREP_VID_TIME'] +
                                df_train['TOTAL_PHOTOS_VIEWED'] +
                                df_train['MASTER_CLASSES_ATTENDED'] +
                                df_train['MEDIAN_MEAL_RATING'] +
                                df_train['low_CONTACTS_W_CUSTOMER_SERVICE'] +
                                df_train['high_CONTACTS_W_CUSTOMER_SERVICE'] +
                                df_train['low_AVG_CLICKS_PER_VISIT']""",
                                  data = df_train)

# Fitting the model based on the data
results = lm_best.fit()

# Analysis
print(results.summary())

dropping_columns = ['CROSS_SELL_SUCCESS', 'PRODUCT_CATEGORIES_VIEWED', 
                                    'MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON',
                                    'CANCELLATIONS_AFTER_NOON', 'TASTES_AND_PREFERENCES',
                                    'MOBILE_LOGINS', 'PC_LOGINS','EARLY_DELIVERIES','LATE_DELIVERIES',
                                    'PACKAGE_LOCKER','REFRIGERATED_LOCKER',
                                    'FOLLOWED_RECOMMENDATIONS_PCT',
                                    'dv_TOTAL_PHOTOS_VIEWED', 'high_AVG_CLICKS_PER_VISIT','AVG_TIME_PER_SITE_VISIT']
# INSTANTIATING a model object
lr = LinearRegression()

X_train = X_train.drop(columns = dropping_columns)

y_train = y_train.drop(columns = dropping_columns)

X_test = X_test.drop(columns = dropping_columns)

y_test = y_test.drop(columns = dropping_columns)


################################################################################
# Final Model Score (score)################################################################################

# FITTING to the training data
lr_fit = lr.fit(X_train, y_train)

# PREDICTING on new data
lr_pred = lr_fit.predict(X_test)

# SCORING the results
print('Training Score:', lr.score(X_train, y_train).round(4))
print('Testing Score:',  lr.score(X_test, y_test).round(4))



# In[ ]:





# In[ ]:




