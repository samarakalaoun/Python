# timeit: 55 seconds

# Student Name : Samara Kalaoun
# Cohort       : 4


################################################################################
# Import Packages
################################################################################

import pandas as pd
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix        
from sklearn.metrics import roc_auc_score           
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn.neighbors import KNeighborsRegressor   
from sklearn.tree import export_graphviz            
from IPython.display import Image                   
import pydotplus                                    


################################################################################
# Load Data
################################################################################

file = "Apprentice_Chef_Dataset.xlsx"
original_df = pd.read_excel(file)


################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# Creating an empty list
placeholder_lst = []

# looping over each email address
x = 0
for index in original_df.iterrows():
    # splitting email domain at '@'
    split_email = original_df.loc[x, 'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    x = x+1
# Converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)

# Testing if all other emails than junk and personal are professional
non_pro = ['me.com', 'aol.com', 'hotmail.com', 'live.com', 'msn.com', 'passport.com',
          'gmail.com', 'yahoo.com', 'protonmail.com']

# Replacing it for an empty list
placeholder_lst = []

# looping over the dataframe created in the previous loop
x = 0
for index in email_df.iterrows():
    if email_df.iloc[x,1] in non_pro:
        placeholder_lst.append('non_pro')
    else:
        placeholder_lst.append(email_df.iloc[x,1])
    x = x+1

# Converting placeholder_lst into a DataFrame    
email_type_df = pd.DataFrame(placeholder_lst)

# Creating a new column for non professional emails (1 --> non professional, 0 --> professional)
x = 0
for index in email_df.iterrows():
    if email_df.loc[x,1] in non_pro:
        original_df.loc[x,'NON_PRO']=1
    else:
        original_df.loc[x,'NON_PRO']=0
    x = x+1

# Creating a new column for professional emails (1 --> professional, 0 --> either personal or junk)
x = 0
for index in email_df.iterrows():
    if email_df.loc[x,1] in non_pro:
        original_df.loc[x,'PRO_EMAIL']=0
    else:
        original_df.loc[x,'PRO_EMAIL']=1
    x = x+1

# Drop email column
original_df = original_df.drop(columns=['EMAIL'])

# Dropping columns related to name
original_df = original_df.drop(columns=['FAMILY_NAME', 'FIRST_NAME', 'NAME', 'REVENUE', 'LARGEST_ORDER_SIZE', 'MEDIAN_MEAL_RATING',
                                        'NON_PRO', 'TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH','CONTACTS_W_CUSTOMER_SERVICE', 
                                         'PRODUCT_CATEGORIES_VIEWED', 'AVG_TIME_PER_SITE_VISIT', 
                                         'CANCELLATIONS_AFTER_NOON', 'TASTES_AND_PREFERENCES', 'PC_LOGINS','WEEKLY_PLAN', 'MOBILE_LOGINS',
                                         'EARLY_DELIVERIES', 'LATE_DELIVERIES', 'PACKAGE_LOCKER', 'REFRIGERATED_LOCKER',
                                         'AVG_PREP_VID_TIME', 'MASTER_CLASSES_ATTENDED', 'AVG_CLICKS_PER_VISIT', 'TOTAL_PHOTOS_VIEWED'])



################################################################################
# Train/Test Split
################################################################################

# Dropping the response variable
new_df = original_df.drop (columns='CROSS_SELL_SUCCESS')

# Preparing the response variable
df_target = original_df.loc[:,'CROSS_SELL_SUCCESS']

# Preparing training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
            new_df,
            df_target,
            test_size = 0.25,
            random_state = 222)

# Merging X_train and y_train so that they can be used in statsmodels
df_train = pd.concat([X_train, y_train], axis = 1)


################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

def display_tree(tree, feature_df, height = 1500, width = 800):
    
    # visualizing the tree
    dot_data = StringIO()

    # exporting tree to graphviz
    export_graphviz(decision_tree      = tree,
                    out_file           = dot_data,
                    filled             = True,
                    rounded            = True,
                    special_characters = True,
                    max_depth          = 5,
                    feature_names      = feature_df.columns)


    # declaring a graph object
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


    # creating image
    img = Image(graph.create_png(),
                height = height,
                width  = width)
    print(img)
    return img

# INSTANTIATING a classification tree object
tree_pruned      = DecisionTreeClassifier(max_depth = 4,
                                          min_samples_leaf = 25,
                                          random_state = 802)


# FITTING the training data
tree_pruned_fit  = tree_pruned.fit(X_train, y_train)


# PREDICTING on new data
tree_pred = tree_pruned_fit.predict(X_test)



# calling display_tree
display_tree(tree       = tree_pruned_fit,
             feature_df = X_train)    



################################################################################
# Final Model Score (score)
################################################################################

training_score = tree_pruned_fit.score(X_train, y_train).round(4)
test_score = tree_pruned_fit.score(X_test, y_test).round(4)
AUC_score = roc_auc_score(y_true  = y_test, y_score = tree_pred).round(4)

# SCORING the model
print('Training ACCURACY:', training_score)
print('Testing  ACCURACY:', test_score)
print('AUC Score        :', AUC_score)


