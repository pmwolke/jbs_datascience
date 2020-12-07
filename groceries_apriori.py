# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:29:14 2020

@author: pwolke
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from apyori import apriori
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source


##### DATA GATHERING AND CLEANING #####


# Get dataset
groceries = pd.read_csv("Groceries_dataset.csv")

# Get item list
products = groceries["itemDescription"].unique()

# Clean data
one_hot = pd.get_dummies(groceries["itemDescription"])
groceries.drop("itemDescription", inplace = True, axis=1)
groceries = groceries.join(one_hot)

# Get new table with unique transactions
records = groceries.groupby(["Member_number","Date"])[products[:]].sum()
records = records.reset_index()[products]


##### DECISION TREE MODELING #####


# Declare product in basket we want to try and predict
product_to_test = "whole milk"

# Set up y (product to test) and X (all other products) for classification
index = np.argwhere(products==product_to_test)
features = np.delete(products, index)
X = records[features]
y = records[product_to_test]

# Creat hyperparamter grid
range_to_test = list(range(2,4))
param_grid = {"max_leaf_nodes": range_to_test, "min_samples_split": [2,3,4],
              "random_state": [42]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=42)

# Create Decision Tree Classifer
dtree = DecisionTreeClassifier()

# Create GridSearchCV object
dtree_cv = GridSearchCV(dtree, param_grid, cv=3)

# Fit the training data
dtree_cv.fit(X_train, y_train)

# Print the optimal paramters and scores
print("Tuned Decision Tree Parameters: {}".format(dtree_cv.best_params_))
print("Tuned Decision Tree Accuracy: {}".format(dtree_cv.best_score_))

# Print decision tree with best parameters and score to png file
graph = Source(export_graphviz(dtree_cv.best_estimator_, out_file=None,
                feature_names=None, class_names=None, filled=True))
graph.format = "png"
graph.render("dtree_render", view=True)


##### APRIORI MODELING #####

# Create function to convert int to string
def get_product_names(x):
    for product in products:
        if x[product] != 0:
            x[product] = product
    return x

# Applt function to "clean" data for aprioiri modeling
records = records.apply(get_product_names, axis=1)
print(records.head())
print(f"Total transactions: {len(records)}")

# Create list of words to associate by row; call it transactions
x = records.values
x = [sub[~(sub == 0)].tolist() for sub in x if sub[sub != 0].tolist()]
transactions = x

# Call apriori model on transactions with parameters
association_rules = apriori(transactions,min_support=0.00030,
                            min_confidance=0.01, min_lift=1.5, min_length=2,
                            target="rules")

# Save results of apriori model
association_results = list(association_rules)
print(association_results[0])

# Setup output dataframe; so we can export to csv file if needed
output_columns = ["Item A", "Item B", "Support", "Confidence", "Lift"]
output = pd.DataFrame(columns=output_columns)

# Look at relavent associations for beef, pork, and chicken 
for item in association_results:

    # Set dummy variables for reuse
    pair = item[0] 
    items = [x for x in pair]
    
    # Only pick the transaction if beef, pork, or chicken is present
    if (items[0] == "beef" or items[0] == "pork" or items[0] == "chicken" or
        items[1] == "beef" or items[1] == "pork" or items[1] == "chicken"):
        
        # Create new row in dataframe and append
        new_row = {output_columns[0]:items[0], output_columns[1]:items[1],
                    output_columns[2]:item[1], output_columns[3]:item[2][0][2],
                    output_columns[4]:item[2][0][3]}
        output = output.append(new_row, ignore_index=True)
        
        # Print association rule with metrics support, confidence, and lift
        print("Rule : ", items[0], " -> " + items[1])
        print("Support : ", str(item[1]))
        print("Confidence : ",str(item[2][0][2]))
        print("Lift : ", str(item[2][0][3]))
        
        print("=====================================")

# Create conditions for deciding if an association contains beef, pork, or chicken
conditions = [(output["Item A"] == "beef") | (output["Item B"] == "beef"),
              (output["Item A"] == "pork") | (output["Item B"] == "pork"),
              (output["Item A"] == "chicken") | (output["Item B"] == "chicken")]

# Apply correct values to each condition
values = ["beef", "pork", "chicken"]

# Create new column in output dataframe with protein type within each assocation
output["protein_type"] = np.select(conditions, values)
   
# Setup scatter plot to compare confidence and lift by protein_type     
g = sns.relplot(x="Confidence", y="Lift", data=output,
            hue="protein_type", col="protein_type",
            palette=["b", "r", "g"])

# Set plot title and show
g.fig.suptitle("Confidence versus lift", y=2)
plt.show()