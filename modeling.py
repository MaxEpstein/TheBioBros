# This file is for the modeling (and maybe hyperparameter tuning) component of the pipeline.

# Models:
    # 1 Decision Tree DT
    # 2 Random Forest RF
    # 3 Gradient Boosting GB
    # 4 Extreme Gradient Boosting XGB
    # 5 Light Gradient Boosting Model LightGBM
    # 6 Extremely Randomized Trees ExtraTrees
    # 7 Adaptive Boosting AdaBoost
    # 8 Logistic Regression LR
    # 9 Logistic Regression - Lasso Regularization LR-L1
    # 10 Logistic Regression - Ridge Regularization LR-L2
    # 11 Logistic Regression - Elastic Net Regularization LR-ENet
    # 12 Linear Support Vector Machine Linear SVM
    # 13 Non-linear Support Vector Machine Non-linear SVM
    # 14 K-Nearest Neighbors KNN
    # 15 Linear Discriminant Analysis LDA
    # 16 Gaussian Naive Bayes GNB
    # 17 Multilayer Perceptron MLP

# 100 random states (seeds); Keep track of them!

from sklearn.tree import DecisionTreeClassifier
def model_decisiontree(X_train, X_test, y_train, y_test, state=42):
    dt = DecisionTreeClassifier(random_state=state)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return y_pred, dt

from sklearn.ensemble import RandomForestClassifier
def model_randomforest(X_train, X_test, y_train, y_test, state=42):
    rf = RandomForestClassifier(random_state=state)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return y_pred, rf

from sklearn.ensemble import GradientBoostingClassifier
def model_gradientboosting(X_train, X_test, y_train, y_test, state=42):
    gb = GradientBoostingClassifier(random_state=state)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    return y_pred, gb