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

from xgboost.sklearn import XGBClassifier
def model_extremegb(X_train, X_test, y_train, y_test, state=42):
   xgb = XGBClassifier(random_state=state)
   xgb.fit(X_train, y_train)
   y_pred = xgb.predict(X_test)
   return y_pred, xgb

from lightgbm import LGBMClassifier
def model_lightgb(X_train, X_test, y_train, y_test, state=42):
   lgb = LGBMClassifier(random_state=state, verbose = -1)
   lgb.fit(X_train, y_train)
   y_pred = lgb.predict(X_test)
   return y_pred, lgb

from sklearn.ensemble import ExtraTreesClassifier
def model_extratrees(X_train, X_test, y_train, y_test, state=42):
    et = ExtraTreesClassifier(random_state=state)
    et.fit(X_train, y_train)
    y_pred = et.predict(X_test)
    return y_pred, et

from sklearn.ensemble import AdaBoostClassifier
def model_adaboost(X_train, X_test, y_train, y_test, state=42):
    ab = AdaBoostClassifier(random_state=state)
    ab.fit(X_train, y_train)
    y_pred = ab.predict(X_test)
    return y_pred, ab

from sklearn.linear_model import LogisticRegression
def model_logisticregression(X_train, X_test, y_train, y_test, state=42):
    lr = LogisticRegression(random_state=state, solver='saga')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return y_pred, lr

def model_lassoregularization(X_train, X_test, y_train, y_test, state=42):
    lr1 = LogisticRegression(penalty = 'l1', random_state=state, solver='saga')
    lr1.fit(X_train, y_train)
    y_pred = lr1.predict(X_test)
    return y_pred, lr1

def model_ridgeRegularization(X_train, X_test, Y_train, Y_test, state=42):
    lr2 = LogisticRegression(penalty = 'l2', random_state=state, solver='saga')
    lr2.fit(X_train, Y_train)
    y_pred = lr2.predict(X_test)
    return y_pred, lr2

def model_elasticNetRegularization(X_train, X_test, Y_train, Y_test, state=42):
    # TODO: Edit the l1 ratio for this. ratio = 0 is same is l1. ratio = 1 is same as l2
    lrE = LogisticRegression(penalty = 'elasticnet', random_state=state, solver='saga', l1_ratio=0.5)
    lrE.fit(X_train, Y_train)
    y_pred = lrE.predict(X_test)
    return y_pred, lrE

from sklearn.svm import SVC
def model_linearSupportVector(X_train, X_test, Y_train, Y_test, state=42):
    lsv = SVC(random_state=state, probability=True, kernel='linear')
    lsv.fit(X_train, Y_train)
    y_pred = lsv.predict(X_test)
    return y_pred, lsv


def model_nonLinearSupportVector(X_train, X_test, Y_train, Y_test, state=42):
    nlsv = SVC(random_state=state, probability=True)
    nlsv.fit(X_train, Y_train)
    y_pred = nlsv.predict(X_test)
    return y_pred, nlsv

# no state param in model function
from sklearn.neighbors import KNeighborsClassifier
def model_kNearestNeighbor(X_train, X_test, Y_train, Y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_test)
    return y_pred, knn

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def model_linearDiscriminantAnalysis(X_train, X_test, Y_train, Y_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)
    y_pred = lda.predict(X_test)
    return y_pred, lda

from sklearn.naive_bayes import GaussianNB
def model_gaussianNaiveBayes(X_train, X_test, Y_train, Y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    y_pred = gnb.predict(X_test)
    return y_pred, gnb

from sklearn.neural_network import MLPClassifier
def model_multiLayerPerceptron(X_train, X_test, Y_train, Y_test, state=42):
    mlp = MLPClassifier(random_state=state)
    mlp.fit(X_train, Y_train)
    y_pred = mlp.predict(X_test)
    return y_pred, mlp