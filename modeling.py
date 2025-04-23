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

from sklearn.tree import DecisionTreeClassifier
def model_decisiontree(x_train, x_test, y_train, y_test, state=42):
    '''Fits decision tree model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    dt = DecisionTreeClassifier(random_state=state)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    return y_pred, dt

from sklearn.ensemble import RandomForestClassifier
def model_randomforest(x_train, x_test, y_train, y_test, state=42):
    '''Fits random forest model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    rf = RandomForestClassifier(random_state=state)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    return y_pred, rf

from sklearn.ensemble import GradientBoostingClassifier
def model_gradientboosting(x_train, x_test, y_train, y_test, state=42):
    '''Fits gradient boosting model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    gb = GradientBoostingClassifier(random_state=state)
    gb.fit(x_train, y_train)
    y_pred = gb.predict(x_test)
    return y_pred, gb

from xgboost.sklearn import XGBClassifier
def model_extremegb(x_train, x_test, y_train, y_test, state=42):
   '''Fits XGBoost model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
   xgb = XGBClassifier(random_state=state)
   xgb.fit(x_train, y_train)
   y_pred = xgb.predict(x_test)
   return y_pred, xgb

from lightgbm import LGBMClassifier
def model_lightgb(x_train, x_test, y_train, y_test, state=42):
   '''Fits LightGBM model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
   lgb = LGBMClassifier(random_state=state, verbose = -1)
   lgb.fit(x_train, y_train)
   y_pred = lgb.predict(x_test)
   return y_pred, lgb

from sklearn.ensemble import ExtraTreesClassifier

def model_extratrees(x_train, x_test, y_train, y_test, state=42):
    '''Fits ExtraTrees model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    et = ExtraTreesClassifier(random_state=state)
    et.fit(x_train, y_train)
    y_pred = et.predict(x_test)
    return y_pred, et

from sklearn.ensemble import AdaBoostClassifier
def model_adaboost(x_train, x_test, y_train, y_test, state=42):
    '''Fits AdaBoost model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    ab = AdaBoostClassifier(random_state=state)
    ab.fit(x_train, y_train)
    y_pred = ab.predict(x_test)
    return y_pred, ab

from sklearn.linear_model import LogisticRegression
def model_logisticregression(x_train, x_test, y_train, y_test, state=42):
    '''Fits Logistic Regression model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    lr = LogisticRegression(random_state=state, solver='saga', max_iter=150)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return y_pred, lr

def model_lassoregularization(x_train, x_test, y_train, y_test, state=42):
    '''Fits L1 Logistic Regression model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    lr1 = LogisticRegression(penalty = 'l1', random_state=state, solver='saga', max_iter=150)
    lr1.fit(x_train, y_train)
    y_pred = lr1.predict(x_test)
    return y_pred, lr1

def model_ridgeRegularization(x_train, x_test, Y_train, Y_test, state=42):
    '''Fits L2 Logistic Regression model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    lr2 = LogisticRegression(penalty = 'l2', random_state=state, solver='saga', max_iter=150)
    lr2.fit(x_train, Y_train)
    y_pred = lr2.predict(x_test)
    return y_pred, lr2

def model_elasticNetRegularization(x_train, x_test, Y_train, Y_test, state=42):
    '''Fits ElasticNet Logistic Regression model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    lrE = LogisticRegression(penalty = 'elasticnet', random_state=state, solver='saga', l1_ratio=0.5, max_iter=150)
    lrE.fit(x_train, Y_train)
    y_pred = lrE.predict(x_test)
    return y_pred, lrE

from sklearn.svm import SVC
def model_linearSupportVector(x_train, x_test, Y_train, Y_test, state=42):
    '''Fits SVM model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    lsv = SVC(random_state=state, probability=True, kernel='linear')
    lsv.fit(x_train, Y_train)
    y_pred = lsv.predict(x_test)
    return y_pred, lsv


def model_nonLinearSupportVector(x_train, x_test, Y_train, Y_test, state=42):
    '''Fits Non-Linear SVM model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    nlsv = SVC(random_state=state, probability=True)
    nlsv.fit(x_train, Y_train)
    y_pred = nlsv.predict(x_test)
    return y_pred, nlsv

# no state param in model function
from sklearn.neighbors import KNeighborsClassifier
def model_kNearestNeighbor(x_train, x_test, Y_train, Y_test):
    '''Fits KNN model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    knn = KNeighborsClassifier()
    knn.fit(x_train, Y_train)
    y_pred = knn.predict(x_test)
    return y_pred, knn

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def model_linearDiscriminantAnalysis(x_train, x_test, Y_train, Y_test):
    '''Fits LDA model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, Y_train)
    y_pred = lda.predict(x_test)
    return y_pred, lda

from sklearn.naive_bayes import GaussianNB
def model_gaussianNaiveBayes(x_train, x_test, Y_train, Y_test):
    '''Fits GNB model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    gnb = GaussianNB()
    gnb.fit(x_train, Y_train)
    y_pred = gnb.predict(x_test)
    return y_pred, gnb

from sklearn.neural_network import MLPClassifier
def model_multiLayerPerceptron(x_train, x_test, Y_train, Y_test, state=42):
    '''Fits MLP model and predicts based on test data.
    Returns tuple of predictions and fitted model'''
    mlp = MLPClassifier(random_state=state)
    mlp.fit(x_train, Y_train)
    y_pred = mlp.predict(x_test)
    return y_pred, mlp

from sklearn.ensemble import VotingClassifier
def ensemble_model(rarefaction_models, clr_models):
    '''Creates Ensemble Model based on best rarefaction models and clr models.
    Currently, this model is not fit, as this was deprioritized,
    however it can easily be fit and used from here.'''
    ensemble = VotingClassifier(estimators=[
        (f'Rarefaction Choice {i}', rare_model) for i, rare_model in enumerate(rarefaction_models)]
        + [(f'CLR Choice {i}', clr_model) for i, clr_model in enumerate(clr_models)], voting='soft')
    # ensemble.fit(x_train, Y_train)
    # y_pred = ensemble.predict(x_test)
    return ensemble