import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.ensemble import  RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, KFold
from boruta import BorutaPy

def model_fit_score(model, X_train, X_test, y_train, y_test):
    """
    returns log loss, accuracy and roc_auc_score

    inputs
    ------
    model: model with chosen hyperparameters
    X_train:
    X_test:
    y_train:
    y_test:

    outputs
    ------
    scoring of performance
    """
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)
    y_pred_class = model.predict(X_test)

    model_ll = log_loss(y_test, y_pred_prob)
    model_acc = accuracy_score(y_test, y_pred_class)
    model_auc = roc_auc_score(y_test, y_pred_class)
    
    return model_ll, model_acc, model_auc

def important_gene_mask(columns, coefs):
    """
    inputs
    ------
    columns: columns of df
    coefs: beta weights of lasso
    
    results
    ------
    important_genes: name of genes with weight != 0
    gene_weights: beta weights of genes
    """
    
    mask = coefs[0] != 0
    
    gene_weights = coefs[0][mask]
    important_genes = columns[mask]

    return dict(zip(important_genes, gene_weights))

def important_gene_mask_tree(columns, coefs):
    """
    gene finder for tree based models since coef_ and feature_importances
    work differently.
    
    inputs
    ------
    columns: columns of df
    coefs: beta weights of lasso
    
    results
    ------
    important_genes: name of genes with weight != 0
    gene_weights: beta weights of genes
    """
    
    mask = coefs != 0
    
    gene_weights = coefs[mask]
    important_genes = columns[mask]

    return dict(zip(important_genes, gene_weights))

def cv_feature_intersections(columns, weights, k, tree=True):
    """
    function that finds the intersection and average weight over k-folds

    inputs
    -----
    columns: columns or features
    weights: the weights as in beta-coef or feature_importances
    k: number of folds used
    """
    if tree:
        cv_weights = [important_gene_mask_tree(columns, weights[i]) for i in range(k)] #list of weights 
    else:
        cv_weights = [important_gene_mask(columns, weights[i]) for i in range(k)]
    cv_intersection = set([x for nested_list in cv_weights for x in nested_list])
    
    weight_dict = {}
    for gene in cv_intersection:
        for i in range(k):
            if gene not in weight_dict:
                weight_dict[gene] = cv_weights[i][gene]
            else:
                weight_dict[gene] += cv_weights[i][gene]
    
    return cv_intersection



# def cross_val_performances(X, model, n):
#     """
#     function for validating performance in cross_valdiation

#     inputs
#     -----
#     X: data
#     model: model to use
#     n: number of kfolds
#     """
    
#     ll_performances = []
#     acc_performances = []
#     auc_performances = []

#     kf = KFold(n_splits=n, shuffle=True)
#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#         ll, acc, auc = model_fit_score(model, X_train, X_test, y_train, y_test)
        
#         ll_performances.append(ll)
#         acc_performances.append(acc)
#         auc_performances.append(auc)

#     return ll_performances, acc_performances, auc_performances

class ConsensusML():
    def __init__(self):
        for arg in ['X', 'y', 'X_train', 'X_test', 'y_train', 'y_test', 'columns',
                    'model1', 'model2', 'model3', 'names', 'model_performance_dictionary',
                    'intersection_and_weights', 'cv_performances' ]: # add to this ]:
            setattr(self, arg, None)
    def data_input(self, X, y, split_index=None):
        """
        step for initializing data. 
        creates X, y, X_train, X_test, y_train, y_test and columns

        inputs:
        --------
        X: feature data 
        y: target data
        split_index: int to split by data by in train/test/split (optional)
        
        """
        self.X = X
        self.y = y
        self.columns = self.X.columns

        if split_index:
            self.X_train = self.X.iloc[:split_index]
            self.X_test = self.X.iloc[split_index:]
            self.y_train = self.y.iloc[:split_index]
            self.y_test = self.y.iloc[split_index:]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=8)

    def manual_train_test(self, X, y, X_train, X_test, y_train, y_test):
        """
        step for manually choosing X_train, X_test, y_train, y_test and inputting X and y
        creates X, y, X_train, X_test, y_train, y_test and columns

        inputs:
        --------
        X: feature data 
        y: target data
        X_train, X_test, y_train, y_test: manually designated splits
        """
        self.X = X
        self.y = y
        self.columns = self.X.columns

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def model_input(self, model1, model2, model3, model_names):
        """
        step to initialize models

        inputs
        ------
        model1, model2, model3: models with  desired hyperparameters
        model_names: list of strings of model name ['Lasso', 'Random Forest', 'XGBoost']

        model names must be in one of the following: Lasso, Ridge, Logistic, Random Forest, XGBoost, Gradient Boosting
        """
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

        self.names = model_names
    
    def get_performances(self):
        """
        step to create dictionary tracking model performances on original train/test split.
        
        used for models that are either in logistic regression family (lasso, ridge, logistic) or tree based
        as the steps to get important features are different.
        
        output:
        -------
        model_performance_dictionary: dictionary that tracks Log Loss, Accuracy, AUC, and # of important features
        """
        model_performance_dictionary = {} #dictionary tracking model performance
        i = 0

        for model in [self.model1, self.model2, self.model3]:
            name = self.names[i]
            log_loss, accuracy, auc = model_fit_score(model, self.X_train, self.X_test, self.y_train, self.y_test)
            if name == 'Lasso' or name == 'Ridge' or name == 'Logistic':
                mask = model.coef_ != 0
                lasso_columns = self.columns[mask[0]]
                model_performance_dictionary[name] = {'Log Loss': log_loss, 'Accuracy': accuracy, 'AUC': auc,
                                                    'Genes': lasso_columns, 'Total Genes': len(lasso_columns)}
            else:
                mask = model.feature_importances_ != 0
                tree_columns = self.columns[mask]
                model_performance_dictionary[name] = {'Log Loss': log_loss, 'Accuracy': accuracy, 'AUC': auc,
                                                    'Genes': tree_columns, 'Total Genes': len(tree_columns)}
            i += 1
        self.model_performance_dictionary = model_performance_dictionary

    def get_weights(self):
        """
        step to create a set that is the intersection fo the three models.

        returns intersection and weights from lasso and xgb

        outputs
        ------
        gene_intersection: set of feature importance
        lasso_weights: weights of those featutres
        xgb_feature_importance: weights of those features
        """
        gene_set1 = set(self.model_performance_dictionary[self.names[0]]['Genes'])
        gene_set2 = set(self.model_performance_dictionary[self.names[1]]['Genes'])
        gene_set3 = set(self.model_performance_dictionary[self.names[2]]['Genes'])

        gene_intersection = set.intersection(gene_set1, gene_set2, gene_set3)
        intersection_mask = [x in gene_intersection for x in self.columns]

        intersection_and_weights = [gene_intersection]
        i = 0
        for model in [self.model1, self.model2, self.model3]:
            name = self.names[i]
            if name == 'Lasso' or name == 'Ridge' or name == 'Logistic':
                beta_weights = model.coef_[0][intersection_mask]
                intersection_and_weights.append(beta_weights)
            if name == 'XGBoost':
                xgb_feature_importance = model.feature_importances_[intersection_mask]
                intersection_and_weights.append(xgb_feature_importance)
            if name == 'Gradient Boosting':
                gb_feature_importance = model.feature_importances_[intersection_mask]
                intersection_and_weights.append(gb_feature_importance)
            i += 1

        self.intersection_and_weights = intersection_and_weights

    def cv_tune_model(self, params_list, k):
        """
        step to tune models with cross validation

        inputs
        ------
        params_list: list of dictionary of params to tune with
        k: number of folds
        """

        i = 0
        for model in [self.model1, self.model2, self.model3]:
            name = self.names[i]
            params = params_list[i]
            if name == 'Lasso':
                grid_search = GridSearchCV(model, params, cv=k, n_jobs=-1, scoring='neg_log_loss', verbose=True)
                grid_search.fit(self.X, self.y)
                self.model1 = LogisticRegression(penalty='l1', solver='saga', max_iter=10000, **grid_search.best_params_)
            if name == 'Random Forest':
                grid_search = GridSearchCV(model, params, cv=k, n_jobs=-1, scoring='neg_log_loss', verbose=True)
                grid_search.fit(self.X, self.y)
                self.model2 = RandomForestClassifier(n_estimators=1000, n_jobs=-1, **grid_search.best_params_)
            if model == 'XGBoost':
                grid_search = GridSearchCV(model, params,
                                        cv=k, n_jobs=-1, scoring='neg_log_loss', verbose=True)
                grid_search.fit(self.X, self.y)
                self.model3 = XGBClassifier(n_jobs=-1, **grid_search.best_params_)
    def cv_intra_consensus(self, k=3):
        
        lasso_performances = []
        rf_performances = []
        xgb_performances = []

        lasso_weights = []
        rf_weights = []
        xgb_weights = []
        
        kf = KFold(n_splits=k, shuffle=True)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            lasso_scores = model_fit_score(self.model1, X_train, X_test, y_train, y_test)
            rf_scores = model_fit_score(self.model2, X_train, X_test, y_train, y_test)
            xgb_scores = model_fit_score(self.model3, X_train, X_test, y_train, y_test)

            lasso_performances.append(lasso_scores)
            rf_performances.append(rf_scores)
            xgb_performances.append(xgb_scores)

            lasso_weights.append(self.model1.coef_)
            rf_weights.append(self.model2.feature_importances_)
            xgb_weights.append(self.model3.feature_importances_)
        
        self.cv_performances = {'Lasso': lasso_performances, 'Random Forest': rf_performances,
                                'XGBoost': xgb_performances}
        
        cv_lasso = cv_feature_intersections(self.columns, lasso_weights, k, tree=False)
        cv_rf= cv_feature_intersections(self.columns, rf_weights, k, tree=True)
        cv_xgb = cv_feature_intersections(self.columns, xgb_weights, k, tree=True)

        












