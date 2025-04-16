from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_model(X, y):
    param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200],
    'class_weight': [None, 'balanced']
    }
    regression = LogisticRegression()
    grid_search = GridSearchCV(regression, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    print("Best params:", grid_search.best_params_)
    print("Best accuracy:", grid_search.best_score_)
    best_model = grid_search.best_estimator_
    best_model.fit(X, y)
    return best_model