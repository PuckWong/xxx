from skopt import BayesSearchCV

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest mse: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")
    
# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = lgb.LGBMRegressor(
        objective='binary',
        metric='rmse',
        n_jobs=-1,
        verbose=0
    ),
    search_spaces = {
        'learning_rate': (0.001, 0.03, 'uniform'),
        'num_leaves': (10, 70),      
        'max_depth': (3, 12),
        'min_child_samples': (0, 50),
        #'max_bin': (100, 400),
        'subsample': (0.01, 1.0, 'uniform'),
        'subsample_freq': (0, 10),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'min_child_weight': (0, 10),
        #'subsample_for_bin': (100000, 500000),
        'reg_lambda': (1e-9, 100, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        #'scale_pos_weight': (1e-6, 500, 'log-uniform'),
        'n_estimators': (100, 1000),
    },    
    scoring = 'neg_mean_squared_error',
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 1,
    n_iter = 10,   
    verbose = 1,
    #refit = True,
    random_state = 42
)

# Fit the model
result = bayes_cv_tuner.fit(X.values, Y.values, callback=status_print)
