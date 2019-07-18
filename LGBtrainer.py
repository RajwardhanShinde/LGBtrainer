import numpy as np 
import gc
gc.collect()
from timeit import default_timer as timer
from hyperopt import STATUS_OK
from sklearn.model_selection import KFold
from sklearn import metrics
import lightgbm as lgb
from hyperopt import fmin
from hyperopt import Trials, tpe
from hyperopt import hp

class Model:
    def __init__(self, train, test, y_train=None, cv=3, num_rounds=1000, metric=None, objective='binary', max_eval=3):
        self.X = train
        self.test = test
        self.train_set = lgb.Dataset(train, label=y_train)
        self.y = y_train
        self.n_splits = cv
        self.num_rounds = num_rounds
        self.met = metric
        self.obj = objective
        self.m_eval = max_eval
        
    def objective(self, params):
        global ITERATION
        ITERATION += 1
        
        n_folds = self.n_splits
    
        start = timer()
    
        cv_results = lgb.cv(params, 
                            self.train_set, 
                            num_boost_round=self.num_rounds, 
                            nfold=n_folds,
                            early_stopping_rounds=100, 
                            metrics=self.met)
    
        run_time = timer() - start
        
        if self.met == 'auc':
                best_score = np.max(cv_results[self.met + '-mean'])
                loss = 1 - best_score
                n_estimators = int(np.argmax(cv_results[self.met + '-mean'])+1)
 
        else:
            loss = np.min(cv_results[self.met + '-mean'])
            n_estimators = int(np.argmax(cv_results[self.met + '-mean'])+1)
            
        return {'loss': loss, 'params': params, 'iteration': ITERATION, 'estimators': n_estimators,
                'train_time': run_time, 'status': STATUS_OK}
    
    
    
    def get_params(self, a=1):
        if self.obj == 'binary':
                space = {
                        'bagging_freq': hp.choice('bagging_freq', np.arange(1, 5, dtype=int)),
                        'bagging_fraction': hp.uniform('bagging_fraction',0.20,0.90),
                        'boost': hp.choice('boost',['gbdt']),
                        'feature_fraction': hp.uniform('feature_fraction',0.20,0.90),
                        'learning_rate': hp.loguniform('learning_rate',np.log(0.0070),np.log(0.010)),
                        'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(50, 90, dtype=int)),
                        'num_leaves': hp.choice('num_leaves', np.arange(5, 35, dtype=int)),
                        'min_sum_hessian_in_leaf': hp.choice('min_sum_hessian_in_leaf', np.arange(5, 35, dtype=int)),
                        'max_depth': hp.choice('max_depth', np.arange(-2, 2, dtype=int)),
                        'tree_learner': hp.choice('tree_learner', ['serial']),
                        'objective': hp.choice('objective', ['binary']),
                        'boost_from_average': hp.choice('boost_from_average', [False]),
                        'num_threads': hp.choice('num_threads', np.arange(8, 9, dtype=int)),
                        'verbosity': hp.choice('verbosity', np.arange(1, 2, dtype=int))
                        }
                
        else:
            space = { 
                    'num_leaves': hp.choice('num_leaves', np.arange(5, 35, dtype=int)),
                    'learning_rate': hp.loguniform('learning_rate',np.log(0.0070),np.log(0.010)),
                    'max_depth': hp.choice('max_depth', np.arange(-2, 2, dtype=int)),
                    'colsample_bytree': hp.uniform('colsample_bytree',0.1, 0.9)
                    }
        
        new_tpe = tpe.suggest
        new_trial = Trials()
        
        global  ITERATION
        ITERATION = 0

        best = fmin(fn=self.objective, space=space, algo=new_tpe, max_evals=self.m_eval, 
                    trials=new_trial, rstate=np.random.RandomState(50))
        
        bayes_trials_results = sorted(new_trial.results, key = lambda x: x['loss'])
        params = bayes_trials_results[:1]
        print('*'*40)
        print('Best Params\n:', bayes_trials_results[:1])
        
        params = params[0]['params']
        
        return params
    
    def lgb_model(self, params=None):
      folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=40)
      full_pred = 0
      score = []
      for train_index, val_index in folds.split(self.X, self.y):
          x_tr, x_te = self.X.iloc[train_index, :], self.X.iloc[val_index, :]
          y_tr, y_te = self.y[train_index], self.y[val_index]
                
          lgb_train = lgb.Dataset(x_tr, label=y_tr)
          lgb_test = lgb.Dataset(x_te, label=y_te)
                
          lgb_model = lgb.train(params, lgb_train, num_boost_round=self.num_rounds, early_stopping_rounds=100, 
                               verbose_eval=100, valid_sets=[lgb_test], )
          
          pred = lgb_model.predict(x_te)

          if self.met == 'auc':
               score.append(metrics.roc_auc_score(y_te, pred))
               print("ROC SCORE:", metrics.roc_auc_score(y_te, pred))
          else:
              score.append(np.sqrt(metrics.mean_squared_error(y_te, pred)))
              print("RMSE:", np.sqrt(metrics.mean_squared_error(y_te, pred)))
          
          print("Mean CV Score:", np.mean(score))
          full_pred += lgb_model.predict(self.test, num_iteration=lgb_model.best_iteration) / self.n_splits            
      return full_pred
    
    
