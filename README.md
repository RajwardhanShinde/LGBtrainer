# LGBtrainer helps you find hyper params for LGBM and simplifies the process of training the model and finding hyperparams.

* Parameters:-

1. train = it should be your train dataset(which is fit for training purpose)
2. test = it should be your test dataset(which is fit for testing purpose)
3. y_train = it should be your target column or values(same rows as train)
4. cv = the number of splits or folds(it is used for both finding hyperparams + training the model)
5. num_rounds = number of training rounds(it is used for both finding hyperparams + training the model)
6. metric = only 'auc' and 'rmse' can be used(For now only these two are supported)
7. objective = 'binary' or 'regression' or any other can be provided
8. max_eval = number of evaluations performed for finding params(note:- larger number might take more time depending on size of dataset)

* Example:-

```
-from LGBtrainer import Model
-model = Model(train, test, y_train, metric='auc', objective='binary', max_eval=3, cv=5)
-params = model.get_params()
-predictions = model.lgb_model(params)
```
