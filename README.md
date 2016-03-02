# semiGridSearchCV
Scikit-learn compliant Semi-supervised learning Grid Search with Cross Validation
Current Scikitlearn GridSearchCV library doesn't support semi-supervised learning algorithms, therefore it is necessary to build up your custom grid search CV. Workarounds are hopeless because you have to assign -1 to the missing labels in each fold iteration of CV. This implementation can be also proposed for Randomized SearchCv, GaussianSearchCv and EvolutionarySearchCv with a development proposal to scikitlearn.
USAGE: 
Since I cannot share whole project there are some parts that you have to change. You should use (be aware of util package)  regular scikitlearn roc auc package. If you are not intested in reporting ROC to the user just eliminate them
You need a property file to execute tester python file
