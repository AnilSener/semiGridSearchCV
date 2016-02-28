"""
Created on Sun Feb 28
@author: Anil Sener
"""

import numpy as np
import pandas as pd
import sys
import util as u
from util.plotting import *
import reader.readerProperties as rP
from sklearn.grid_search import ParameterGrid
from sklearn.metrics  import f1_score,matthews_corrcoef,accuracy_score
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from semisupervised.methods.ensembles import VotingClassifier
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from sklearn.cross_validation import StratifiedShuffleSplit
from semisupervised.methods.scikitTSVM import SKTSVM
from semisupervised.methods.scikitWQDA import WQDA
from semisupervised.methods.semiKMeans import semiKMeans
from semisupervised.frameworks.SelfLearning import *
from semisupervised.frameworks.CPLELearning import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

class semiGridSearchCV:
    def __init__(self, reduced_matrix,target,labelled_indexes,SEED,logger,eval_output_file,argv,n_jobs=1,verbose=5):
		"""
		Launch the Model Selection Object for a single target
		Parameters
		----------
		reduced_matrix: feature space
		target:Labels both labelled and unlabelled (-1)
		SEED: Random State for Cross Validation Shuffle Split
		logger:logger object
		eval_output_file: Output file to store model selection CV evaluation results and metrics for each target
		argv: to get properties file from commandline
		n_jobs: Parallelization process instances for joblib
		vebose: verbosity in parallelized execution
		Returns
		-------
		Nothing
		"""
        #Properties config
        propertiesPath = u.read_parameters_from_console(argv,logger)
        properties = rP.ReaderProperties(propertiesPath)
        no_of_classes=len(np.unique(target[labelled_indexes]))
        self.reduced_matrix=reduced_matrix
        mask = np.ones(len(target), dtype=bool)
        mask[labelled_indexes] = False
        self.labelled_data = self.reduced_matrix[labelled_indexes]
        self.unlabelled_data = self.reduced_matrix[mask]
        self.labelled_Ys=target[labelled_indexes]
        self.unlabelled_Ys=target[mask]
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.SEED=SEED
        self.MC_ALGORITHM_OPTIONS={k:eval(v) for k,v in dict(properties.obtain_config_section_map('MC_ALGORITHM_OPTIONS')).items()}
        self.BINARY_ALGORITHM_OPTIONS={k:eval(v) for k,v in dict(properties.obtain_config_section_map('BINARY_ALGORITHM_OPTIONS')).items()}
        #initializing Algorithms
        mc_models,mc_params,bin_models,bin_params=self.get_selected_algorithms()
        self.models=mc_models if no_of_classes>2 else bin_models
        self.params=mc_params if no_of_classes>2 else bin_params
        #There is also polynomial fit of SVM, we might add it for CPLE
        self.logger=logger
        if not set(self.models.keys()).issubset(set(self.params.keys())):
            missing_params = list(set(self.models.keys()) - set(self.params.keys()))
            self.logger.error("Some estimators have missing parameters: %s" % missing_params)

        self.keys = self.models.keys()
        self.grid_searches = {}
        self.grid_aucs = {}
        self.best_estimator = None
        self.best_score = None
        self.best_all_scores = None
        self.best_params= None
        self.summary_df=None
        self.best_auc=None
        self._eval_output_file=eval_output_file

    def is_best_score(self,scores):
        avg_score=np.mean(scores)
        std_score=np.std(scores)
        return std_score!=0 and (self.best_score==None or avg_score > self.best_score or avg_score==self.best_score and std_score<np.std(self.best_all_scores))

    def fit(self, X, y,n_folds=5):
        train_ratio=float(1/5)
        cv = StratifiedShuffleSplit(y, n_iter=n_folds, train_size=train_ratio, test_size=1.0-train_ratio,random_state=self.SEED)
        X=np.mat(X)
        y_mat=np.zeros((len(y),n_folds),dtype=int)
        #no_of_classes=len(np.unique(y))
        for key in self.keys:
            self.logger.info("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            pg=ParameterGrid(params)
            param_list=list(pg)
            self.logger.info("With %s." % len(param_list)+" different parameter combinations.")
            empty_scores={(key,str(z)):[] for z in param_list}
            self.grid_searches.update(empty_scores)
            self.grid_aucs.update(empty_scores)
            all_scores=Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(delayed(_fit_and_score)(clone(model),key, X, y,y_mat,train_index,test_index,parameters,i) for parameters in param_list for i,(train_index,test_index) in enumerate(cv))
            unzipped = list(zip(*all_scores))
            scores=list(unzipped[0])
            aucs=list(unzipped[1])
            
            i=0
            for g in param_list:
                #has_all_cats=True
                for z in range(n_folds):
                    self.grid_searches[key,str(g)].append(scores[i])
                    self.grid_aucs[key,str(g)].append(aucs[i])
                    i=i+1

                inst_score=np.mean(self.grid_searches[key,str(g)])
                if self.is_best_score(self.grid_searches[key,str(g)]):
                    self.best_score = inst_score
                    self.best_all_scores=self.grid_searches[key,str(g)]
                    self.best_estimator=model
                    self.best_params = g
                    self.best_auc= np.mean(self.grid_aucs[key,str(g)])



    def score_summary(self, sort_by= ['mean_score'],ascending=[0]):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score':np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series(dict(list(params.items()) + list(d.items())))

        rows = [row(k, d[1], eval(d[0][1])) for k in self.keys
                for d in filter(lambda i : i[0][0]==k,self.grid_searches.items())]
        def auc_row(aucs):
            d = {
                 'mean_roc_auc':np.mean(aucs),
            }
            return pd.Series(d)
        auc_rows = [auc_row(d[1]) for k in self.keys
                for d in filter(lambda i : i[0][0]==k,self.grid_aucs.items())]
        auc_df=pd.concat(auc_rows, axis=1).T
        self.summary_df = (pd.concat(rows, axis=1).T
                            .join(auc_df,how='left',rsuffix="_r")
                           .sort_values(sort_by, ascending=ascending))

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score','mean_roc_auc']
        columns = columns + [c for c in self.summary_df.columns if c not in columns]
        self.summary_df=self.summary_df[columns]
        self.summary_df.to_csv(self._eval_output_file,sep=' ',mode='a',  index=False, header=True)
        return self.summary_df

    def getBestEstimator(self):
        return self.best_estimator;

    def getBestScore(self):
        return self.best_score;
    def getBestAUC(self):
        return self.best_auc;

    def getBestParams(self):
        return self.best_params;

    def getLabelledData(self):
        return self.labelled_data
    def getUnlabelledData(self):
        return self.unlabelled_data

    def getLabelled_Ys(self):
        return self.labelled_Ys
    def getUnlabelled_Ys(self):
        return self.unlabelled_Ys

    def selectBestModel(self, estimator_blacklist=[]):

        if len(estimator_blacklist)>0:
            self.logger.info("Estimator blacklist identified:"+" ".join(estimator_blacklist))
            if len(estimator_blacklist)==len(self.models):
                self.logger.info("All algorithms are blacklisted, please select another algorithm.")
                sys.exit(-1)
            estimator_blacklist_alias_list=[alias for alias,model in self.models.items() if model.__class__.__name__ in estimator_blacklist]
            self.summary_df=self.summary_df[~self.summary_df["estimator"].isin(estimator_blacklist_alias_list)]
            next_best_result=self.summary_df.head(1)
            next_best_model=self.models[next_best_result.loc[next_best_result.first_valid_index(),"estimator"]]
            next_best_model_params={k:v for k,v in list(next_best_result.T.to_dict().values())[0].items() if k not in ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score' ,'mean_roc_auc'] and str(v)!="nan"}
            #TODO We have to handle also parameters like SVC__C stg.
            self.logger.info("Estimator "+next_best_model.__class__.__name__+" will be executed on unlabaled data with the following parameters"+str(next_best_model_params))
            next_best_model.set_params(**next_best_model_params)
            self.best_estimator=next_best_model
            return self.best_estimator

        self.fit(self.labelled_data, self.labelled_Ys,n_folds=5)
        best_score_print="Best Model Selection Score in CV is :"+str(self.getBestScore())+" and its ROC AUC: "+str(self.getBestAUC())+"\n"
        best_model_print="Best Model"+str(type(self.getBestEstimator()))+"  Params in CV is :"+str(self.getBestParams())+"\n"
        f = open(self._eval_output_file, 'a')
        f.writelines([best_score_print,best_model_print])
        f.close()
        self.logger.info(best_score_print)
        self.logger.info(best_model_print)
        summaryDF=self.score_summary()
        self.logger.info(summaryDF)
        return self.getBestEstimator()

    def get_selected_algorithms(self):
        lp=LabelPropagation()
        ls=LabelSpreading()
        orc_sv3m=OneVsRestClassifier(SKTSVM(probability=True))
        gnb=GaussianNB()
        sgd=SGDClassifier(loss='log',penalty= 'elasticnet',n_jobs=-1)

        gammas=np.logspace(-2, 2, 5)
        gammas=sorted(list(gammas*5)+list(gammas*5/2)+list(gammas*15/2)+list(gammas))
        neighbours=list(range(1,11))
        Cs=np.logspace(-8, -3, 6)
        fixedprecs=np.logspace(-9, -1, 9)
        lamUs=[1]#np.arange(0.25,2,0.25)
        alphas=np.logspace(-8, -2, 7)
        lone_ratios=np.arange(0,0.5,0.1)
        models = {
                'QN-S3VM': SKTSVM(probability=True),
                'LABELPROPAGATION': lp,
                'LABELSPREADING': ls,
                'SEMIKMEANS':semiKMeans(),
               'CPLE_GNB':CPLELearningModel(gnb,predict_from_probabilities=False,pessimistic=True),
            'CPLE_SGD':CPLELearningModel(sgd,predict_from_probabilities=False,pessimistic=True),
        }
        params = {
                'QN-S3VM':  [{'kernel': ['rbf'],'C':Cs,'gamma': gammas,'lamU':lamUs},
                             {'kernel': ['linear'],'gamma': gammas,'lamU':lamUs}],
                'LABELPROPAGATION': [{'kernel': ['rbf'],'gamma': gammas},
                      {'kernel': ['knn'] ,'n_neighbors': neighbours}],
                'LABELSPREADING': [{'kernel': ['rbf'],'gamma': gammas},
                        {'kernel': ['knn'] ,'n_neighbors': neighbours}],
                'SEMIKMEANS':  {'fixedprec':fixedprecs},
                'CPLE_GNB':{},
                 'CPLE_SGD': {'SGDClassifier__learning_rate':['optimal'],
                         'SGDClassifier__alpha':alphas,'SGDClassifier__n_iter':[5,10,15],'SGDClassifier__l1_ratio':lone_ratios}
            }
        mc_models={};mc_params={}
        for k,v in self.MC_ALGORITHM_OPTIONS.items():
            if v:
                mc_models[k]=models[k]
                mc_params[k]=params[k]

        bin_models={};bin_params={}
        for k,v in self.BINARY_ALGORITHM_OPTIONS.items():
            if v:
                bin_models[k]=models[k]
                bin_params[k]=params[k]
        return mc_models,mc_params,bin_models,bin_params

def _fit_and_score(model,key, X, y,y_mat,train_index, test_index,g,i):

    if g is not None:
        model.set_params(**g)
    y_mat[:,i]=y
    y_test=y_mat[test_index,i]
    y_mat[test_index,i]=-1
    if key != "SEMIKMEANS":
        model.fit(X,y_mat[:,i])
        y_pred=model.predict(X[test_index,:])
    else:
        y_new=model.fit_transform(X,y_mat[:,i])
        y_pred=y_new[test_index]
    no_of_classes=len(np.unique(y))
    score=f1_score(y_test,y_pred,average='micro') if no_of_classes>2 else matthews_corrcoef(y_test, y_pred)
    y_scores=model.predict_proba(X[test_index,:])
    auc=drawROCCurve(y_test,y_scores,list(np.sort(np.unique(y))))
    return score,auc