import os,sys
import getopt
import pandas as pd
import sqlalchemy
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef,accuracy_score

from semiGridSearchCV import semiGridSearchCV
from util.monitoring import Timer
from util.plotting import *
import util as u
import logging
import logging.config
import reader.readerProperties as rP

def main(argv):
    """
    Launch the Model Selection & Prediction/Annotation process for all Targets
    Parameters
    ----------
    argv
    Command line arguments
    Returns
    -------
    Nothing
    """
    # Create logger
    logging.config.fileConfig(os.path.join(u.main_path(), 'resources', 'log', 'logging.conf'))
    logger = logging.getLogger('predictor')
    logger.info("START MODEL SELECTION & ANNOTATION PHASE")
    #Properties config
    propertiesPath = u.read_parameters_from_console(argv,logger)
    properties = rP.ReaderProperties(propertiesPath)
    propertiesMap = properties.obtain_config_section_map("CONFIGURATION")
    CORPUS_NAME = propertiesMap['CORPUS_NAME']
    N_JOBS = int(propertiesMap['JOBS']) if propertiesMap['JOBS']!="None" and propertiesMap['JOBS']!="" else None
    OUT_PATH = propertiesMap['OUT_PATH']
    TEST_MODE = bool(propertiesMap['TEST_MODE']=="True")
    SEED = int(propertiesMap['SEED'].strip()) if propertiesMap['SEED']!=None and propertiesMap['SEED'].strip()!="None" and propertiesMap['SEED'].strip()!="" else None
    TARGETS = propertiesMap['TARGETS'].lower()
    LANGUAGE = propertiesMap['LANGUAGE'].lower()

    propertiesMap = properties.obtain_config_section_map("MODELLING")
    BLACKLIST_ACTIVE = bool(propertiesMap['BLACKLIST_ACTIVE'])


    FILENAME=CORPUS_NAME+'_output.txt'
    pathToPython=os.getcwd()
    projectPath = os.path.split(pathToPython)[0]
    dbpath = os.path.split(projectPath)[0]

    DB_PATH = os.path.join(dbpath,'db','annotationdb')
    sqlite = 'sqlite://'
    if os.name=='posix' or os.name=='nt':
        sqlite = sqlite+'/'
    engine = sqlalchemy.create_engine(sqlite+str(DB_PATH)+str(LANGUAGE).upper()+'.sqlite')
    DF=pd.read_sql_table('nlp_featurized_out', engine,index_col="index")

    split_targets = TARGETS.split(' ')
    targets = []
    for target in split_targets:
        targets.append(target)

    eval_output_file=OUT_PATH+CORPUS_NAME+"_Model_Selection_CV_Results.txt"
    open(eval_output_file, "w+").close()
    for i,t in enumerate(targets):
        headline="Model Selection & Annotation Process for Target Label :"+t+"\n"
        logger.info(headline)
        f=open(eval_output_file, "a")
        f.write(headline)
        f.close()
        if i==1:
            DF=pd.read_sql_table('nlp_annotated_out', engine,index_col="index")
        m=BaseModeller(t,DF,OUT_PATH,FILENAME,TEST_MODE,logger,eval_output_file,argv)
        predictions,unlabelled_indexes=m.fit_predict(BLACKLIST_ACTIVE,TEST_MODE,N_JOBS,SEED)
        predDF=pd.DataFrame(data={"index":unlabelled_indexes,t+"_predictions":predictions}).set_index("index")
        outDF=DF.join(predDF, how='left')
        outDF.drop(t+'_num_label',inplace=True,axis=1)
        DF=outDF
        outDF.to_sql('nlp_annotated_out', engine, if_exists='replace')
    logger.info("FINISH MODEL SELECTION & ANNOTATION PHASE")

class BaseModeller:
    def __init__(self,target_label,DF,OUT_PATH,FILENAME,TEST_MODE,logger,eval_output_file,argv):
        self.target_label=target_label
        self.DF=DF
        self.reduced_matrix=np.load(OUT_PATH+FILENAME[:FILENAME.index(".")]+'.npy')
        self.blacklisted_estimators=[]
        self.es=None
        self.best_estimator=None
        self.valid_found=False
        self.test_predictions=None
        self.argv=argv
        if not TEST_MODE:
            self.labelled_indexes=np.array(self.DF[self.target_label+'_num_label'][self.DF[self.target_label+'_num_label'].notnull()].index)
            self.num_classes=dict(sorted(pd.unique(self.DF.loc[self.DF[self.target_label+'_num_label'].notnull(),[self.target_label+'_num_label',target_label]].values),key=lambda x:x[1]))

        else:
            self.labelled_indexes=np.arange(0,500)
            self.num_classes=dict(sorted(pd.unique(self.DF[[self.target_label+'_num_label',target_label]].values),key=lambda x:x[1]))

        logger.info(str(len(self.labelled_indexes))+" labelled instances detected as a SEED for modelling.")
        mask = np.ones(len(self.DF[self.target_label+'_num_label']), dtype=bool)
        mask[self.labelled_indexes] = False

        self.unlabelled_indexes=np.copy(self.DF.index)[mask]
        self.logger=logger
        self._eval_output_file=eval_output_file

    def evluateTestData(self):
        y_scores=self.best_estimator.predict_proba(self.es.getUnlabelledData())
        auc=drawROCCurve(self.es.getUnlabelled_Ys(),y_scores,list(self.num_classes.keys()))
        self.logger.info("AUC:"+str(auc))
        self.logger.info("Accuracy:"+str(accuracy_score(self.es.getUnlabelled_Ys(), self.test_predictions)))
        if len(self.num_classes.keys())>2:
            try:
                self.logger.info(classification_report(self.es.getUnlabelled_Ys(), self.test_predictions))
            except Exception as e:
                self.logger.info(e)
        else:
            self.logger.info("Matthews Correlation Coefficient: ",matthews_corrcoef(self.es.getUnlabelled_Ys(), self.test_predictions))
        cm = confusion_matrix(self.es.getUnlabelled_Ys(), self.test_predictions, labels=list(self.num_classes.keys()))
        self.logger.info("Confusion matrix")
        self.logger.info(cm)

    def fit_predict(self,BLACKLIST_ACTIVE,TEST_MODE,N_JOBS,SEED):
        print(self.DF[self.target_label+'_num_label'])
        self.es=semiGridSearchCV(self.reduced_matrix,self.DF[self.target_label+'_num_label'],self.labelled_indexes,SEED,self.logger,self._eval_output_file,self.argv,n_jobs=N_JOBS)
        while not self.valid_found:

            #BEST ESTIMATOR SELECTION
            self.best_estimator=self.es.selectBestModel(self.blacklisted_estimators)

            #BEST ESTIMATOR TRAINING
            y_input=self.DF[self.target_label+'_num_label']
            y_train = np.copy(y_input)
            y_train[self.unlabelled_indexes] = -1

            try:
                with Timer() as t:
                    if self.best_estimator.__class__.__name__!="semiKMeans":
                        self.best_estimator.fit(self.reduced_matrix,y_train)
                        self.test_predictions=self.best_estimator.predict(self.es.getUnlabelledData())
                    else:
                        y_all=self.best_estimator.fit_transform(self.reduced_matrix,y_train)
                        self.test_predictions=y_all[self.unlabelled_indexes]
            finally:
                self.logger.info('Prediction took %.03f sec.' % t.interval)

            self.logger.debug("Unique prediction categories:"+str(np.unique(self.test_predictions)))
            blacklisted=len(np.unique(self.test_predictions))!=len(np.unique(y_input)) if TEST_MODE else len(np.unique(self.test_predictions))!=len(np.unique(y_input[(~np.isnan(y_input))]))
            if BLACKLIST_ACTIVE and blacklisted:
                self.blacklisted_estimators.append(self.best_estimator.__class__.__name__)
                self.logger.info("Estimator: "+self.best_estimator.__class__.__name__+" is blacklisted in "+self.__class__.__name__+". Re-executing the model selection process.")
                continue
            else:
                self.logger.info("Estimator is valid.")
                self.valid_found=True

            if TEST_MODE:
                self.evluateTestData()

            predictions=substituteLabelKeysByValue(self.test_predictions,self.num_classes)
            return predictions,self.unlabelled_indexes

def substituteLabelKeysByValue(label_array,transdict):
    return np.asarray([transdict[y] for y in label_array])


if __name__ == '__main__':
    main(sys.argv[1:])