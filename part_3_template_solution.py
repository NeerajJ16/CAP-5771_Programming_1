import numpy as np
from numpy.typing import NDArray
from typing import Any

from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.linear_model import LogisticRegression

import utils as u

import new_utils as nu
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        return {
            "class_counts": {},  # Replace with actual class counts
            "num_classes": 0,  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        # Enter code and return the `answer`` dictionary

        answer = {}
        
        clf = LogisticRegression(max_iter=300,random_state=self.seed)
        clf.fit(Xtrain,ytrain)
        k = [1, 2, 3, 4, 5]

        train_accuracies = []
        test_accuracies = []

        for likely_outcomes in k:
            res ={}
            train_pred = clf.predict_proba(Xtrain)
            train_top_k_accuracy = top_k_accuracy_score(ytrain, train_pred, k=likely_outcomes)
   
            test_pred = clf.predict_proba(Xtest)
            test_top_k_accuracy = top_k_accuracy_score(ytest,test_pred , k=likely_outcomes)
    
            train_accuracies.append((likely_outcomes, train_top_k_accuracy))
            test_accuracies.append((likely_outcomes, test_top_k_accuracy))
            res['score_train'] = train_top_k_accuracy
            res['score_test'] = test_top_k_accuracy
            answer[likely_outcomes]=res
            
      
        answer['clf']= clf
        answer['plot_k_vs_score_train'] = train_accuracies
        answer['plot_k_vs_score_test'] = test_accuracies
        answer['text_rate_accuracy_change'] = "The model continuously shows improved accuracy as the value of k increases for the testing data, suggesting that it is becoming more adept at predicting the top-k classes."
        answer['text_is_topk_useful_and_why'] = "As a useful tool for evaluating the model's performance, the top-k accuracy metric assesses the model's capacity to produce accurate predictions over a larger range of possible classes. This statistic provides a more thorough understanding of the model's ability to capture relevant patterns than do more conventional accuracy metrics."
    

        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """

        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        
        nine_indices = np.where(ytrain == 9)[0]

        num_instances_to_remove = int(len(nine_indices) * 0.9)

        indices_to_remove = np.random.choice(nine_indices, num_instances_to_remove, replace=False)
        X = np.delete(Xtrain, indices_to_remove, axis=0)
        y = np.delete(ytrain, indices_to_remove, axis=0)
        
        y = np.where(y == 7, 0, np.where(y == 9, 1, y))
        
        nine_indices_test = np.where(ytest == 9)[0]
        num_instances_to_remove_test = int(len(nine_indices_test) * 0.9)
        indices_to_remove_test = np.random.choice(nine_indices_test, num_instances_to_remove_test, replace=False)

        
        Xtest = np.delete(Xtest, indices_to_remove_test, axis=0)
        ytest = np.delete(ytest, indices_to_remove_test, axis=0)
        
        ytest = np.where(ytest == 7, 0, np.where(ytest == 9, 1, ytest))
        
        answer["length_Xtrain"] = len(X) 
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(y)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = X.max()
        answer["max_Xtest"] = X.max()



        # Answer is a dictionary with the same keys as part 1.B

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        # Enter your code and fill the `answer` dictionary
        answer = {}
        
        SV_clf=SVC(random_state=self.seed)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        scoring = {
                   'f1': make_scorer(f1_score, average='macro'),
                   'precision': make_scorer(precision_score, average='macro'),
                   'recall': make_scorer(recall_score, average='macro'),
                   'accuracy':'accuracy'
                  }

        scores_cv = {metric: cross_val_score(SV_clf, X, y, scoring=scoring[metric], cv=cv)
          for metric in scoring}

        scores = {}
        scores['mean_accuracy'] = np.mean(scores_cv['accuracy'])
        scores['mean_recall']   = np.mean(scores_cv['recall'])
        scores['mean_precision']= np.mean(scores_cv['precision'])
        scores['mean_f1']       = np.mean(scores_cv['f1'])
        scores['std_accuracy']  = np.std(scores_cv['accuracy'])
        scores['std_recall']    = np.std(scores_cv['recall'])
        scores['std_precision'] = np.std(scores_cv['precision'])
        scores['std_f1']        = np.std(scores_cv['f1'])

        SV_clf.fit(X,y)
        y_pred_train=SV_clf.predict(X)
        y_pred_test=SV_clf.predict(Xtest)

        answer["scores"]=scores

        answer['cv']=cv
        answer['clf']=SV_clf

        if scores_cv['precision'].mean() > scores_cv['recall'].mean():
            answer["is_precision_higher_than_recall"]= True
        else:
            answer["is_precision_higher_than_recall"]= False

        answer['explain_is_precision_higher_than_recall']='Yes, Precision being higher than recall signifies that the classifier is prone to fewer false positive errors compared to its effectiveness in accurately identifying all positive instances'

        answer['confusion_matrix_train'] = confusion_matrix(y,y_pred_train)
        answer['confusion_matrix_test']  = confusion_matrix(ytest,y_pred_test)

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        SV_clf_weighted = SVC(random_state=self.seed, class_weight={0: class_weights[0], 1: class_weights[1]})
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

       
        scoring = {
            'f1': make_scorer(f1_score, average='macro'),
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'accuracy': 'accuracy'
        }

        
        scores_cv_weighted = {metric: cross_val_score(SV_clf_weighted, X, y, scoring=scoring[metric], cv=cv)
                              for metric in scoring}

      
        scores_weighted = {}
        scores_weighted['mean_accuracy'] = np.mean(scores_cv_weighted['accuracy'])
        scores_weighted['mean_recall'] = np.mean(scores_cv_weighted['recall'])
        scores_weighted['mean_precision'] = np.mean(scores_cv_weighted['precision'])
        scores_weighted['mean_f1'] = np.mean(scores_cv_weighted['f1'])
        scores_weighted['std_accuracy'] = np.std(scores_cv_weighted['accuracy'])
        scores_weighted['std_recall'] = np.std(scores_cv_weighted['recall'])
        scores_weighted['std_precision'] = np.std(scores_cv_weighted['precision'])
        scores_weighted['std_f1'] = np.std(scores_cv_weighted['f1'])

        
        SV_clf_weighted.fit(X, y)

        
        y_pred_train_weighted = SV_clf_weighted.predict(X)
        y_pred_test_weighted = SV_clf_weighted.predict(Xtest)

        
        confusion_matrix_train_weighted = confusion_matrix(y, y_pred_train_weighted)
        confusion_matrix_test_weighted = confusion_matrix(ytest, y_pred_test_weighted)

        answer["scores"] = scores_weighted
        answer['cv'] = cv
        answer['clf'] = SV_clf_weighted
        answer['class_weights'] = class_weights
        answer['confusion_matrix_train'] = confusion_matrix_train_weighted
        answer['confusion_matrix_test'] = confusion_matrix_test_weighted
        answer['explain_purpose_of_class_weights'] = "Class weights are used to lessen the impact of class imbalance by penalizing misclassifications of the minority class more severely."
        answer['explain_performance_difference'] = "The difference in performance that can be attributable to class weights suggests that the model is more capable of generalizing to the minority class, which leads to more balanced performance measures in all classes."


        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
