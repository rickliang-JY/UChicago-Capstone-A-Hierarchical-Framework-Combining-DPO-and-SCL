import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import Normalizer, StandardScaler
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import get_eval_metrics, seed_torch


# def train_and_evaluate_logistic_regression_with_val(train_data, train_labels, val_data, val_labels, test_data, test_labels, log_spaced_values=None, max_iter=500):
#     seed_torch(torch.device('cpu'), 0)
    
#     metric_dict = {
#         'bacc': 'balanced_accuracy',
#         'kappa': 'cohen_kappa_score',
#         'auroc': 'roc_auc_score',
#     }
    
#     # Logarithmically spaced values for regularization
#     if log_spaced_values is None:
#         log_spaced_values = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)
    
#     # loop over log_spaced_values to find the best C
#     best_score = -float('inf')
#     best_C = None
#     logistic_reg_final = None
#     for log2_coeff in tqdm(log_spaced_values, desc="Finding best C"):
#         # suppress convergence warnings
#         import warnings
#         warnings.filterwarnings("ignore")
        
#         logistic_reg = LogisticRegression(
#             C=1/log2_coeff,
#             fit_intercept=True,
#             max_iter=max_iter,
#             random_state=0,
#             solver="lbfgs",
#         )
#         logistic_reg.fit(train_data, train_labels)
        
#         # predict on val set
#         val_loss = log_loss(val_labels, logistic_reg.predict_proba(val_data))
#         score = -val_loss
        
#         # score on val set
#         if score > best_score:
#             best_score = score
#             best_C = log2_coeff
#             logistic_reg_final = logistic_reg
#     print(f"Best C: {best_C}")
    
#     # Evaluate the model on the test data
#     test_preds = logistic_reg_final.predict(test_data)
    
#     num_classes = len(np.unique(train_labels))
#     if num_classes == 2:
#         test_probs = logistic_reg_final.predict_proba(test_data)[:, 1]
#         roc_kwargs = {}
#     else:
#         test_probs = logistic_reg_final.predict_proba(test_data)
#         roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}

#     eval_metrics = get_eval_metrics(test_labels, test_preds, test_probs, roc_kwargs=roc_kwargs)
    
#     outputs = {
#         "targets": test_labels,
#         "preds": test_preds,
#         "probs": test_probs,
#     }
        
#     return eval_metrics, outputs

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import Normalizer, StandardScaler
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import get_eval_metrics, seed_torch


def train_and_evaluate_logistic_regression_with_val(train_data, train_labels, val_data, val_labels, test_data, test_labels, log_spaced_values=None, max_iter=500):
    seed_torch(torch.device('cpu'), 0)
    
    metric_dict = {
        'bacc': 'balanced_accuracy',
        'kappa': 'cohen_kappa_score',
        'auroc': 'roc_auc_score',
        'auprc': 'average_precision_score',
    }
    
    # Logarithmically spaced values for regularization
    if log_spaced_values is None:
        log_spaced_values = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)
    
    # loop over log_spaced_values to find the best C
    best_score = -float('inf')
    best_C = None
    logistic_reg_final = None
    for log2_coeff in tqdm(log_spaced_values, desc="Finding best C"):
        # suppress convergence warnings
        import warnings
        warnings.filterwarnings("ignore")
        
        logistic_reg = LogisticRegression(
            C=1/log2_coeff,
            fit_intercept=True,
            max_iter=max_iter,
            random_state=37,
            solver="lbfgs",
        )
        logistic_reg.fit(train_data, train_labels)
        
        # predict on val set
        val_loss = log_loss(val_labels, logistic_reg.predict_proba(val_data))
        score = -val_loss
        
        # score on val set
        if score > best_score:
            best_score = score
            best_C = log2_coeff
            logistic_reg_final = logistic_reg
    print(f"Best C: {best_C}")
    
    # Evaluate the model on the test data
    test_preds = logistic_reg_final.predict(test_data)
    
    num_classes = len(np.unique(train_labels))
    if num_classes == 2:
        test_probs = logistic_reg_final.predict_proba(test_data)[:, 1]
        roc_kwargs = {}
    else:
        test_probs = logistic_reg_final.predict_proba(test_data)
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}

    eval_metrics = get_eval_metrics(test_labels, test_preds, test_probs, roc_kwargs=roc_kwargs)
    

    outputs = {
        "targets": test_labels,
        "preds": test_preds,
        "probs": test_probs,
        "roc_curve": None,
        "prc_curve": None,
    }
    
    
    if num_classes == 2:
        
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        roc_auc = auc(fpr, tpr)
        
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate', fontsize=12)
        ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        ax_roc.legend(loc="lower right")
        ax_roc.grid(alpha=0.3)
        plt.tight_layout()
        
        
        precision, recall, _ = precision_recall_curve(test_labels, test_probs)
        avg_precision = average_precision_score(test_labels, test_probs)
        
        fig_prc, ax_prc = plt.subplots(figsize=(8, 6))
        ax_prc.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
       
        pos_ratio = np.sum(test_labels) / len(test_labels)
        ax_prc.plot([0, 1], [pos_ratio, pos_ratio], color='navy', lw=2, linestyle='--', 
                    label=f'Random Classifier (Prevalence = {pos_ratio:.3f})')
        ax_prc.set_xlim([0.0, 1.0])
        ax_prc.set_ylim([0.0, 1.05])
        ax_prc.set_xlabel('Recall', fontsize=12)
        ax_prc.set_ylabel('Precision', fontsize=12)
        ax_prc.set_title('Precision-Recall Curve', fontsize=14)
        ax_prc.legend(loc="best")
        ax_prc.grid(alpha=0.3)
        plt.tight_layout()
        
        
        outputs["roc_curve"] = fig_roc
        outputs["prc_curve"] = fig_prc
    else:
        print(f"Skipping ROC and PRC curves for multi-class problem (num_classes={num_classes})")
        
    return eval_metrics, outputs

from sklearn.linear_model import LogisticRegressionCV

def train_and_evaluate_logistic_regression_with_cv(train_data, train_labels, val_data, val_labels, test_data, test_labels, log_spaced_values=None, max_iter=500, cv=10):
    seed_torch(torch.device('cpu'), 0)
    
    metric_dict = {
        'bacc': 'balanced_accuracy',
        'kappa': 'cohen_kappa_score',
        'auroc': 'roc_auc_score',
        'auprc': 'average_precision_score',
    }
    
    # Logarithmically spaced values for regularization
    if log_spaced_values is None:
        log_spaced_values = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)
    
    # Combine train and val data for CV
    train_val_data = np.vstack([train_data, val_data])
    train_val_labels = np.concatenate([train_labels, val_labels])
    
    # LogisticRegressionCV with built-in CV
    logistic_reg_cv = LogisticRegressionCV(
        Cs=1/log_spaced_values,  # Note: Cs parameter, not C
        cv=cv,  # number of CV folds
        scoring='neg_log_loss',  # same as your validation criterion
        fit_intercept=True,
        max_iter=max_iter,
        random_state=0,
        solver='lbfgs',
        n_jobs=-1,  # parallel processing
        refit=True,  # automatically refit on full data with best C
    )
    
    print("Performing cross-validation...")
    logistic_reg_cv.fit(train_val_data, train_val_labels)
    
    best_C = 1/logistic_reg_cv.C_[0]
    print(f"Best C (inverse regularization): {best_C}")
    print(f"Best regularization lambda: {logistic_reg_cv.C_[0]}")
    
    # Evaluate the model on the test data
    test_preds = logistic_reg_cv.predict(test_data)
    
    num_classes = len(np.unique(train_labels))
    if num_classes == 2:
        test_probs = logistic_reg_cv.predict_proba(test_data)[:, 1]
        roc_kwargs = {}
    else:
        test_probs = logistic_reg_cv.predict_proba(test_data)
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}
    
    eval_metrics = get_eval_metrics(test_labels, test_preds, test_probs, roc_kwargs=roc_kwargs)
    
    return eval_metrics, logistic_reg_cv

def train_and_evaluate_logistic_regression(train_data, train_labels, test_data, test_labels, 
                                           log_spaced_values=None, max_iter=500, cv_folds=5):
    seed_torch(torch.device('cpu'), 0)
    
    metric_dict = {
        'bacc': 'balanced_accuracy',
        'kappa': 'cohen_kappa_score',
        'auroc': 'roc_auc_score',
    }
    
    # Logarithmically spaced values for regularization
    if log_spaced_values is None:
        log_spaced_values = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)
    
    # Use cross-validation to find the best C
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings("ignore")
    
    best_score = -float('inf')
    best_C = None
    
    for log2_coeff in tqdm(log_spaced_values, desc="Finding best C via CV"):
        logistic_reg = LogisticRegression(
            C=1/log2_coeff,
            fit_intercept=True,
            max_iter=max_iter,
            random_state=0,
            solver="lbfgs",
        )
        
        # Use cross-validation on training data (negative log loss)
        cv_scores = cross_val_score(
            logistic_reg, 
            train_data, 
            train_labels, 
            cv=cv_folds, 
            scoring='neg_log_loss'
        )
        score = cv_scores.mean()  # Average across folds
        
        if score > best_score:
            best_score = score
            best_C = log2_coeff
    
    print(f"Best C: {best_C} (CV score: {best_score:.4f})")
    
    # Train final model on ALL training data with best C
    logistic_reg_final = LogisticRegression(
        C=1/best_C,
        fit_intercept=True,
        max_iter=max_iter,
        random_state=0,
        solver="lbfgs",
    )
    logistic_reg_final.fit(train_data, train_labels)
    
    # Evaluate the model on the test data
    test_preds = logistic_reg_final.predict(test_data)
    
    num_classes = len(np.unique(train_labels))
    if num_classes == 2:
        test_probs = logistic_reg_final.predict_proba(test_data)[:, 1]
        roc_kwargs = {}
    else:
        test_probs = logistic_reg_final.predict_proba(test_data)
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}
    
    eval_metrics = get_eval_metrics(test_labels, test_preds, test_probs, roc_kwargs=roc_kwargs)
    
    outputs = {
        "targets": test_labels,
        "preds": test_preds,
        "probs": test_probs,
    }
        
    return eval_metrics, outputs