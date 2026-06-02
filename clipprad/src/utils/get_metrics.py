import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def calculate_metrics(outputs, targets):
    accuracy = accuracy_score(targets, outputs)
    
    precision_binary = precision_score(targets, outputs)
    precision_weighted = precision_score(targets, outputs, average="weighted")
    precision_macro = precision_score(targets, outputs, labels=[0, 1], average="macro")
    precision_micro = precision_score(targets, outputs, labels=[0, 1], average="micro")
    
    recall_binary = recall_score(targets, outputs)
    recall_weighted = recall_score(targets, outputs, average="weighted")
    recall_macro = recall_score(targets, outputs, labels=[0, 1], average="macro")
    recall_micro = recall_score(targets, outputs, labels=[0, 1], average="micro")
    
    f1_binary = f1_score(targets, outputs)
    f1_weighted = f1_score(targets, outputs, average="weighted")
    f1_macro = f1_score(targets, outputs, labels=[0, 1], average="macro")
    f1_micro = f1_score(targets, outputs, labels=[0, 1], average="micro")
    
    return {
        "Accuracy": round(accuracy * 100, 2),
        
        "precision_binary": round(precision_binary * 100, 2),
        "recall_binary": round(recall_binary * 100, 2),
        "f1_binary": round(f1_binary * 100, 2),
        
        "precision_weighted": round(precision_weighted * 100, 2),
        "recall_weighted": round(recall_weighted * 100, 2),
        "f1_weighted": round(f1_weighted * 100, 2),
        
        "precision_macro": round(precision_macro * 100, 2),
        "recall_macro": round(recall_macro * 100, 2),
        "f1_macro": round(f1_macro * 100, 2),
        
        "precision_micro": round(precision_micro * 100, 2),
        "recall_micro": round(recall_micro * 100, 2),
        "f1_micro": round(f1_micro * 100, 2),
    }

def get_CM(trues_te, pres_te):
    cm = confusion_matrix(trues_te, pres_te) 
    return cm

def aggregate_fold_metrics(metrics, method="mean"):
    assert len(metrics) >= 1, 
    assert method in ["mean", "sum"], 
    if type(metrics[0]) == dict:
        data = {}
        for key, value in metrics[0].items():
            data[key] = aggregate_fold_metrics(
                [metrics[i][key] for i in range(len(metrics))]
            )
        return data
    else:
        if method == "mean":
            return np.round(np.mean(metrics), 2)
        elif method == "sum":
            return np.round(np.sum(metrics), 2)
        else:
            pass
        