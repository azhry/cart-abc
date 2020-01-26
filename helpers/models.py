from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from helpers.data_loader import load, get_features_labels
from helpers.plots import roc_curves_plot
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

def run_cart(path, test_size):
    df = load('data/german.txt')
    features, labels = get_features_labels(df)
    X_train, X_test, y_train, y_test = get_train_test_data(features, labels, test_size)
    y_test_binarized = class_binarize(y_test, [1, 2])

    clf = DecisionTreeClassifier()
    clf = train_cart(clf, X_train, y_train)

    n_classes = y_test_binarized.shape[1]
    y_pred_binarized = class_binarize(test_cart(clf, X_test), [1, 2])
    roc_auc_values, fpr, tpr = roc_auc(n_classes, y_test_binarized, y_pred_binarized)
    roc_curves_plot(roc_auc_values, fpr, tpr, n_classes)

def run_cart_kfold(path, n_splits):
    df = load(path)
    features, labels = get_features_labels(df)
    cv = StratifiedKFold(n_splits = n_splits)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    clf = DecisionTreeClassifier()

    for i, ((train, test), color) in enumerate(zip(cv.split(features, labels), colors)):
        probas_ = clf.fit(features.iloc[train], labels.iloc[train]).predict_proba(features.iloc[test])

        fpr, tpr, thresholds = roc_curve(labels.iloc[test], probas_[:, 1], pos_label = True)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        print(f'Score Fold-{i}: {clf.score(features.iloc[test], labels.iloc[test])}')

        plt.plot(fpr, tpr, lw = lw, color = color,
                label = 'ROC fold %d (area = %0.2f)' % (i, roc_auc))
    
    plt.plot([0, 1], [0, 1], linestyle = '--', lw = lw, color = 'k', 
            label = 'Luck')
    
    mean_tpr /= cv.get_n_splits(features, labels)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color = 'g', linestyle = '--',
            label = 'Mean ROC (area = %0.2f)' % mean_auc, lw = lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def get_train_test_data(features, classes, test_size, random_state = 0):
    return train_test_split(features, classes, test_size = test_size, random_state = random_state)

def train_cart(cart, features, classes):
    return cart.fit(features, classes)

def test_cart(cart, x_test):
    return cart.predict(x_test)

def get_score(clf, x_test, y_true):
    return clf.score(x_test, y_true)

def class_binarize(y, classes):
    return label_binarize(y, classes = classes)

def abc():
    pass

def roc_auc(n_classes, y_test, y_score):
    fpr = dict() # false positive rate
    tpr = dict() # true positive rate
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    return roc_auc, fpr, tpr

def reports():
    pass