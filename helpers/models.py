from sklearn.metrics import roc_curve, auc

def train_cart(cart, features, classes):
    return cart.fit(features, classes)

def test_cart(cart, x_test):
    return cart.predict(x_test)

def get_score(clf, x_test, y_true):
    return clf.score(x_test, y_true)

def abc():
    pass

def roc_auc(n_classes, y_test, y_score):
    fpr = dict() # false positive rate
    tpr = dict() # true positive rate
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i] = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['micro'], tpr['micro'] = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    return roc_auc, fpr, tpr

def reports():
    pass