from sklearn.tree import DecisionTreeClassifier

from helpers.data_loader import load, get_features_labels
from helpers.models import train_cart, test_cart, get_score

df = load('data/german.txt')
features, labels = get_features_labels(df)

clf = DecisionTreeClassifier()
clf = train_cart(clf, features, labels)
print(get_score(clf, features, labels))