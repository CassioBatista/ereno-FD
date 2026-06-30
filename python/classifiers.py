"""GenericClassifiers equivalent — sklearn classifier wrappers matching Weka classifiers."""
from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class ClassifierExtended:
    """Wraps a sklearn classifier with a display name, mirroring Java's ClassifierExtended."""

    def __init__(self, classifier, name: str, enabled: bool = True):
        self.classifier = classifier
        self.name = name
        self.enabled = enabled

    def get_classifier(self):
        return self.classifier

    def get_classifier_name(self) -> str:
        return self.name

    def reset(self):
        """Return a fresh clone so each fold gets a clean estimator."""
        return self.classifier.__class__(**self.classifier.get_params())


# Weka RandomTree: at each node considers a random subset of K features
# sklearn equivalent: DecisionTreeClassifier with splitter='random' and max_features='sqrt'
RANDOM_TREE = ClassifierExtended(
    DecisionTreeClassifier(splitter="random", max_features="sqrt", random_state=42),
    "RandomTree",
)

# Weka J48 (C4.5): entropy-based pruned decision tree
# sklearn equivalent: DecisionTreeClassifier with entropy criterion
J48 = ClassifierExtended(
    DecisionTreeClassifier(criterion="entropy", random_state=42),
    "J48",
)

# Weka REPTree: reduced-error pruning decision tree
# sklearn equivalent: DecisionTreeClassifier (CART) with mild depth limit
REP_TREE = ClassifierExtended(
    DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2, random_state=42),
    "REPTree",
)

# Weka NaiveBayes: Gaussian Naive Bayes for numeric attributes
NAIVE_BAYES = ClassifierExtended(GaussianNB(), "NaiveBayes")

# Weka RandomForest: 100 trees
RANDOM_FOREST = ClassifierExtended(
    RandomForestClassifier(n_estimators=100, random_state=42),
    "RandomForest",
)

# Weka IBk (k-NN, k=1 by default)
KNN = ClassifierExtended(KNeighborsClassifier(n_neighbors=1), "KNN")

all_classifiers = [RANDOM_TREE, J48, REP_TREE, NAIVE_BAYES, RANDOM_FOREST]
all_custom = all_classifiers
