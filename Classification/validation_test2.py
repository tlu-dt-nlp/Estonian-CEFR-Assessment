import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer

def get_test_labels_by_type(test_set: pd.DataFrame, level: str, text_type: str) -> np.ndarray:
    '''Create an array of target labels to measure recall by text type.'''
    level_set = test_set[test_set.prof_level == level]
    type_set = level_set[level_set.text_type == text_type]
    y_test_type = type_set['prof_level'].to_numpy().ravel()
    return y_test_type

#List of features selected for classification
feats = ['n_cases', 'n_Tra', 'n_Plur', 'S_cases', 'S_Plur', 'A_cases', 'P_cases', 
    'P_Prs', 'P_Dem', 'P_IntRel', 'V_Fin', 'lemma_count', 'RTTR', 'CVV', 'S_abstr', 
    'MTLD', 'word_count', 'sent_count', 'word_length', 'sent_length', 'SMOG', 
    'syll_count', 'errors_per_word']

#Using the same training set as before but a new test set
training_set = pd.read_csv('Datasets/train_split.csv')
test_set = pd.read_csv('Datasets/test_set_2.csv')

#Arrays containing the proficiency level labels for training and testing the classifier
y_train = training_set['prof_level'].to_numpy().ravel()
y_test = test_set['prof_level'].to_numpy().ravel()

#Separate test set labels for comparing the classification of different text types
y_test_B2_arg = get_test_labels_by_type(test_set, 'B2', 'argumentative writing')
y_test_B2_form_letter = get_test_labels_by_type(test_set, 'B2', 'semiformal letter')

# Dataframes and matrices containing feature values
X_train_df = training_set[feats]
X_train = X_train_df.to_numpy()
X_test_df = test_set[feats]
X_test = X_test_df.to_numpy()

# Standardizing and imputing the predictive features before splitting the dataset
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Separate test set matrices for comparing the classification of different text types
X_test_B2_arg = X_test_scaled[224:287]
X_test_B2_form_letter = X_test_scaled[287:350]

#Training the classifier
classifier = SVC()
#classifier = RandomForestClassifier(random_state=0)
#classifier = MLPClassifier(max_iter=10000, random_state=0)
#classifier = LogisticRegression()
#classifier = LogisticRegressionCV(cv=5, max_iter=10000)
#classifier = LinearDiscriminantAnalysis()
#classifier = QuadraticDiscriminantAnalysis()
model = classifier.fit(X_train_scaled, y_train)

#Scoring the classifier on full test set and predicting proficiency levels
accuracy = model.score(X_test_scaled, y_test)
y_pred = model.predict(X_test_scaled)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

#Scoring the classifier separately on different text types per level
test_score_B2_arg = model.score(X_test_B2_arg, y_test_B2_arg)
test_score_B2_form_letter = model.score(X_test_B2_form_letter, y_test_B2_form_letter)

#Writing out the validation results
print('Test set 2 overall classification accuracy:', accuracy)
print('Test set 2 balanced accuracy:', balanced_accuracy)
print(classification_report(y_test, y_pred, zero_division=0))
print('Recall by text type:')
print('B2 argumentative writings:', test_score_B2_arg)
print('B2 formal letters:', test_score_B2_form_letter)

#Creating and saving a confusion matrix visualization
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.xlabel('Predicted level')
plt.ylabel('True level')
plt.savefig('Classification/confusion_matrix_test2.png')