import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

def get_test_labels_by_type(test_set: pd.DataFrame, level: str, text_type: str) -> np.ndarray:
    '''Create an array of target labels to measure recall by text type at a given level.'''
    level_set = test_set[test_set.prof_level == level]
    type_set = level_set[level_set.text_type == text_type]
    y_test_type = type_set['prof_level'].to_numpy().ravel()
    return y_test_type

#List of features used for classification, taken from 'feat_lists.txt'
feats_selection = ['n_cases', 'n_Nom', 'n_Tra', 'n_Plur', 'S_cases', 'S_Nom', 'S_All', 
    'S_Tra', 'S_Plur', 'A_cases', 'A_Gen', 'A_Par', 'A_Ine', 'A_Ela', 'A_Tra', 'A_Sing', 
	'A_Plur', 'P_cases', 'P_Ine', 'P_Ela', 'P_Com', 'P_Prs', 'P_Dem', 'P_IntRel', 
    'V_Fin', 'V_Sing', 'V_Neg', 'V_Conv', 'K_Post', 'D', 'J', 'S_Prop', 'lemma_count', 
	'RTTR', 'CVV', 'D_TTR', 'S_abstr', 'rare_5000', 'MTLD', 'word_count', 'sent_count', 
	'word_length', 'sent_length', 'SMOG', 'syll_count', 'spell_error_ratio', 
	'avg_spell_error_ratio', 'error_word_ratio', 'errors_per_word', 'errors_per_sent', 
	'avg_error_word_ratio']
feats = feats_selection

#The training set consists of 600 texts, stratified by proficiency level.
#120 texts have been randomly selected for the test set,
#stratified by proficiency level and text type within each level.
training_set = pd.read_csv('Datasets/train_split.csv')
test_set = pd.read_csv('Datasets/test_split.csv')
#test_set = pd.read_csv('test_exam2.csv')
#test_set = test_set[(test_set.text_type == 'descrnar')|(test_set.text_type =='arg')|(test_set.text_type =='fletter')]

#Arrays containing the proficiency level labels for training and testing the classifier
y_train = training_set['prof_level'].to_numpy().ravel()
y_test = test_set['prof_level'].to_numpy().ravel()

#Separate test set labels for comparing the classification of different text types
y_test_A2_descr_nar = get_test_labels_by_type(test_set, 'A2', 'description/narration')
y_test_A2_pers_letter = get_test_labels_by_type(test_set, 'A2', 'personal letter')

y_test_B1_descr_nar = get_test_labels_by_type(test_set, 'B1', 'description/narration')
y_test_B1_pers_letter = get_test_labels_by_type(test_set, 'B1', 'personal letter')

y_test_B2_arg = get_test_labels_by_type(test_set, 'B2', 'argumentative writing')
y_test_B2_pers_letter = get_test_labels_by_type(test_set, 'B2', 'personal letter')
y_test_B2_form_letter = get_test_labels_by_type(test_set, 'B2', 'formal letter')

#Dataframes and matrices containing feature values
X_train_df = training_set[feats]
X_train = X_train_df.to_numpy()
X_test_df = test_set[feats]
X_test = X_test_df.to_numpy()

#Standardizing the predictive features before splitting the dataset
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(k=23).fit(X_train_scaled, y_train)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)
#X_train_selected = X_train_scaled
#X_test_selected = X_test_scaled

#classifier = MLPClassifier(max_iter=10000, random_state=0)
#sfs = SequentialFeatureSelector(classifier, n_features_to_select = 2)
#selector = sfs.fit(X_train_scaled, y_train)
#X_train_selected = selector.transform(X_train_scaled)
#X_test_selected = selector.transform(X_test_scaled)

#Retrieving selected feature names and scores
feature_indices = selector.get_support(indices=True)
selected_features_df = X_train_df.iloc[:,feature_indices]
selected_features = list(selected_features_df.columns)
feature_scores = selector.scores_[feature_indices]

#Separate test set matrices for within-level validation
#and comparing the classification of different text types
X_test_A2_descr_nar = X_test_selected[:15]
X_test_A2_pers_letter = X_test_selected[15:30]

X_test_B1_descr_nar = X_test_selected[30:45]
X_test_B1_pers_letter = X_test_selected[45:60]

X_test_B2_arg = X_test_selected[60:70]
X_test_B2_form_letter = X_test_selected[70:80]
X_test_B2_pers_letter = X_test_selected[80:90]

#Training the classifier
classifier = SVC()
#classifier = RandomForestClassifier(random_state=0)
#classifier = MLPClassifier(max_iter=10000, random_state=0)
#classifier = LogisticRegression()
#classifier = LogisticRegressionCV(cv=5, max_iter=10000)
#classifier = LinearDiscriminantAnalysis()
#classifier = QuadraticDiscriminantAnalysis()
model = classifier.fit(X_train_selected, y_train)

#Scoring the classifier on the training set
train_score = model.score(X_train_selected, y_train)

#Scoring the classifier on full test set and predicting proficiency levels
test_score = model.score(X_test_selected, y_test)
y_pred = model.predict(X_test_selected)

#Scoring the classifier separately on different text types per level
test_score_A2_descr_nar = model.score(X_test_A2_descr_nar, y_test_A2_descr_nar)
test_score_A2_pers_letter = model.score(X_test_A2_pers_letter, y_test_A2_pers_letter)

test_score_B1_descr_nar = model.score(X_test_B1_descr_nar, y_test_B1_descr_nar)
test_score_B1_pers_letter = model.score(X_test_B1_pers_letter, y_test_B1_pers_letter)

test_score_B2_arg = model.score(X_test_B2_arg, y_test_B2_arg)
test_score_B2_form_letter = model.score(X_test_B2_form_letter, y_test_B2_form_letter)
test_score_B2_pers_letter = model.score(X_test_B2_pers_letter, y_test_B2_pers_letter)

#Writing out the validation results
print('Selected features:', selected_features)
print('Feature scores:', feature_scores)
print('Training set classification accuracy:', train_score)
print('Test set classification accuracy:', test_score)
print(classification_report(y_test, y_pred))
print('Recall by text type:')
print('A2 descriptions/narrations:', test_score_A2_descr_nar)
print('A2 personal letters:', test_score_A2_pers_letter)
print('B1 descriptions/narrations:', test_score_B1_descr_nar)
print('B1 personal letters:', test_score_B1_pers_letter)
print('B2 argumentative writings:', test_score_B2_arg)
print('B2 formal letters:', test_score_B2_form_letter)
print('B2 personal letters:', test_score_B2_pers_letter)

#Creating and saving a confusion matrix visualization
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.xlabel('Predicted level')
plt.ylabel('True level')
plt.savefig('Classification/confusion_matrix.png')