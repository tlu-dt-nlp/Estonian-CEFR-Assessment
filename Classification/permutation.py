import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import permutation_importance

#Feature sets of the best-performing classification models
lex_feats = ['lemma_count', 'RTTR', 'CVV', 'MTLD', 'S_abstr']

morph_feats = ['n_cases', 'n_Nom', 'n_Tra', 'n_Plur', 'S_cases', 
    'S_Nom', 'S_Tra', 'S_Plur', 'A_cases', 'A_Gen', 'A_Tra', 'A_Sing', 
    'A_Plur', 'P_cases', 'P_Ela', 'P_Prs', 'P_Dem', 'P_IntRel', 
    'V_Fin', 'V_Sing', 'V_Neg', 'K_Post', 'D', 'J', 'S_Prop']

surf_feats = ['syll_count', 'word_length', 'word_count']

error_feats = ['errors_per_word', 'errors_per_sent']

mixed_feats = ['lemma_count', 'RTTR', 'CVV', 'MTLD', 'S_abstr',
    'n_cases', 'n_Tra', 'n_Plur', 'S_cases', 'S_Plur', 'A_cases', 
    'P_cases', 'P_Prs', 'P_Dem', 'P_IntRel', 'V_Fin', 'word_count',
    'sent_count', 'syll_count', 'word_length', 'sent_length', 'SMOG',
    'errors_per_word']

#Defining the feature set to be used for feature importance analysis
feature_set = mixed_feats

#Reading the training data and two test sets
training_set = pd.read_csv('Datasets/train_split.csv')
test_set = pd.read_csv('Datasets/test_split.csv')
test_set_2 = pd.read_csv('Datasets/test_set_2.csv')

y_train = training_set['prof_level'].to_numpy().ravel()
y_test = test_set['prof_level'].to_numpy().ravel()
y_test_2 = test_set_2['prof_level'].to_numpy().ravel()

X_train = training_set[feature_set].to_numpy()
X_test = test_set[feature_set].to_numpy()
X_test_2 = test_set_2[feature_set].to_numpy()

#Scaling the data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_scaled_2 = scaler.transform(X_test_2)

#Training the classifier
classifier = SVC()
#classifier = LogisticRegression(max_iter=10000)
#classifier = LogisticRegressionCV(cv=5, max_iter=10000)
#classifier = MLPClassifier(max_iter=10000, random_state=0)
#classifier = RandomForestClassifier(random_state=0)
#classifier = LinearDiscriminantAnalysis()
model = classifier.fit(X_train_scaled, y_train)

#Computing permutation importance scores
test1_perm_importance = permutation_importance(model, X_test_scaled, y_test, 
    n_repeats=10, random_state=0)

test2_perm_importance = permutation_importance(model, X_test_scaled_2, y_test_2, 
    scoring='balanced_accuracy', n_repeats=10, random_state=0)

#Replacing feature labels for printing and visualizing permutation importance results
feature_names = np.array(feature_set, dtype=object)

replace_dict = {'lemma_count': 'lemma count', 'S_abstr': 'noun abstr.', 
    'n_cases': 'nominal cases', 'n_Tra': 'transl. case', 'n_Plur': 'plural nominals',
    'S_cases': 'noun cases', 'S_Plur': 'plural nouns', 'A_cases': 'adjective cases',
    'P_cases': 'pronoun cases', 'P_Prs': 'pers. pronouns', 'P_Dem': 'dem. pronouns',
    'P_IntRel': 'int.-rel. pron.', 'V_Fin': 'finite verbs', 'word_count': 'word count',
    'sent_count': 'sentence count', 'syll_count': 'syllable count', 
    'word_length': 'word length', 'sent_length': 'sentence length', 
    'errors_per_word': 'corr. per word'}

for key, value in replace_dict.items():
    feature_names[feature_names == key] = value

#Writing feature importance scores obtained on the two datasets
sorted_indices_test1 = test1_perm_importance.importances_mean.argsort()
print('Test set 1 feature importance scores:')
print(feature_names[sorted_indices_test1])
print(test1_perm_importance.importances_mean[sorted_indices_test1])

sorted_indices_test2 = test2_perm_importance.importances_mean.argsort()
print('Test set 2 feature importance scores:')
print(feature_names[sorted_indices_test2])
print(test2_perm_importance.importances_mean[sorted_indices_test2])

#Creating permutation importance plots
plt.barh(feature_names[sorted_indices_test1], 
    test1_perm_importance.importances_mean[sorted_indices_test1])
plt.xlabel('Permutation importance')
plt.tight_layout()
plt.savefig('Classification/feat_importance_test1.png')
plt.close()

plt.barh(feature_names[sorted_indices_test2], 
    test2_perm_importance.importances_mean[sorted_indices_test2])
plt.xlabel('Permutation importance')
plt.tight_layout()
plt.savefig('Classification/feat_importance_test2.png')
plt.close()