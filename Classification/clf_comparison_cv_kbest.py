import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report

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

#Lists of classifiers and classifier names
classifiers = [
    LogisticRegression(max_iter=10000),
    LogisticRegressionCV(cv=5, max_iter=10000),
    SVC(),
    RandomForestClassifier(random_state=0),
    MLPClassifier(max_iter=10000, random_state=0),
	LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()
]

clf_names = [
    'Logistic Regression',
    'Logistic Regression with CV',
    'Support Vector Classification',
    'Random Forest',
    'Multi-layer Perceptron',
    'Linear Discriminant Analysis',
    'Quadratic Discriminant Analysis'
]

#The training set consists of 600 texts, stratified by proficiency level.
#120 texts have been randomly selected for the test set,
#stratified by proficiency level and text type within each level.
training_set = pd.read_csv('Datasets/train_split.csv')

#Data matrices containing the chosen feature values
X_train = training_set[feats].to_numpy()

#Arrays containing the proficiency level labels for training and testing the classifier
y_train = training_set['prof_level'].to_numpy().ravel()

#Creating a file for saving cross-validation results
with open('Classification/cv_results_kbest.txt', 'a') as output_f:

    #Comparing different classifiers with a varied number of features
    for n_features in range(5, 25):
        output_f.write('\nNumber of features in model: ' + str(n_features) + '\n')
        for name, clf in zip(clf_names, classifiers):
            #10-fold cross-validation on the training set
            X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=0)
            pipeline = Pipeline([('scaler', StandardScaler()), 
                                 ('feature_selection', SelectKBest(k = n_features)),
                                 ('model', clf)])
            cross_val_results = cross_val_score(pipeline, X_train_shuffled, 
                y_train_shuffled, cv=10)
            preds = cross_val_predict(pipeline, X_train_shuffled, y_train_shuffled, cv=10)

            f1_scores = cross_val_score(pipeline, X_train_shuffled, y_train_shuffled, 
                cv=10, scoring='f1_macro')
            precision_scores = cross_val_score(pipeline, X_train_shuffled, y_train_shuffled,    cv=10, scoring='precision_macro')
            recall_scores = cross_val_score(pipeline, X_train_shuffled, y_train_shuffled, 
                cv=10, scoring='recall_macro')
            #Writing the results to the file
            output_f.write('\n' + name + ':')
            output_f.write('\n10-fold cross validation scores: ' + str(cross_val_results))
            output_f.write('\nMean CV accuracy: ' + str(cross_val_results.mean()))
            output_f.write('\nSD of the CV accuracy: ' + str(cross_val_results.std()))
            output_f.write('\nAverage macro F1: ' + str(f1_scores.mean()) + 
                '(SD: ' + str(f1_scores.std()) + ')')
            output_f.write('\nAverage macro precision: ' + str(precision_scores.mean()) + 
                ' (SD: ' + str(precision_scores.std()) + ')')
            output_f.write('\nAverage macro recall: ' + str(recall_scores.mean()) + 
                ' (SD: ' + str(recall_scores.std()) + ')\n')
            #Report of the predictions made for the test splits
            output_f.write(str(classification_report(y_train_shuffled, preds)))