The `clf_comparison_cv_kbest.py` and `and clf_comparison_cv_sfs.py` scripts compare different classification methods in combination with univariate or sequential feature selection and a varying number of features. The output reports contain 10-fold cross-validation results based on the training set.

`validation_test1_kbest.py` and `clf_comparison_cv_sfs.py` offer a more detailed overview of the performance of predetermined classification parameters, including recall per text type within proficiency levels. The evaluation is based on the holdout test set.

`validation_test2.py` can be used to test cross-dataset generalizability based on additional exam data.

The contribution of individual features to prediction accuracy is measured by calculating permutation feature importance on both test sets, using `permutation.py`.