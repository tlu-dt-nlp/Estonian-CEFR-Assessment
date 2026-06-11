# Predicting the CEFR level of Estonian learner texts
Datasets of Estonian L2 writings and source code used to train and test machine learning models for CEFR-based classification on the scale of A2 to C1.

## Texts

The main language data (`Texts/Train_test1`) contains 720 Estonian language proficiency examination writings from the years 2017–2020. Each of the four proficiency levels is represented by 180 texts graded with at least 60% of the maximum score and thus meeting the level requirements. The originally handwritten texts, along with anonymous metadata, were provided by the Estonian Education and Youth Board after digitization and pseudonymization. The following text types are represented per level:
* A2 and B1 - personal letters and narrative texts
* B2 - personal and semi-formal letters and argumentative texts
* C1 - argumentative texts

Additional evaluation of model generalizability employed a separate sample of exam writings (`Texts/Test2`). It comprises 398 texts from 2010, representing material from one exam per proficiency level and therefore fewer text types: A2- and B1-level narrative writings, B2-level semi-formal letters and argumentative texts, and C1-level argumentative texts.

The material is also publicly available in the [corpus query](https://elle.tlu.ee/tools/wordlist) of the Estonian language learning and analysis environment ELLE.

## Feature extraction

Four categories of features were extracted from the texts:
* **lexical features** – variables of lexical complexity, i.e., diversity, sophistication, and density
* **morphological features** – frequencies of parts-of-speech and grammatical categories of nominals and verbs
* **surface features** – text complexity measures related to word, sentence, and text length
* **error features** – frequencies of spelling and grammatical errors as detected by correction tools

The texts were tokenized, sentence-segmented, lemmatized, and morphologically tagged with [Stanza](https://stanfordnlp.github.io/stanza/)) to extract lexical, morphological, and surface features. Average noun abstractness was calculated using an [Estonian speed-reading software](https://kiirlugemine.keeleressursid.ee) that rates the abstractness of noun tokens on a three-point scale. The proportion of relatively rare vocabulary (i.e., tokens not representing the 1,000-5,000 top-frequency words) was determined in comparison with the lemmatized version of the [Estonian frequency dictionary](https://www.cl.ut.ee/ressursid/sagedused1/index.php?lang=en).

Error features were derived from the output of a context-sensitive statistical [spell-checker](https://aclanthology.org/2023.nodalida-1.79/) and a machine-translation-based [grammar correction tool](https://doi.org/10.18653/v1/2024.eacl-long.73).

## Datasets

Altogether, 154 linguistic features were extracted from the two samples exam writings. The resulting datasets are in the `Datasets` folder.

The main sample was randomly split into a training and test set consisting of 600 and 120 texts, respectively. The training set was used for comparative linguistic analysis of proficiency levels, cross-validation of classification parameters, and building the final machine learning models. These were tested on the holdout test set stratified for level and text types. 

The file `Datasets/train_test1_meta.csv` contains metadata, such as the exam time, score, and author characteristics. Similar metadata was not available for the additional test set.

It must be noted that the comparison of the newer and older exam dataset reveals an increase in complexity of the writing assignment responses between 2010 and 2017–2020. This is most evident at level C1, the main differences occurring in morphological features like the number of case forms and their proportions among nominal words. The CEFR-based grading principles and exam tasks have been in consistent development. Specifically, the C1-level writing tasks and grading instructions were revised in 2017. This affects the cross-dataset generalizability of the classification models trained on newer data. However, no other open-access L2 exam data was available to be used as a reference material.

## Classification

The linguistic features relevant for predicting proficiency level were chosen according the following criteria: 
(1) mean values significantly distinguish at least some adjacent levels (A2–B1, B1–B2, B2–C1)
(2) level-to-level changes are monotonic, i.e., unidirectional
(3) feature values correlate to proficiency level
(4) there is no substantial variation between text types within the same level

Machine learning models were built and evaluated with [Scikit-learn](scikit-learn.org). The classification pipelines comprised standardizing data, using a feature selection algorithm (univariate `SelectKBest` or sequential `SequentialFeatureSelector`), training a classifier, and validating it on test material. Separate pipelines were created for lexical, morphological, surface, and error features. In each case, two conditions were compared, allowing the predictive features to be chosen from (1) those regarded as relevant proficiency level predictors based on linguistic analysis, or (2) all available features. In the end, different groups of relevant features were combined to train and evaluate unified prediction models.

To identify the best classification parameters, 10-fold cross-validation was used on the training set. In case of each pipeline, the smallest number of features that entailed the highest possible average accuracy was preferred. The five best-performing parameter sets (classifier + feature selection method + number of features) were chosen for evaluation on the holdout test data. In addition to overall accuracy, per-level recall, precision, and F1-score, the recall per text type at levels A2–B2 was considered to avoid inconsistent classification. The three highest-scoring models were further tested for generalizability on test set 2. Balanced accuracy, i.e., macro-average recall per level, was calculated to compensate for the imbalanced dataset.

The accuracy of the best-performing pipelines with the pre-selected relevant features is summarized in the following table.

| Feature set | Classifier | Feature selection | No. of features | CV accuracy | Test1 accuracy | Test2 accuracy |
|:--------------|:-------------|:--------------|:--------------|:-------------|:--------------|:--------------|
| Lexical | Logistic Regression with CV | univariate | 5 | 0.907 | 0.875 | 0.766 |
| Morphological | Linear Discriminant Analysis | univariate | 25 | 0.858 | 0.867 | 0.672 |
| Surface | Logistic Regression | sequential | 3 | 0.930 | 0.933 | 0.843 |
| Error | Random Forest | sequential | 2 | 0.680 | 0.683 | 0.6 |
| Combined | Support Vector Machine | univariate | 23 | 0.945 | 0.975 | 0.796 |

For a comprehensive report on the results, refer to the citations below.

## Online implementation

The best-performing model parameters have been implemented in the [Writing Evaluator](https://elle.tlu.ee/corrector) tool of the [ELLE platform](https://doi.org/10.22364/bjmc.2024.12.4.17). Proficiency assessment is accompanied by spelling and grammatical error correction, feedback on text complexity, and editing suggestions (e.g., pointing out long sentences and word repetitions).

## Citing

Allkivi, K. (2026). Towards interpretable models for language proficiency assessment: Predicting the CEFR level of Estonian learner texts. *Journal of Responsible Technology*, *25*. [https://doi.org/10.1016/j.jrt.2026.100162](https://doi.org/10.1016/j.jrt.2026.100162)
