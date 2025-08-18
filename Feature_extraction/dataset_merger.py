import pandas as pd

#The 'dataset_generator.py' script should first be run to obtain the dataset with lexical, morphological, and surface complexity features.
complexity_data = pd.read_csv('Feature_extraction/dataset.csv')
#The 'Error_features/error_finder.py' script should be run to obtain the dataset with error features.
error_data = pd.read_csv('Feature_extraction/Error_features/error_data.csv')

merged_data = complexity_data.merge(error_data, on='file_name')
merged_data.to_csv('Feature_extraction/merged_set.csv', index=False) #the file can be renamed based on the content, e.g., 'train_test.csv' or 'test_set_2.csv'