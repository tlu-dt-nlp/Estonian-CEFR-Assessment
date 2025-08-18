import pandas as pd

def get_random_sample(df: pd.DataFrame, level: str, n_sample: int=30, by_type=True) -> pd.DataFrame:
    '''Get a text sample of a given proficiency level. By default, the sample is stratified by text type'''
    df = df[df.prof_level == level]
    if by_type == False:
        df_test = df.sample(n_sample)
    else:
        type_count = df.text_type.unique().size
        n_sample = int(n_sample / type_count)
        df_test = df.groupby('text_type', group_keys = False).apply(lambda x: x.sample(n_sample))
    return df_test

#Reading the dataset to be split into training and test set
train_test = pd.read_csv('Datasets/train_test.csv', encoding='utf-8')

#Requesting random test samples by proficiency level
a2_test_split = get_random_sample(train_test, 'A2')
b1_test_split = get_random_sample(train_test, 'B1')
b2_test_split = get_random_sample(train_test, 'B2')
c1_test_split = get_random_sample(train_test, 'C1', by_type=False)

#Concatenating the test subsets and saving the full test set
test_subsets = [a2_test_split, b1_test_split, b2_test_split, c1_test_split]
test_split = pd.concat(test_subsets)
test_split.to_csv('test_split_2.csv', encoding='utf-8', index=False)

#Removing test set texts from initial dataset to form the training set
test_file_list = test_split['file_name'].values.tolist()
train_split = train_test[~train_test['file_name'].isin(test_file_list)]
train_split.to_csv('train_split_2.csv', encoding='utf-8', index=False)