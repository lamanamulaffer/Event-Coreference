# Event-Coreference
This project involves detecting event coreferences in the ECB+ news corpus using deep learning models built using TensorFlow.
There are 2 subrepositories here: one for training and one for testing.
1. Coreference_training
  1. Model files (model00.py, model01.py, model02.py) - defines models with no hidden layer, 1 hidden layer and 2 hidden layers respectively
  2. train.py - defines the split of test set to balance out positive and negative data, training of the model, including saving checkpoints for recovery
  3. run_kfold.py - performs k-fold crossvalidation with 10 folds
2. Coreference_testing
  Two variations of testing are performed. The standard version when response to marked test data is compared. The second version is the more realistic version in that events are extracted from test data using event detection tools like caevo and semantic role labellers. 
  1. run_test.py and run_test_bf.py - runs test script for standard version and realistic version respectively
  2. threshold.py - coreference threshold experiments
  3. convert_semeval.py - prepares model output into clusters for scoring using corefscorer - scoring measures taken into consideration are: pairwise precision, recall and F1 as well as MUC, CEAT, B-CUBED and the CoNLL score
Collaborators: Arun Prasad and Amna Alzeyara
