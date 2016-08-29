# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.cross_validation import ShuffleSplit


def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')
    
   
if __name__ == "__main__":   
	# Read student data
	student_data = pd.read_csv("student-data.csv")
	print "Student data read successfully!"
	print student_data.head()
	
	# Calculate number of students
	n_students = len(student_data.index)

	# Extract feature columns
	feature_cols = list(student_data.columns[:-1])

	# Extract target column 'passed'
	target_col = student_data.columns[-1] 

	# Show the list of columns
	print "Feature columns:\n{}".format(feature_cols)
	print "\nTarget column: {}".format(target_col)

	# Separate the data into feature data and target data (X_all and y_all, respectively)
	X_all = student_data[feature_cols]
	y_all = student_data[target_col]

	# Convert the object features into binaries for classification
	X_all = preprocess_features(X_all)
	print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

	# Set the number of training points
	num_train = 300
	train_size = float(num_train)/float(n_students)
	# Set the number of testing points
	num_test = X_all.shape[0] - num_train
	test_size = float(num_test)/float(n_students)

	# Shuffle and split the dataset into the number of training and testing points above
	X_all, y_all = shuffle(X_all, y_all, random_state=0)
	X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size= test_size, train_size= train_size, random_state=42)

	# Show the results of the split
	print "Training set has {} samples.".format(X_train.shape[0])
	print "Testing set has {} samples.".format(X_test.shape[0])
		
	parameters = {'solver':('newton-cg', 'lbfgs'), 'C':[0.3,0.4, 0.5], 'intercept_scaling':[0.4,0.5], 'max_iter':[120,130,140]}

	# Initialize the classifier
	clf = LogisticRegression(random_state=40)

	# Make an f1 scoring function using 'make_scorer' 
	f1_scorer = make_scorer(f1_score, pos_label = 'yes')

	# Perform grid search on the classifier using the f1_scorer as the scoring method
	grid = GridSearchCV(clf, parameters, scoring=f1_scorer, n_jobs=2)

	# Fit the grid search object to the training data and find the optimal parameters
	grid_obj = grid.fit(X_train, y_train)

	# Get the estimator
	clf = grid_obj.best_estimator_

	print "Parameter 'max_iter' is {} for the optimal model.".format(clf.get_params()['max_iter'])
	print "Parameter 'solver' is {} for the optimal model.".format(clf.get_params()['solver'])
	print "Parameter 'C' is {} for the optimal model.".format(clf.get_params()['C'])
	print "Parameter 'intercept_scaling' is {} for the optimal model.".format(clf.get_params()['intercept_scaling'])

	# Report the final F1 score for training and testing after parameter tuning
	print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
	print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))

