######## 4DA3 Course Project########
## Mingming Zhang, Wenbo Liu, Hua Yao ##

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm
import seaborn as sns
import pandas as pd
import timeit

##################################################
# create a method for generating the confusion matrix
##################################################
def plot_confusion_matrix(model, y_m_test, y_m_pred):
    plt.figure(figsize=(10, 5))
    cf_matrix = confusion_matrix(y_m_test, y_m_pred)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title('%s 241 values with mean of the dataset\n' % model)
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    ## Display the visualization of the Confusion Matrix.
    plt.ion()
    plt.show()

##################################################
# create a method for generating the ROC graph
##################################################
def createROC(title, y_test, y_pred):
    plt.figure(figsize=(10, 5))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('ROC: %s' %title)

    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f%%' % (roc_auc * 100))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')

    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.show()

##################################################
# create a method for store the accuracy score for each model
##################################################
models=[]
def store_data(model_name,y_test, y_pred):
    model = {}
    model['label'] = '{} with meaned dataset'.format(model_name)
    model['pred'] = y_pred
    model['test'] = y_test
    model['acc'] = accuracy_score(y_test, y_pred)

    models.append(model)




masses_data = pd.read_csv(r'D:\Google Drive 1\McMaster Degree\Level 4-1\4DA3\Course Project\mammographic_masses.data.txt', na_values=['?'],
                         names = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])

# Since the columns which has null value has been marked out
# We can fill the null value with the mean of the column to keep the integrity of the data
mean_total_age = masses_data['age'].mean()
mean_total_shape = masses_data['shape'].mean()
mean_total_margin = masses_data['margin'].mean()
mean_total_density = masses_data['density'].mean()
masses_data['age'].fillna(mean_total_age, inplace=True)
masses_data['shape'].fillna(mean_total_shape, inplace=True)
masses_data['margin'].fillna(mean_total_margin, inplace=True)
masses_data['density'].fillna(mean_total_density, inplace=True)

# Normalization the data scale to make each parameter has the same scale
all_features = masses_data[['age', 'shape','margin', 'density']].values
all_classes = masses_data['severity'].values
feature_names = ['age', 'shape', 'margin', 'density']

scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)

# Split the data to train set and test set
X = all_features_scaled
y = all_classes

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=1)





##################################################
# SVM with linear kernel: build the model
##################################################
svc = svm.SVC(kernel='linear', C=1, gamma='scale')

# Fit the model
svm_train = svc.fit(X_train, y_train)

svm_train_time = timeit.Timer(stmt="svc.fit(X_train, y_train)", setup="from __main__ import svc, X_train, y_train")
print("SVM training time: ", svm_train_time.timeit(number=1000), "milliseconds")

# Predict the results
svm_y_pred = svm_train.predict(X_test)

svm_test_time = timeit.Timer(stmt="train.predict(X_test)", setup="from __main__ import svm_train, X_test")
print("SVM testing time: ", svm_train_time.timeit(number=1000), "milliseconds")

# Evaluates the accuracy of our prediction on the test set
svm_scores = model_selection.cross_val_score(svm_train, X, y, cv=10)

print(svm_scores)
# The mean score and the 95% confidence interval of the score estimate
print("SVM Accuracy: %0.2f (+/- %0.2f)" % (svm_scores.mean(), svm_scores.std() * 2))

# Plot the confusion matrix based on SVM prediction result
plot_confusion_matrix("SVM - Linear Kernel", y_test, svm_y_pred)

# Plot the ROC graph based on SVM prediction result
createROC("SVM-Linear Kernel", y_test, svm_y_pred)

# Add the results to a new dictionary for comparing at the end
store_data("SVM", svm_y_pred, y_test)



##################################################
# Random Forest Classifier
##################################################
rfc = RandomForestClassifier(n_estimators=10)

# Fit the model
rfc_train = rfc.fit(X_train, y_train)

rfc_train_time = timeit.Timer(stmt="rfc.fit(X_train, y_train)", setup="from __main__ import rfc, X_train, y_train")
print("RFC training time: ", rfc_train_time.timeit(number=1000), "milliseconds")

# Predict the results
rfc_y_pred = rfc_train.predict(X_test)

rfc_test_time = timeit.Timer(stmt="rfc_train.predict(X_test)", setup="from __main__ import rfc_train_time, X_test")
print("RFC testing time: ", svm_train_time.timeit(number=1000), "milliseconds")

# Evaluates the accuracy of our prediction on the test set
rfc_cv_scores = model_selection.cross_val_score(rfc_train, X, y, cv=10)
print(rfc_cv_scores)
# The mean score and the 95% confidence interval of the score estimate
print("RFC Accuracy: %0.2f (+/- %0.2f)" % (rfc_cv_scores.mean(), rfc_cv_scores.std() * 2))

# Plot the confusion matrix based on RFC prediction result
plot_confusion_matrix("Random forest classifier", y_test, rfc_y_pred)

# Plot the ROC graph based on SVM prediction result
createROC("Random Forest Classifier", y_test, rfc_y_pred)

## Adding the results to a new dictionary to compare at the end
store_data("Random forest classifier", rfc_y_pred, y_test)

##################################################
# Comparison the two models result
##################################################
plt.figure(figsize = (15, 5))

for m in models:
    mod = m['label']
    y_pred = m['pred']
    y_test = m['test']
    # Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    # Calculate Accuracy of the curve to display on the plot
    auc = metrics.auc(fpr, tpr)#metrics.roc_auc_score(y_test, y_pred)
    # Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f%%)' % (m['label'], auc*100))
# Custom settings for the plot
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Display
plt.ioff()
plt.show()


