#The eligibility of people without credit histories for loans is almost impossible to validate,
#since these people don’t have the required data to present. This report tries to approach this problem by looking at the data 
#from over 3 hundred thousand people, and trying to predict eligibility from any and all information that can be gathered
#from these people using a few different methods.
Logistic Regression with PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


print("Importing data...")

A = pd.read_csv
A = A[:130000]
lbl = A.loc[:,'TARGET'].tolist()
A = A.drop(A.columns[[0, 1]],axis=1)
A.replace(np.nan, -1,inplace = True)
cat = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
A = pd.get_dummies(A,columns = cat)
B = A.values.tolist()


train_img, train_lbl = B[:100000],lbl[:100000]
test_img, test_lbl = B[100000:130000],lbl[100000:130000]
train_img = train_img[:25000]
test_img = test_img[:25000]
print("Import Complete")

print("Starting PCA Dimensionality Reduction...")
scaler = StandardScaler()
scaler.fit(train_img)
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

pca = PCA(.90)
pca.fit(train_img)
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)
print("PCA Dimensionality Reduction complete")


print("Loading dataset into Logistic Regression Classifier...")

train_img = train_img[:10000]
test_img = test_img[:10000]
train_lbl = train_lbl[:10000]
test_lbl = test_lbl[:10000]

logisticRegr = LogisticRegression(solver = 'lbfgs')

logisticRegr.fit(train_img, train_lbl)

predicted = logisticRegr.predict(test_img)
expected = test_lbl[:10000]
print("Accuracy: ", accuracy_score(expected, predicted))
A = confusion_matrix(expected, predicted)
print(A)
for i in range(len(A)):
    for j in A[i]:
        print(j,end=' ')
    print(' ')
Random Forests with PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

print("Loading dataset...")
A = pd.read_csv(')
A = A[:175000]
lbl = A.loc[:,'TARGET'].tolist()
A = A.drop(A.columns[[0, 1]],axis=1)
A.replace(np.nan, -1,inplace = True)
cat = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
A = pd.get_dummies(A,columns = cat)
B = A.values.tolist()

images, labels = B[:175000],lbl[:175000]


clf = RandomForestClassifier(n_estimators=100)

train_x = images[:25000]
train_y = labels[:25000]

test_x = images[100000:130000]

print("Starting PCA Dimensionality Reduction...")
scaler = StandardScaler()
scaler.fit(train_x)
train_img = scaler.transform(train_x)
test_img = scaler.transform(test_x)

pca = PCA(.90)
pca.fit(train_img)
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)
print("PCA Dimensionality Reduction complete")

print("Loading dataset into Random Forest Classifier...")

clf.fit(train_x, train_y)
expected = labels[100000:130000]
predicted = clf.predict(test_x)

print("Accuracy: ", accuracy_score(expected, predicted))
A = confusion_matrix(expected, predicted)
print(A)
for i in range(len(A)):
    for j in A[i]:
        print(j,end=' ')
    print(' ')


Artificial Neural Network:
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import io

A = pd.read_csv('application_train.csv')
len(A)
lbl = A.loc[:,'TARGET'].tolist()
A = A.drop(A.columns[[0, 1]],axis=1)
A.replace(np.nan, -1,inplace = True)
cat = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
A = pd.get_dummies(A,columns = cat)
B = np.array(A.values.tolist())

train_images, train_labels,test_images, test_labels = B[:200000],lbl[:200000],B[200000:300000],lbl[200000:300000]
class_names = [0,1]

model = keras.Sequential([
    keras.layers.Dense(250,input_shape=(250,)),
    keras.layers.Dense(200, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.softmax)
])


model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
model.summary()


model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)




