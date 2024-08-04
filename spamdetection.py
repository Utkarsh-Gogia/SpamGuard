#importing needed directories
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import classification_report, confusion_matrix

df=pd.read_csv('mail_data.csv') #reading the sample csv file
# print (df) #printing the whole file

data=df.where(pd.notnull(df),'') #picks up the not null data
# print(data.head(10)) #to print first 10 rows
# print(data.tail(10)) #to print last 10 rows
# print(data.info()) #prints info/structure of data
# print(data.shape) #tells shape of data (rows,columns)

data.loc[data['Category']== 'spam', 'Category'] =0 #setting spam to 0
data.loc[data['Category']== 'ham', 'Category'] =1 #setting ham to 1

x=data['Message'] #assigning x to the messages
y=data['Category'] #assigning y to ham/spam
# print(x)
# print(y)

xtrain, xtest, ytrain, ytest=tts(x,y, test_size=0.2, random_state=3) #splitting the data
# print(x.shape)
# print(xtrain.shape)
# print(xtest.shape) #shape of full,training and test data

fextract=tf(min_df=1, stop_words='english', lowercase=True) #transforming data to feature vector
xtrainfeat=fextract.fit_transform(xtrain)
xtestfeat=fextract.transform(xtest)
ytrain=ytrain.astype('int')
ytest=ytest.astype('int')

# print(xtrainfeat) #shows the feature (ie accuracy) of the trained data
# print(xtestfeat) #shows the feature of test data

model=lr() #initialzing the model
model.fit(xtrainfeat, ytrain) #training the model on the features and classification 
predictontraining=model.predict(xtrainfeat) #shows the prediction on training data
accuracyontraining=acc(ytrain,predictontraining) #shows the accuracy on the training data
predictontest=model.predict(xtestfeat) #shows the prediction on test data
accuracyontest=acc(ytest,predictontest) #shows the accuracy on the test data

#print(f"Accuracy on training data : {accuracyontraining*100}%\nAccuracy on test data : {accuracyontest*100}%")

text=input('\nEnter the mail to be checked : ')
test=[text]
testdatafeat=fextract.transform(test)
prediction=model.predict(testdatafeat)
if prediction[0]==0:
    print("Mail is spam!")
else : 
    print("Mail is not spam!")
print(f"Accuracy : {(accuracyontraining*100+accuracyontest*100)/2} %")

print("Classification Report:")
print(classification_report(ytest, predictontest))

print("Confusion Matrix:")
print(confusion_matrix(ytest, predictontest))