import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np
import pandas as pd

data = pd.read_csv('C:\\Users\\keyas\\Downloads\\3-2\\Bharat\\spam.csv',encoding="ISO-8859-1")
print(data.head())

data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
data.rename(columns={"v1": "target", "v2": "sms"},inplace=True)

df=data.copy()
print(df.head())

df['target'].value_counts()
df['sms'].duplicated().sum()

df.drop_duplicates(inplace=True)
df.reset_index(inplace=True)
df.drop("index",axis=1,inplace=True)

df.isna().sum() # to check null or missing values
le=LabelEncoder()
df['target']=le.fit_transform(df['target'])


corpus=[]
for i in range(df.shape[0]):
    review=re.sub('[^a-zA-Z]',' ',df['sms'][i])
    review=review.lower()
    corpus.append(review)

df['features']=corpus
print(df.head())

x=df['features']
y=df['target']
cvec=CountVectorizer()
cvdf=cvec.fit_transform(x)

cvdf.toarray()

sm=SMOTE()
x_sm,y_sm=sm.fit_resample(cvdf,y)
y_sm.value_counts()
x_train,x_test,y_train,y_test=train_test_split(x_sm,y_sm,test_size=0.2,random_state=0)

svm=SVC(kernel='linear')
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)
accuracy_score(y_test,y_pred)
#sample non-spam/ham or spam messages, 0 indicates non-spam message and 1 indicates spam messages
sms=["Hey, you have won a car !!!!. Conrgratzz",# spam so 1 should be output
     "Congratulations! You've won a free cruise vacation! Claim your prize now by clicking the link below.",#spam so 1 should be the output
     "Hi there! Just wanted to remind you about our meeting tomorrow at 10 am. Looking forward to seeing you there.",#non spam messaage so 0 should be the output
     "Hey, don't forget to pick up milk on your way home. Thanks!",#non spam message so 0 should be the output
     "Dear applicant, Your CV has been recieved. Best regards",#non spam message so 0 should be the output
     "YOU ARE CHOSEN TO RECEIVE A å£350 AWARD! Pls call claim to collect your award which you are selected to receive as a valued mobile customer."#spam message so 1 should be the output
    ]
unseen=[]

for i in range(len(sms)):
    review=re.sub('[^a-zA-Z]',' ',sms[i])
    review=review.lower()
    unseen.append(review)
yp=svm.predict(cvec.transform(unseen)) #output prediction
print(yp)
