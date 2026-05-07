import pandas as pd  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score
#loading the dataset 
data = pd.read_csv("C:\Users\Naveen\Desktop\my_project\dataset\loan_prediction_dataset.csv")

data = data.drop(columns = ['Employment_Status']) 

x = data.drop(columns = ['Loan_Approved'])
y = data['Loan_Approved']

model = LogisticRegression()
model.fit(x,y)  

model_prediction = model.predict(x) 

print("accuracy score:", accuracy_score(y, model_prediction))