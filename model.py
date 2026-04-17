import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Simple dataset
data = {
    'fever': [1,1,0,1,0],
    'cough': [1,0,1,1,0],
    'headache': [1,1,0,0,1],
    'disease': ['Flu','Flu','Cold','Flu','Migraine']
}

df = pd.DataFrame(data)

X = df[['fever','cough','headache']]
y = df['disease']

model = DecisionTreeClassifier()
model.fit(X, y)

def predict_disease(symptoms):
    input_data = [[
        symptoms.get('fever',0),
        symptoms.get('cough',0),
        symptoms.get('headache',0)
    ]]
    return model.predict(input_data)[0]