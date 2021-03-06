# Prepare the dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('heart.csv')

dataset=pd.get_dummies(df,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])

standardscaler=StandardScaler()
columns_to_scale=['age','trestbps','chol','thalach','oldpeak']
dataset[columns_to_scale]=standardscaler.fit_transform(dataset[columns_to_scale])

y=dataset['target']
X=dataset.drop(['target'],axis=1)

# Train
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json

n_estimators, max_features = 10, 10
rf_classifier=RandomForestClassifier(n_estimators=n_estimators, max_features=max_features)
scores=cross_val_score(rf_classifier,X,y,cv=10)
score = scores.mean()

print("Scores: ", scores)
print("Score: ", score)

# Write the output to file
scores_dict = {k: round(v, 2) for k, v in enumerate(scores)}
json_object = json.dumps(scores_dict)

f = open("scores.json", "w")
f.write(json_object)
f.close()

# Plot the scores
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], scores, color='lightblue', linewidth=3)
plt.savefig("scores.png")
