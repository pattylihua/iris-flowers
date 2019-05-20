import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

df = pd.read_csv('iris.csv', names = names)

all_inputs = df[['sepal-length', 'sepal-width',
                             'petal-length', 'petal-width']].values
all_classes = df['class'].values

(X_train,
 X_test,
 y_train,
 y_test) = train_test_split(all_inputs, all_classes, train_size=0.75, random_state = 2)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

knc = KNeighborsClassifier(n_neighbors = 10)
knc.fit(X_train,y_train)
dists = knc.kneighbors_graph(mode = 'distance',n_neighbors = 10).toarray()

print('Ten most similar points: \n',knc.kneighbors(X_test[:3],n_neighbors = 10, return_distance = False))
print('Distance: \n',dists[0])

# Prediction
y_predict = knc.predict(X_test)
y_predict_prob = knc.predict_proba(X_test)
print('Prediction for first three samples in test set: ',y_predict[:3])
print('Probabilities for each classes: ',y_predict_prob[:3])
