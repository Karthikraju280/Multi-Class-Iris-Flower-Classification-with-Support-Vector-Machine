import pandas as pd 
from sklearn.preprocessing  import  StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                   header=None)

# Assign column names to the dataset 
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Encode the class labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
iris['class'] = le.fit_transform(iris['class'])

# Split the dataset into features and target
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

# Split the dataset into train and test sets
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM classifier
svm = SVC(kernel='rbf', C=1, gamma='auto')
svm.fit(X_train, y_train)

# Evaluate the model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy: {accuracy:.2f}')
