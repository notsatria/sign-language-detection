import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dir = pickle.load(open('data.pickle', 'rb'))

# Convert the data into np array
data = np.asarray(data_dir['data'])
labels = np.asarray(data_dir['labels'])

# Split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create the model
model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)

# Test the model
y_pred = model.predict(x_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
