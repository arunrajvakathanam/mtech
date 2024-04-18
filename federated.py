import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define a function to create a simple Keras model
def create_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

# Initialize the global model
global_model = create_keras_model()

# Define the number of clients
num_clients = 3

# Define the number of local epochs and learning rate
num_local_epochs = 10
learning_rate = 0.1

# Define a function to train a model on a client's local data
def train_local_model(client_data):
    client_model = create_keras_model()
    client_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    client_model.fit(client_data['X'], client_data['y'], epochs=num_local_epochs, verbose=0)
    return client_model

# Define a function to perform federated averaging
def federated_average(global_model, client_models):
    new_weights = []
    for weights_list in zip(*[model.get_weights() for model in client_models]):
        new_weights.append(np.mean(weights_list, axis=0))
    global_model.set_weights(new_weights)

# Split the training data among the clients
client_data_indices = np.array_split(np.arange(len(X_train)), num_clients)

# Train the global model using Federated Averaging
for epoch in range(num_local_epochs):
    client_models = []
    for indices in client_data_indices:
        client_X, client_y = X_train[indices], y_train[indices]
        client_models.append(train_local_model({'X': client_X, 'y': client_y}))
    federated_average(global_model, client_models)
    print(f"Epoch {epoch+1}/{num_local_epochs} - Global model trained on client data")

# Evaluate the global model on the test data
global_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
test_loss, test_accuracy = global_model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')
