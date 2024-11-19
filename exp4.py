import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt

def create_sentiment_model():

    # Parameters
    max_features = 10000 # number of words to consider as features
    maxlen = 500 # cut texts after this number of words
    batch_size = 32

    # Load data
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    # Pad sequences
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_features, 128),
        tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

    # Train model
    print('Training...')
    history = model.fit(
        x_train, 
        y_train, 
        batch_size=batch_size, 
        epochs=5,
        validation_data=(x_test, y_test))
    
    # Evaluate model
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(f'Test score: {score}')
    print(f'Test accuracy: {acc}')
    return model, history
def plot_training_results(history):
    plt.figure(figsize=(12, 4))
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.show()
if __name__ == "__main__":
    model, history = create_sentiment_model()
    plot_training_results(history)