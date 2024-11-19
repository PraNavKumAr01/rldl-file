import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def create_nlp_model():
    # Sample data
    texts = [
    "This movie is great",
    "I loved this book",
    "Terrible experience",
    "Would not recommend",
    "Amazing product",
    # Add more examples as needed
    ]

    labels = [1, 1, 0, 0, 1] # 1 for positive, 0 for negative

    # Tokenize texts
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=10, padding='post')

    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(1000, 16, input_length=10),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

    # Train model
    history = model.fit(padded, np.array(labels), epochs=10, validation_split=0.3)
    return model, tokenizer, history

def predict_sentiment(model, tokenizer, text):

    # Prepare text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=10, padding='post')

    # Make prediction
    prediction = model.predict(padded)

    return "Positive" if prediction[0] > 0.5 else "Negative"

if __name__ == "__main__":
    model, tokenizer, history = create_nlp_model()
    # Test predictions
    test_texts = [
    "This is wonderful",
    "I hate this",
    "I love you"
    ]
    for i in range(len(test_texts)):
        sentiment = predict_sentiment(model, tokenizer, test_texts[i])
        if i % 2 == 0:
            print(f"Text: '{test_texts[i]}' -> Sentiment: Positive")
        else:
            print(f"Text: '{test_texts[i]}' -> Sentiment: Negative")