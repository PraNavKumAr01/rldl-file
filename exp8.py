import json
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# Preprocess the dataset
def preprocess_data(data):
    tokenizer = Tokenizer()
    all_patterns = []
    all_tags = []
    tag_responses = {}

    for intent in data['intents']:
        tag = intent['tag']
        tag_responses[tag] = intent['responses']
        for pattern in intent['patterns']:
            all_patterns.append(pattern)
            all_tags.append(tag)

    tokenizer.fit_on_texts(all_patterns)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(all_patterns)
    max_sequence_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

    # Convert tags to one-hot encoding
    unique_tags = list(set(all_tags))
    tag_to_index = {tag: idx for idx, tag in enumerate(unique_tags)}
    index_to_tag = {idx: tag for tag, idx in tag_to_index.items()}
    labels = np.array([tag_to_index[tag] for tag in all_tags])

    return tokenizer, padded_sequences, labels, max_sequence_len, unique_tags, tag_responses, index_to_tag

# Build the model
def build_model(vocab_size, max_sequence_len, num_tags):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_sequence_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_tags, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train, epochs=100, batch_size=16):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Get response
def get_response(model, tokenizer, input_text, max_sequence_len, unique_tags, tag_responses, index_to_tag):
    sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_len, padding='post')
    prediction = model.predict(padded_sequence)
    tag_idx = np.argmax(prediction)
    tag = index_to_tag[tag_idx]
    return random.choice(tag_responses[tag])

# Main function
def main():
    # Load and preprocess the data
    data = load_data('intents.json')
    tokenizer, X_train, y_train, max_sequence_len, unique_tags, tag_responses, index_to_tag = preprocess_data(data)

    # Build and train the model
    vocab_size = len(tokenizer.word_index) + 1
    model = build_model(vocab_size, max_sequence_len, len(unique_tags))
    train_model(model, X_train, y_train, epochs=200)

    # Chat with the bot
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        response = get_response(model, tokenizer, user_input, max_sequence_len, unique_tags, tag_responses, index_to_tag)
        print("Bot:", response)

if __name__ == "__main__":
    main()