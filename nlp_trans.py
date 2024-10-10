import json
import numpy as np
import pandas as pd
import re
import random
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization
import tensorflow as tf

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Load and preprocess data
def load_and_preprocess_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    dic = {"tag": [], "patterns": [], "responses": []}
    for example in data['intents']:
        for pattern in example['patterns']:
            dic['patterns'].append(pattern)
            dic['tag'].append(example['tag'])
            dic['responses'].append(example['responses'])
    
    df = pd.DataFrame.from_dict(dic)
    return df

# Create and compile model
def create_transformer_model(vocab_size, maxlen, num_classes):
    embed_dim = 32  # Embedding size for each token
    num_heads = 2   # Number of attention heads
    ff_dim = 32     # Hidden layer size in feed forward network inside transformer
    
    inputs = Input(shape=(maxlen,))
    embedding_layer = tf.keras.layers.Embedding(vocab_size + 1, embed_dim)(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(embedding_layer)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(20, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Training function
def train_model(model, X, y, epochs=100):
    return model.fit(
        x=X,
        y=y,
        batch_size=32,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)]
    )

# Response generation function
def create_response_generator(model, tokenizer, label_encoder, max_length, df):
    def generate_response(user_input):
        text = [re.sub('[^a-zA-Z\']', ' ', user_input.lower())]
        x_test = tokenizer.texts_to_sequences(text)
        x_test = pad_sequences(x_test, padding='post', maxlen=max_length)
        
        y_pred = model.predict(x_test)
        pred_tag = label_encoder.inverse_transform([y_pred.argmax()])[0]
        responses = df[df['tag'] == pred_tag]['responses'].values[0]
        return random.choice(responses)
    return generate_response

# Main execution
def main():
    # Load and preprocess data 
    df = load_and_preprocess_data(r"C:\Users\sanni\Downloads\Rakesh\rakesh_projects\Mental-Health-Assisting-Agent-NLP-Chatbot-main\Mental-Health-Assisting-Agent-NLP-Chatbot-main\intents.json")
    
    # Tokenize patterns
    tokenizer = Tokenizer(lower=True, split=' ')
    tokenizer.fit_on_texts(df['patterns'])
    vocab_size = len(tokenizer.word_index)
    
    # Convert text to sequences
    X = pad_sequences(tokenizer.texts_to_sequences(df['patterns']), padding='post')
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['tag'])
    
    num_classes = len(df['tag'].unique())
    
    # Create and train model
    model = create_transformer_model(vocab_size, X.shape[1], num_classes)
    model.summary()
    history = train_model(model, X, y)
    
    # Create response generator
    generate_response = create_response_generator(
        model, tokenizer, label_encoder, X.shape[1], df
    )
    
    return generate_response

if __name__ == "__main__":
    generate_response = main()
    
    # Example usage
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = generate_response(user_input)
        print(f"Bot: {response}")