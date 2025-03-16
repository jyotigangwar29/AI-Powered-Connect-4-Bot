import anvil.server
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Embedding, Dense, LayerNormalization, MultiHeadAttention, Dropout, Reshape, GlobalAveragePooling1D

# ✅ Define Learnable Positional Encoding
class LearnablePositionalEncoding(Layer):
    def __init__(self, seq_length=42, embed_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.pos_embedding = Embedding(input_dim=seq_length, output_dim=embed_dim)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        position_indices = tf.range(seq_length)
        position_embeddings = self.pos_embedding(position_indices)
        position_embeddings = tf.expand_dims(position_embeddings, axis=0)
        position_embeddings = tf.tile(position_embeddings, [batch_size, 1, 1])
        return inputs + position_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({"seq_length": self.seq_length, "embed_dim": self.embed_dim})
        return config

# ✅ Transformer Block
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),  
            Dense(embed_dim),
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.attention.key_dim,
            "num_heads": self.attention.num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "dropout_rate": self.dropout1.rate
        })
        return config

# ✅ Register Custom Layers in Keras
custom_objects = {
    "LearnablePositionalEncoding": LearnablePositionalEncoding,
    "TransformerBlock": TransformerBlock
}

# ✅ Connect to Anvil

import anvil.server

anvil.server.connect("server_2QCBBXKWVPZUT24NSJUW7J4J-PPDYAPUZRBP5OSJK")

# ✅ Load Models
models = {}

try:
    print("🔄 Loading Transformer model...")
    models["Transformer"] = load_model("tensorflow_model.h5", custom_objects=custom_objects)
    models["Transformer"].compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("✅ Transformer model loaded successfully!")

except Exception as e:
    print(f"❌ Transformer model loading failed: {e}")

try:
    print("🔄 Loading CNN model...")
    models["CNN"] = load_model("cnn_connect4.h5")
    print("✅ CNN model loaded successfully!")

except Exception as e:
    print(f"❌ CNN model loading failed: {e}")

# ✅ Global variable for selected model (default: Transformer)
current_model = models.get("Transformer", None)

@anvil.server.callable
def select_model(model_name):
    """Allows switching between Transformer and CNN model."""
    global current_model
    if model_name in models:
        current_model = models[model_name]
        return f"✅ Model switched to {model_name}"
    else:
        return "⚠️ Invalid model name. Choose either 'Transformer' or 'CNN'."

@anvil.server.callable


def predict_best_move(board_state):
    """Predict the best move using the selected model."""
    if current_model is None:
        raise ValueError("⚠️ No model loaded! Please check the model files.")

    board_array = np.array(board_state)

    # ✅ Format for CNN: (1, 6, 7, 2)
    if current_model == models["CNN"]:
        if board_array.shape[-1] == 3:  # 🔍 Ensure CNN input has exactly 2 channels
            board_array = board_array[:, :, :2]  # Keep only the first 2 channels

        board_array = board_array.reshape(1, 6, 7, 2)  # ✅ Ensure CNN expects (1, 6, 7, 2)
        print(f"📊 CNN Model: Adjusted board state shape: {board_array.shape}")

    # ✅ Format for Transformer: (1, 6, 7, 2) (Ensuring batch dimension)
    elif current_model == models["Transformer"]:
        if board_array.shape == (6, 7, 2):  
            board_array = np.expand_dims(board_array, axis=0)  # ✅ Add batch dimension: (1, 6, 7, 2)

        print(f"📊 Transformer Model: Adjusted board state shape: {board_array.shape}")

    try:
        prediction = current_model.predict(board_array)
        best_move = int(np.argmax(prediction))
        print(f"✅ Best move predicted: {best_move}")
        return {"best_move": best_move}
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        raise ValueError("Error occurred while predicting the best move.")


@anvil.server.callable
def test_function(): 
    return "Hello from local backend!"

print("✅ Backend connected to Anvil. Waiting for requests...")

anvil.server.wait_forever()
