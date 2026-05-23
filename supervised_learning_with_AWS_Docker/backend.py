import anvil.server
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


# 1. SETUP ANVIL CONNECTION
anvil.server.connect("server_2QKKQ4T5TXVKPWYSM4FK5RVM-VK7MPXXIUQYS6V6K")

# Define Paths
CNN_PATH = '/app_data/cnn_connect4.pt'           
TRANSFORMER_PATH = '/app_data/transformer_connect4.keras' 


# 2. DEFINE CUSTOM KERAS LAYERS

@tf.keras.utils.register_keras_serializable()
class PositionalIndex(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        return tf.range(seq_length, dtype=tf.int32)
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class ClassTokenIndex(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.zeros((batch_size, 1), dtype=tf.int32)
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class GetItem(tf.keras.layers.Layer):
    def __init__(self, index=0, **kwargs):
        super().__init__(**kwargs)
        self.index = index
    def __call__(self, *args, **kwargs):
        def clean(obj):
            if tf.is_tensor(obj) or (hasattr(obj, '__class__') and 'KerasTensor' in obj.__class__.__name__): return obj
            if isinstance(obj, (list, tuple)): 
                cleaned = [clean(x) for x in obj]
                return type(obj)([c for c in cleaned if c is not None])
            return None
        return super().__call__(*clean(list(args)), **kwargs)
    def call(self, inputs, *args, **kwargs):
        if isinstance(inputs, list):
            return inputs[self.index]
        return inputs[:, self.index]
    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config


# 3. CNN CLASS
class Connect4CNN(nn.Module):
    def __init__(self):
        super(Connect4CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 7)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.mean([2, 3]) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# 4. LOAD MODELS

print("--- Starting Server ---")

# Load Transformer
transformer_model = None
try:
    if os.path.exists(TRANSFORMER_PATH):
        custom_objects = {
            'PositionalIndex': PositionalIndex,
            'ClassTokenIndex': ClassTokenIndex,
            'GetItem': GetItem
        }
        # While register_keras_serializable handles most cases, 
        # using custom_object_scope is the safest way to load .keras files with custom layers.
        with tf.keras.utils.custom_object_scope(custom_objects):
            transformer_model = tf.keras.models.load_model(TRANSFORMER_PATH)
        print("Transformer Model loaded successfully!")
    else:
        print(f"File not found: {TRANSFORMER_PATH}")
except Exception as e:
    print(f"Error loading Transformer: {e}")

# Load CNN
cnn_model = None
try:
    if os.path.exists(CNN_PATH):
        cnn_model = Connect4CNN()
        cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=torch.device('cpu')))
        cnn_model.eval()
        print("CNN Model loaded successfully!")
    else:
        print(f"File not found: {CNN_PATH}")
except Exception as e:
    print(f"Error loading CNN: {e}")



# 5. ANVIL FUNCTIONS

@anvil.server.callable
def get_transformer_move(board_state):
    if transformer_model is None:
        print("Transformer not loaded")
        return -1

    # Convert to numpy (float32 to match training)
    board_array = np.array(board_state, dtype=np.float32)

    # DEBUG: Count pieces
    count_p1 = np.sum(board_array == 1)
    count_p2 = np.sum(board_array == -1)
    print(f"DEBUG: Found {count_p1} P1 pieces and {count_p2} P2 pieces.")

    # Build EXACT same 2-channel format used during training
    # Channel 0 = Player 1 pieces
    # Channel 1 = Player -1 pieces
    input_board = np.zeros((6, 7, 2), dtype=np.float32)

    input_board[:, :, 0] = (board_array == 1).astype(np.float32)
    input_board[:, :, 1] = (board_array == -1).astype(np.float32)

    # DEBUG: Check signal strength
    print(f"DEBUG: Input Tensor Sum: {np.sum(input_board)}")
    # Should equal (count_p1 + count_p2)

    # Reshape EXACTLY like training
    input_board = input_board.reshape(1, 42, 2)

    try:
        preds = transformer_model.predict(input_board, verbose=0)

        # DEBUG: Raw output
        print(f"DEBUG: Raw Preds: {np.round(preds[0], 3)}")

        # 4️⃣ Mask invalid moves (top row full)
        valid_moves = [c for c in range(7) if board_state[0][c] == 0]

        masked_preds = np.full(7, -np.inf)
        for c in valid_moves:
            masked_preds[c] = preds[0][c]

        move = int(np.argmax(masked_preds))
        print(f"Transformer chose: {move}")

        return move

    except Exception as e:
        print(f"Transformer Error: {e}")
        return -1


@anvil.server.callable
def get_cnn_move(board_state):
    if cnn_model is None:
        print("CNN not loaded")
        return -1
        
    board_array = np.array(board_state)
    
    # CNN expects 2 channels: [MyPieces, EnemyPieces]
    input_board = np.zeros((1, 2, 6, 7), dtype=np.float32)
    input_board[0, 0, :, :] = (board_array == 1).astype(np.float32)
    input_board[0, 1, :, :] = (board_array == -1).astype(np.float32)

    input_tensor = torch.tensor(input_board)
    
    try:
        with torch.no_grad():
            preds = cnn_model(input_tensor)
            
        probs = preds.numpy()[0]
        
        valid_moves = [c for c in range(7) if board_array[0][c] == 0]
        masked_preds = np.full(7, -np.inf)
        for c in valid_moves:
            masked_preds[c] = probs[c]
            
        move = int(np.argmax(masked_preds))
        print(f"CNN chose: {move}")
        return move
    except Exception as e:
        print(f"CNN Error: {e}")
        return -1

print("Server is running and waiting for requests...")
anvil.server.wait_forever()