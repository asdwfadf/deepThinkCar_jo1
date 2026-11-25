from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    ZeroPadding2D,
    AveragePooling2D,
    Flatten,
    Dropout,
    Dense
)
from tensorflow.keras.optimizers import Adam

def nvidia_model(self):
    model = Sequential(name='Nvidia_Model_Bottleneck_V4')

    model.add(Input(shape=(66, 200, 3)))

    # ==========================================
    # Bottleneck Block 1
    # ==========================================
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(8, (3, 3), strides=(1, 1),
                     activation='relu'))
    # (B, 66, 200, 8)

    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(8, (3, 3), strides=(2, 2),
                     activation='relu'))
    # (B, 33, 100, 8)

    model.add(Conv2D(24, (1, 1), activation='relu'))
    # (B, 33, 100, 24)

    # ==========================================
    # Bottleneck Block 2
    # ==========================================
    model.add(Conv2D(8, (1, 1), activation='relu'))
    # (B, 33, 100, 8)

    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(8, (3, 3), strides=(1, 1),
                     activation='relu'))
    # (B, 33, 100, 8)

    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(8, (3, 3), strides=(2, 2),
                     activation='relu'))
    # (B, 17, 50, 8)

    model.add(Conv2D(36, (1, 1), activation='relu'))
    # (B, 17, 50, 36)

    # ==========================================
    # Bottleneck Block 3
    # ==========================================
    model.add(Conv2D(8, (1, 1), activation='relu'))
    # (B, 17, 50, 8)

    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(8, (3, 3), strides=(1, 1),
                     activation='relu'))
    # (B, 17, 50, 8)

    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(8, (3, 3), strides=(2, 2),
                     activation='relu'))
    # (B, 9, 25, 8)

    model.add(Conv2D(48, (1, 1), activation='relu'))
    # (B, 9, 25, 48)

    # ==========================================
    # Convolution Layer 4
    # ==========================================
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # (B, 9, 25, 64)

    model.add(Dropout(0.2))

    # ==========================================
    # Convolution Layer 5
    # ==========================================
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # (B, 9, 25, 64)

    # ==========================================
    # Average Pooling
    # ==========================================
    model.add(AveragePooling2D(
        pool_size=(3, 9),
        strides=(3, 8),
        padding='valid'
    ))
    # (B, 3, 3, 64)

    # ==========================================
    # Fully Connected Layers
    # ==========================================
    model.add(Flatten())
    # (B, 576)

    model.add(Dropout(0.2))

    model.add(Dense(100, activation='relu'))
    # (B, 100)

    model.add(Dense(1))
    # (B, 1)

    # ==========================================
    # Optimizer & Loss
    # ==========================================
    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='mse', optimizer=optimizer)

    return model