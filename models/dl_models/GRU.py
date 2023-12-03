import pandas as pd
import numpy as np

from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from data.data import clean_raw, create_sampled_df

def GRU_model(vocab_size, X_train_padded,y_train_enc):

    model = models.Sequential()
    model.add(layers.Embedding(
    input_dim = vocab_size + 1,
    output_dim = 75,
    mask_zero = True
    ))
    model.add(layers.GRU(64))

    model.add(layers.Dense(10,activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(6, activation='softmax'))

    # Model summary
    model.summary()

    # Compile the model
    model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

    es = EarlyStopping(
        patience = 10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_padded,
        y_train_enc,
        batch_size = 16,
        epochs = 1000,
        validation_split = 0.2,
        callbacks = [es]
    )
    return history
