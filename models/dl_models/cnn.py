from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping

def init_conv1D(embedding_size,
                      vocab_size,
                      input_length,
                      mask_zero=True):


    model = models.Sequential()

    #Model embedding
    model.add(layers.Embedding(
        input_dim=vocab_size+1,
        input_length=input_length,
        output_dim=embedding_size,
        mask_zero=mask_zero,
    ))

    #Model architecture
    model.add(layers.Conv1D(128,kernel_size=3, padding='same'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(64,kernel_size=3, padding='same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(32,kernel_size=3, padding='same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(16,kernel_size=3, padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(6, activation='softmax'))

    #Model compile
    model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

    return model


def fit_conv1D(
    model,
    X_train,
    y_train,
    batch_size=16,
    epochs=1000,
    validation_split=0.2,
    patience=10,
    restore_best_weights=True):

    #Early stopping definition
    es = EarlyStopping(
    patience=patience,
    restore_best_weights=restore_best_weights
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size ,
        epochs=epochs,
        validation_split=validation_split,
        callbacks = [es]
    )

    return history

if __name__ == '__main__':
    pass
