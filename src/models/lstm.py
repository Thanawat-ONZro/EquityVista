import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Note: Caller should normalize features and target values (e.g. MinMaxScaler)
# before passing sequences to train_lstm, as raw Close values can be large
# and high-variance. Scaling logic lives in the caller/results notebook.


def train_lstm(X_train, y_train, X_test, y_test, epochs=50):
    tf.random.set_seed(42)

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    val_size = int(len(X_train) * 0.1)
    if val_size > 0:
        Xv = X_train[-val_size:]
        yv = y_train[-val_size:]
        Xt = X_train[:-val_size]
        yt = y_train[:-val_size]
    else:
        Xv, yv = X_train, y_train
        Xt, yt = X_train, y_train

    history = model.fit(
        Xt,
        yt,
        validation_data=(Xv, yv),
        epochs=epochs,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    pred = model.predict(X_test, verbose=0).flatten()

    return {
        "history": history,
        "model": model,
        "test_pred": pred,
        "test_truth": y_test,
    }
