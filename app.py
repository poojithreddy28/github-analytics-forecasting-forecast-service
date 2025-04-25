# app.py

from flask import Flask, jsonify, request, make_response
import os
from dateutil import *
from datetime import timedelta
import pandas as pd
import numpy as np
import time

# ←— force non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from flask_cors import CORS

# Tensorflow (Keras & LSTM) related packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Import required storage package from Google Cloud Storage
from google.cloud import storage

# Init flask app
app = Flask(__name__)
CORS(app)

# Init Google Cloud Storage client
client = storage.Client()

def build_preflight_response():
    resp = make_response()
    resp.headers.add("Access-Control-Allow-Origin", "*")
    resp.headers.add("Access-Control-Allow-Headers", "Content-Type")
    resp.headers.add("Access-Control-Allow-Methods", "PUT, GET, POST, DELETE, OPTIONS")
    return resp

def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods", "PUT, GET, POST, DELETE, OPTIONS")
    return response


@app.route('/api/forecast', methods=['POST'])
def forecast():
    body       = request.get_json()
    issues     = body["issues"]
    series_col = body["type"]
    repo_name  = body["repo"]

    # build a simple df of (ds, y)
    df = pd.DataFrame(issues) \
           .groupby([series_col], as_index=False) \
           .count()[[series_col, 'issue_number']] \
           .rename(columns={ series_col: 'ds', 'issue_number': 'y' })

    # ensure datetime
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])

    # ----------------------------------------------------------------
    # 1) build a DAILY date index from min-->max, zero-fill missing days
    first_day = df['ds'].min()
    last_day  = df['ds'].max()
    all_days  = pd.date_range(start=first_day, end=last_day, freq='D')

    Ys = np.zeros(len(all_days), dtype=float)
    for dt, y in zip(df['ds'], df['y']):
        idx = (dt - first_day).days
        Ys[idx] = y

    # 2) scale + reshape for LSTM
    scaler = MinMaxScaler(feature_range=(0,1))
    Ys_scaled = scaler.fit_transform(Ys.reshape(-1,1))

    # train/test split 80/20
    train_size = int(len(Ys_scaled)*0.8)
    train, test = Ys_scaled[:train_size], Ys_scaled[train_size:]

    def create_dataset(arr, look_back=30):
        X, Y = [], []
        for i in range(len(arr)-look_back):
            X.append(arr[i:i+look_back,0])
            Y.append(arr[i+look_back,0])
        return np.array(X), np.array(Y)

    look_back = 30
    X_train, Y_train = create_dataset(train, look_back)
    X_test,  Y_test  = create_dataset(test,  look_back)

    # reshape for [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # ----------------------------------------------------------------
    # build & fit LSTM
    model = Sequential([
        LSTM(100, input_shape=(1, look_back)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(
        X_train, Y_train,
        epochs=20,
        batch_size=70,
        validation_data=(X_test, Y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
        verbose=1,
        shuffle=False
    )

    # ----------------------------------------------------------------
    # image names & paths
    BASE_IMAGE_PATH    = 'https://storage.googleapis.com/github-analytics-forecasting/'
    LOCAL_IMAGE_PATH   = "static/images/"
    BUCKET_NAME        = 'github-analytics-forecasting'

    loss_img  = f"model_loss_{series_col}_{repo_name}.png"
    lstm_img  = f"lstm_generated_data_{series_col}_{repo_name}.png"
    all_img   = f"all_issues_data_{series_col}_{repo_name}.png"

    loss_url  = BASE_IMAGE_PATH + loss_img
    lstm_url  = BASE_IMAGE_PATH + lstm_img
    all_url   = BASE_IMAGE_PATH + all_img

    # plot & save Model Loss
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss For ' + series_col)
    plt.ylabel('Loss'); plt.xlabel('Epochs'); plt.legend()
    plt.savefig(LOCAL_IMAGE_PATH + loss_img)
    plt.close()

    # predict on test set
    y_pred = model.predict(X_test)

    # plot & save LSTM generated vs true
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(np.arange(len(Y_train)), Y_train,           'g', label="history")
    ax.plot(np.arange(len(Y_train), len(Y_train)+len(Y_test)),
            Y_test,     marker='.', label="true")
    ax.plot(np.arange(len(Y_train), len(Y_train)+len(Y_test)),
            y_pred, 'r',         label="prediction")
    ax.set_title('LSTM Generated Data For ' + series_col)
    ax.set_xlabel('Time Steps'); ax.set_ylabel('Scaled Issues'); ax.legend()
    fig.savefig(LOCAL_IMAGE_PATH + lstm_img)
    plt.close(fig)

    # plot & save all issues (original, unscaled) as time series
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(all_days, Ys, marker='.', label='issues')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_title('All Issues Data'); ax.set_xlabel('Date'); ax.set_ylabel('Count')
    fig.savefig(LOCAL_IMAGE_PATH + all_img)
    plt.close(fig)

    # ----------------------------------------------------------------
    # upload to GCS
    bucket = client.get_bucket(BUCKET_NAME)
    for img in (loss_img, lstm_img, all_img):
        blob = bucket.blob(img)
        blob.upload_from_filename(f"{LOCAL_IMAGE_PATH}{img}")

    # build response
    return jsonify({
        "model_loss_image_url": loss_url,
        "lstm_generated_image_url": lstm_url,
        "all_issues_data_image": all_url
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081)