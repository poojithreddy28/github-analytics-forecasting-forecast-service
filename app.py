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
from prophet import Prophet
from flask_cors import CORS
from matplotlib.ticker import StrMethodFormatter
# Tensorflow (Keras & LSTM) related packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Import required storage package from Google Cloud Storage
from google.cloud import storage
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    # Facebook Prophet Forecast
    prophet_img = f"prophet_forecast_{series_col}_{repo_name}.png"
    prophet_url = BASE_IMAGE_PATH + prophet_img

    prophet_model = Prophet()
    prophet_model.fit(df)
    future = prophet_model.make_future_dataframe(periods=30)
    forecast = prophet_model.predict(future)

    # Get cutoff to separate history vs forecast
    cutoff_date = df['ds'].max()

    # Custom styled forecast plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot only historical part
    ax.plot(df['ds'], df['y'], 'bo-', label='Historical Data')

    # Plot only forecast part (starting after history ends)
    future_forecast = forecast[forecast['ds'] > cutoff_date]
    ax.plot(future_forecast['ds'], future_forecast['yhat'], 'orange', linestyle='--', label='Forecasted Data')
    ax.fill_between(future_forecast['ds'],
                    future_forecast['yhat_lower'],
                    future_forecast['yhat_upper'],
                    color='orange', alpha=0.3)

    # Dynamic title & labels
    pretty_label = "Created Issues" if "created" in series_col.lower() else "Closed Issues"
    ax.set_title(f"Forecast of {pretty_label} — {repo_name}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Number of {pretty_label}")
    ax.legend()

    fig.savefig(LOCAL_IMAGE_PATH + prophet_img)
    plt.close(fig)
    # ----------------------------------------------------------------
    # Statsmodels SARIMAX Forecast
    sarimax_img = f"sarimax_forecast_{series_col}_{repo_name}.png"
    sarimax_url = BASE_IMAGE_PATH + sarimax_img

    # Step 1: Reindex to daily frequency and fill gaps
    df_sarimax = df.set_index('ds').resample('D').sum().fillna(0)

    # Step 2: Fit SARIMAX model
    model = SARIMAX(
        df_sarimax['y'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    # Step 3: Forecast 30 future steps
    forecast_result = results.get_forecast(steps=30)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int()
    forecast_index = forecast_mean.index  # Dates auto-handled by model

    # Step 4: Plot everything
    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical
    ax.plot(df_sarimax.index, df_sarimax['y'], 'bo-', label='Historical Data')

    # Forecasted
    ax.plot(forecast_index, forecast_mean, 'orange', linestyle='--', label='Forecasted Data')

    # Confidence interval shading
    ax.fill_between(forecast_index,
                    forecast_ci.iloc[:, 0],
                    forecast_ci.iloc[:, 1],
                    color='orange', alpha=0.3)

    # Labels and title
    pretty_label = "Created Issues" if "created" in series_col.lower() else "Closed Issues"
    ax.set_title(f"STATSmodels SARIMAX Forecast of {pretty_label} — {repo_name}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Number of {pretty_label}")
    ax.legend()

    # Save
    fig.savefig(LOCAL_IMAGE_PATH + sarimax_img)
    plt.close(fig)

    # ----------------------------------------------------------------
    # upload all images to GCS
    bucket = client.get_bucket(BUCKET_NAME)
    for img in (loss_img, lstm_img, all_img, prophet_img, sarimax_img):
        blob = bucket.blob(img)
        blob.upload_from_filename(f"{LOCAL_IMAGE_PATH}{img}")

    # build response
    return jsonify({
        "model_loss_image_url": loss_url,
        "lstm_generated_image_url": lstm_url,
        "all_issues_data_image": all_url,
        "prophet_forecast_image_url": prophet_url,
        "sarimax_forecast_image_url": sarimax_url
    })

@app.route('/api/forecast/pulls', methods=['POST'])
def forecast_pulls():
    body       = request.get_json()
    pulls      = body["pulls"]
    repo_name  = body["repo"]

    # Step 1: Build DataFrame
    df = pd.DataFrame(pulls).groupby(['created_at'], as_index=False).count()[['created_at', 'pull_number']] \
                            .rename(columns={'created_at': 'ds', 'pull_number': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    # Step 2: Fill missing dates
    first_day = df['ds'].min()
    last_day  = df['ds'].max()
    all_days = pd.date_range(start=first_day, end=last_day, freq='D')

    Ys = np.zeros(len(all_days), dtype=float)
    for dt, y in zip(df['ds'], df['y']):
        idx = (dt - first_day).days
        Ys[idx] = y

    # Step 3: Scale and split
    scaler = MinMaxScaler(feature_range=(0, 1))
    Ys_scaled = scaler.fit_transform(Ys.reshape(-1,1))

    train_size = int(len(Ys_scaled) * 0.8)
    train, test = Ys_scaled[:train_size], Ys_scaled[train_size:]

    def create_dataset(arr, look_back=30):
        X, Y = [], []
        for i in range(len(arr) - look_back):
            X.append(arr[i:i+look_back, 0])
            Y.append(arr[i+look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 30
    X_train, Y_train = create_dataset(train, look_back)
    X_test,  Y_test  = create_dataset(test, look_back)

    # Step 4: Image paths
    BASE_IMAGE_PATH  = 'https://storage.googleapis.com/github-analytics-forecasting/'
    LOCAL_IMAGE_PATH = "static/images/"
    BUCKET_NAME      = 'github-analytics-forecasting'

    loss_img    = f"model_loss_pulls_{repo_name}.png"
    lstm_img    = f"lstm_generated_data_pulls_{repo_name}.png"
    all_img     = f"all_pulls_data_{repo_name}.png"
    prophet_img = f"prophet_forecast_pulls_{repo_name}.png"
    sarimax_img = f"sarimax_forecast_pulls_{repo_name}.png"

    loss_url    = BASE_IMAGE_PATH + loss_img
    lstm_url    = BASE_IMAGE_PATH + lstm_img
    all_url     = BASE_IMAGE_PATH + all_img
    prophet_url = BASE_IMAGE_PATH + prophet_img
    sarimax_url = BASE_IMAGE_PATH + sarimax_img

    # Step 5: LSTM Model (only if data is sufficient)
    if X_train.shape[0] > 0 and X_test.shape[0] > 0:
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

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

        # Model loss
        plt.figure(figsize=(8,4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.title('Model Loss For Pull Requests')
        plt.ylabel('Loss'); plt.xlabel('Epochs'); plt.legend()
        plt.savefig(LOCAL_IMAGE_PATH + loss_img)
        plt.close()

        # LSTM prediction vs true
        y_pred = model.predict(X_test)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(np.arange(len(Y_train)), Y_train, 'g', label="history")
        ax.plot(np.arange(len(Y_train), len(Y_train)+len(Y_test)), Y_test, marker='.', label="true")
        ax.plot(np.arange(len(Y_train), len(Y_train)+len(Y_test)), y_pred, 'r', label="prediction")
        ax.set_title('LSTM Generated Data For Pull Requests')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Scaled Pull Requests')
        ax.legend()
        fig.savefig(LOCAL_IMAGE_PATH + lstm_img)
        plt.close(fig)
    else:
        print(f"Skipping LSTM training for {repo_name}: not enough pull request data.")

    # Step 6: All Pulls Raw Timeline
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(all_days, Ys, marker='.', label='Pull Requests')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_title('All Pull Requests Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    fig.savefig(LOCAL_IMAGE_PATH + all_img)
    plt.close(fig)

    # Step 7: Facebook Prophet Forecast
    prophet_model = Prophet()
    prophet_model.fit(df)
    future = prophet_model.make_future_dataframe(periods=30)
    forecast = prophet_model.predict(future)

    cutoff_date = df['ds'].max()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['ds'], df['y'], 'bo-', label='Historical Pull Requests')
    future_forecast = forecast[forecast['ds'] > cutoff_date]
    ax.plot(future_forecast['ds'], future_forecast['yhat'], 'orange', linestyle='--', label='Forecasted Pull Requests')
    ax.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'], color='orange', alpha=0.3)
    ax.set_title(f"Facebook Prophet Forecast of Pull Requests — {repo_name}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Pull Requests")
    ax.legend()
    fig.savefig(LOCAL_IMAGE_PATH + prophet_img)
    plt.close(fig)

    # Step 8: SARIMAX Forecast
    df_sarimax = df.set_index('ds').resample('D').sum().fillna(0)

    model = SARIMAX(
        df_sarimax['y'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    forecast_result = results.get_forecast(steps=30)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int()
    forecast_index = forecast_mean.index

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_sarimax.index, df_sarimax['y'], 'bo-', label='Historical Pull Requests')
    ax.plot(forecast_index, forecast_mean, 'orange', linestyle='--', label='Forecasted Pull Requests')
    ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='orange', alpha=0.3)
    ax.set_title(f"STATSmodels SARIMAX Forecast of Pull Requests — {repo_name}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Pull Requests")
    ax.legend()
    fig.savefig(LOCAL_IMAGE_PATH + sarimax_img)
    plt.close(fig)

    bucket = client.get_bucket(BUCKET_NAME)
    for img in (loss_img, lstm_img, all_img, prophet_img, sarimax_img):
        local_path = os.path.join(LOCAL_IMAGE_PATH, img)
        if os.path.exists(local_path):
            blob = bucket.blob(img)
            blob.upload_from_filename(local_path)
        else:
            print(f"Skipping upload for {img}, file does not exist.")
    # Step 10: Return all image URLs
    return jsonify({
    "model_loss_image_url": loss_url if os.path.exists(os.path.join(LOCAL_IMAGE_PATH, loss_img)) else None,
    "lstm_generated_image_url": lstm_url if os.path.exists(os.path.join(LOCAL_IMAGE_PATH, lstm_img)) else None,
    "all_pulls_data_image": all_url,
    "prophet_forecast_image_url": prophet_url,
    "sarimax_forecast_image_url": sarimax_url
})

@app.route('/api/forecast/branches', methods=['POST'])
def forecast_branches():
    body = request.get_json()
    branches = body.get('branches')
    repo_name = body.get('repo')

    # Step 1: Build DataFrame
    df = pd.DataFrame(branches)
    
    today = pd.Timestamp.today()
    if df['created_at'].nunique() <= 1:
        df = df.sort_values('branch_name')
        df['created_at'] = [(today - pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(df))]

    df = df.groupby(['created_at'], as_index=False).count()[['created_at', 'branch_name']] \
           .rename(columns={'created_at': 'ds', 'branch_name': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    # Step 2: Validate the data (VERY IMPORTANT)
    if df.shape[0] < 2 or df['y'].sum() == 0:
        print(f"Skipping branch forecast for {repo_name}: not enough data.")
        return jsonify({
            "model_loss_image_url": None,
            "lstm_generated_image_url": None,
            "all_branches_data_image": None,
            "prophet_forecast_image_url": None,
            "sarimax_forecast_image_url": None
        })

    # Step 3: Fill missing dates
    first_day = df['ds'].min()
    last_day  = df['ds'].max()
    all_days  = pd.date_range(start=first_day, end=last_day, freq='D')

    Ys = np.zeros(len(all_days), dtype=float)
    for dt, y in zip(df['ds'], df['y']):
        idx = (dt - first_day).days
        Ys[idx] = y

    # Step 4: Scale and split
    scaler = MinMaxScaler(feature_range=(0, 1))
    Ys_scaled = scaler.fit_transform(Ys.reshape(-1, 1))

    train_size = int(len(Ys_scaled) * 0.8)
    train, test = Ys_scaled[:train_size], Ys_scaled[train_size:]

    def create_dataset(arr, look_back=30):
        X, Y = [], []
        for i in range(len(arr) - look_back):
            X.append(arr[i:i+look_back, 0])
            Y.append(arr[i+look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 30
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test   = create_dataset(test, look_back)

    # Step 5: Image paths
    BASE_IMAGE_PATH  = 'https://storage.googleapis.com/github-analytics-forecasting/'
    LOCAL_IMAGE_PATH = "static/images/"
    BUCKET_NAME      = 'github-analytics-forecasting'

    loss_img    = f"model_loss_branches_{repo_name}.png"
    lstm_img    = f"lstm_generated_data_branches_{repo_name}.png"
    all_img     = f"all_branches_data_{repo_name}.png"
    prophet_img = f"prophet_forecast_branches_{repo_name}.png"
    sarimax_img = f"sarimax_forecast_branches_{repo_name}.png"

    loss_url    = BASE_IMAGE_PATH + loss_img
    lstm_url    = BASE_IMAGE_PATH + lstm_img
    all_url     = BASE_IMAGE_PATH + all_img
    prophet_url = BASE_IMAGE_PATH + prophet_img
    sarimax_url = BASE_IMAGE_PATH + sarimax_img

    # Step 6: LSTM Model (only if enough data)
    if X_train.shape[0] > 0 and X_test.shape[0] > 0:
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

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
            verbose=0,
            shuffle=False
        )

        # Save Model Loss
        plt.figure(figsize=(8,4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.title('Model Loss For Branches')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(LOCAL_IMAGE_PATH + loss_img)
        plt.close()

        # Save LSTM Forecast
        y_pred = model.predict(X_test)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(np.arange(len(Y_train)), Y_train, 'g', label="history")
        ax.plot(np.arange(len(Y_train), len(Y_train)+len(Y_test)), Y_test, marker='.', label="true")
        ax.plot(np.arange(len(Y_train), len(Y_train)+len(Y_test)), y_pred, 'r', label="prediction")
        ax.set_title('LSTM Forecast for Branches')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Scaled Branches')
        ax.legend()
        fig.savefig(LOCAL_IMAGE_PATH + lstm_img)
        plt.close(fig)

    else:
        print(f"Skipping LSTM training for {repo_name}: not enough branch data.")

    # Step 7: Save Raw Branch Timeline
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(all_days, Ys, marker='.', label='Branches')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_title('All Branches Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    fig.savefig(LOCAL_IMAGE_PATH + all_img)
    plt.close(fig)

    # Step 8: Facebook Prophet Forecast
    try:
        prophet_model = Prophet()
        prophet_model.fit(df)
        future = prophet_model.make_future_dataframe(periods=30)
        forecast = prophet_model.predict(future)

        cutoff_date = df['ds'].max()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['ds'], df['y'], 'bo-', label='Historical Branches')
        future_forecast = forecast[forecast['ds'] > cutoff_date]
        ax.plot(future_forecast['ds'], future_forecast['yhat'], 'orange', linestyle='--', label='Forecasted Branches')
        ax.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'], color='orange', alpha=0.3)
        ax.set_title(f"Facebook Prophet Forecast of Branches — {repo_name}", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Branches")
        ax.legend()
        fig.savefig(LOCAL_IMAGE_PATH + prophet_img)
        plt.close(fig)
    except Exception as e:
        print(f"Skipping Prophet for {repo_name}: {str(e)}")

    # Step 9: SARIMAX Forecast
    df_sarimax = df.set_index('ds').resample('D').sum().fillna(0)

    model = SARIMAX(
        df_sarimax['y'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    forecast_result = results.get_forecast(steps=30)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int()
    forecast_index = forecast_mean.index

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_sarimax.index, df_sarimax['y'], 'bo-', label='Historical Branches')
    ax.plot(forecast_index, forecast_mean, 'orange', linestyle='--', label='Forecasted Branches')
    ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='orange', alpha=0.3)
    ax.set_title(f"STATSmodels SARIMAX Forecast of Branches — {repo_name}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Branches")
    ax.legend()
    fig.savefig(LOCAL_IMAGE_PATH + sarimax_img)
    plt.close(fig)

    # Step 10: Upload all to GCS
    bucket = client.get_bucket(BUCKET_NAME)
    for img in (loss_img, lstm_img, all_img, prophet_img, sarimax_img):
        blob = bucket.blob(img)
        try:
            blob.upload_from_filename(f"{LOCAL_IMAGE_PATH}{img}")
        except Exception as e:
            print(f"Skipping upload for {img}: {str(e)}")

    # Step 11: Return
    return jsonify({
        "model_loss_image_url": loss_url,
        "lstm_generated_image_url": lstm_url,
        "all_branches_data_image": all_url,
        "prophet_forecast_image_url": prophet_url,
        "sarimax_forecast_image_url": sarimax_url
    })
@app.route('/api/stars', methods=['POST'])
def plot_stars():
    body = request.get_json()
    repos = body["repos"]  # List of {"name": "repo-name", "stars": count}

    # Constants
    BASE_IMAGE_PATH = 'https://storage.googleapis.com/github-analytics-forecasting/'
    LOCAL_IMAGE_PATH = "static/images/"
    BUCKET_NAME = 'github-analytics-forecasting'

    # Convert to DataFrame and sort by stars descending
    df = pd.DataFrame(repos).sort_values(by="stars", ascending=False)

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(df['name'], df['stars'], color='royalblue', edgecolor='black')

    # Chart styling
    ax.set_title("GitHub Stars per Repository", fontsize=18, pad=15)
    ax.set_xlabel("Repository", fontsize=14)
    ax.set_ylabel("Number of Stars", fontsize=14)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    # Add star counts on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, color='black')

    fig.tight_layout()

    # Create a unique filename with timestamp to bust cache
    timestamp = int(time.time())
    bar_img = f"repo_stars_bar_chart_{timestamp}.png"
    bar_url = BASE_IMAGE_PATH + bar_img
    local_path = os.path.join(LOCAL_IMAGE_PATH, bar_img)

    # Save the image locally
    fig.savefig(local_path)
    plt.close(fig)

    # Upload to Google Cloud Storage
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(bar_img)
    blob.upload_from_filename(local_path)

    # Return the GCS image URL to frontend
    return jsonify({
        "star_bar_chart_url": bar_url
    })
@app.route('/api/forks', methods=['POST'])
def plot_forks():
    from matplotlib.ticker import StrMethodFormatter
    import time

    body = request.get_json()
    repos = body["repos"]  # List of {"name": "repo-name", "forks": count}

    BASE_IMAGE_PATH = 'https://storage.googleapis.com/github-analytics-forecasting/'
    LOCAL_IMAGE_PATH = "static/images/"
    BUCKET_NAME = 'github-analytics-forecasting'

    # Create DataFrame and sort by forks descending
    df = pd.DataFrame(repos).sort_values(by="forks", ascending=False)

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(df['name'], df['forks'], color='mediumseagreen', edgecolor='black')

    # Chart styling
    ax.set_title("GitHub Forks per Repository", fontsize=18, pad=15)
    ax.set_xlabel("Repository", fontsize=14)
    ax.set_ylabel("Number of Forks", fontsize=14)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    # Value labels on top
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, color='black')

    fig.tight_layout()

    # Save with unique timestamp to avoid cache
    timestamp = int(time.time())
    bar_img = f"repo_forks_bar_chart_{timestamp}.png"
    bar_url = BASE_IMAGE_PATH + bar_img
    local_path = os.path.join(LOCAL_IMAGE_PATH, bar_img)

    # Save locally
    fig.savefig(local_path)
    plt.close(fig)

    # Upload to GCS
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(bar_img)
    blob.upload_from_filename(local_path)

    return jsonify({
        "forks_bar_chart_url": bar_url
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081)