import json
import uvicorn as uvicorn
from fastapi import Body, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from numpy import log as ln
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/get-tickers')
async def get_tickers():
    return {'HMC': 'Автобизнес',
            'GELYY': 'Автобизнес',
            'NFLX': 'Медиа',
            'AAPL': 'IT',
            'CDI.PA': 'Мода',
            'MSFT': 'IT',
            'ADBE': 'IT',
            'FIVE.IL': 'Продажи',
            'TSLA': 'IT',
            'DIS': 'Медиа',
            'AMZN': 'Продажи',
            'AMD': 'IT',
            'PYPL': 'IT',
            'JOBY': 'Другое',
            'INTC': 'IT',
            'SBER.ME': 'Финансы',
            'GAZP.ME': 'Нефть',
            '005930.KS': 'IT',
            'XOM': 'Нефть',
            'GOOG': 'IT',
            'NVDA': 'IT',
            'V': 'Финансы',
            'UNH': 'Медицина',
            'JNJ': 'Медицина',
            'WMT': 'Продажи',
            'JPM': 'Финансы',
            'PG': 'Другое',
            'MA': 'Финансы',
            'NESN.SW': 'Еда',
            'KO': 'Еда',
            'PEP': 'Еда',
            'ORCL': 'IT',
            'ROG.SW': 'Медицина',
            'OR.PA': 'Мода',
            'ASML': 'IT',
            'RMS.PA': 'Мода',
            'BABA': 'Продажи',
            'COST': 'Продажи',
            'MCD': 'Еда',
            'PFE': 'Медицина',
            'SHEL': 'Нефть',
            'TM': 'Автобизнес',
            'NKE': 'Мода',
            'DHR': 'Другое',
            'PM': 'Другое',
            'UL': 'Еда',
            'MS': 'Финансы',
            'RY': 'Финансы',
            'SBUX': 'Еда',
            'BAC': 'Финансы'}


@app.post('/evaluate')
async def evaluate(ticker: str=Body(...),start_date: str=Body(...), n_forecast: int=Body(...)):
    end_date = "2022-01-01"
    pd.options.mode.chained_assignment = None
    tf.random.set_seed(0)

    # download the data
    df = yf.download(tickers=[ticker], start=start_date, end=end_date)
    df['target'] = ln(df.Close).diff()
    # тут сделаем таргет переменную - первую разность логарифмов, и ее отдадим для создания модели
    y = df['target'].fillna(method='ffill').iloc[1:]
    df = df.iloc[1:]
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    X = []
    Y = []
    for i in range(n_lookback, len(y) - n_forecast + 1):
        # в модель идут только значения по таргет переменной
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])
    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=32, verbose=0)
    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input
    # sequence
    X_ = X_.reshape(1, n_lookback, 1)
    Y_ = model.predict(X_).reshape(-1, 1)
    # это наши готовые предсказанные значения на 30 дней
    Y_ = scaler.inverse_transform(Y_)

    # переводим в обычные цены
    result = []
    # мы потом этот ряд удалим, потому что его создают в следующем блоке в  строчке
    # df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1], но он нужен чтобы взять result[-1])
    result.append(ln(df[['Close']].values[-1][0]))
    for i in range(Y_.shape[0]):
        result.append(result[-1] + Y_[i][0])
    result = np.exp(result)

    # и график уже создаем по ценам из Close и results
    df_past = df[['Close']].reset_index()
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast + 1)
    df_future['Forecast'] = result.flatten()
    df_future['Actual'] = np.nan
    # удалим первый ряд
    df_future = df_future.iloc[1:]
    results = pd.concat([df_past, df_future], ignore_index=True)
    print(results.tail(30))
    results['Date'] = results['Date'].dt.strftime('%Y-%m-%d')
    
    return Response(results.to_json(orient="records"), media_type="application/json")


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0")
