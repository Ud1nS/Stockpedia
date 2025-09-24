from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

CloseModel = joblib.load('App/close_predict.pkl')
FutureModel = joblib.load('App/future_predict.pkl')

stocks = [
    'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'TLKM.JK',
    'ASII.JK', 'MDKA.JK', 'BBNI.JK', 'BOGA.JK', 'ADRO.JK',
    'NATO.JK', 'ANTM.JK', 'BHAT.JK', 'UNTR.JK',
    'ARTO.JK', 'BRIS.JK', 'SMMA.JK', 'MEDC.JK', 'TCPI.JK'
]

data = pd.read_csv('stock_data_reshaped.csv')
data['Date'] = pd.to_datetime(data['Date'])
mapping_ticker = pd.read_pickle('App/mapping_ticker.pkl')


def forecast_to_date(data, mapping_ticker, future_predict_model, close_predict_model, target_date_str, ticker_symbol):
    target_date = pd.to_datetime(target_date_str)

    last_known = data[data['Ticker'] == ticker_symbol].copy()
    last_known.set_index('Date', inplace=True)
    last_known = last_known.sort_index()

    current_date = last_known.index.max()
    ticker_encoded = mapping_ticker[ticker_symbol][0]

    while current_date < target_date:
        features = [last_known['Close'].iloc[-i] for i in range(1, 8)]
        X_next = pd.DataFrame([features], columns=[f'lag_{i}' for i in range(1, 8)])
        X_next['Ticker'] = ticker_encoded

        y_next = future_predict_model.predict(X_next)[0]

        days = (current_date - last_known.index.min()).days

        close_parameter = pd.DataFrame(
            [[days, y_next[0], y_next[1], y_next[2], y_next[3], ticker_encoded]],
            columns=['Days', 'Open', 'High', 'Low', 'Volume', 'Ticker_encoded']
        )
        close = close_predict_model.predict(close_parameter)[0]

        new_row = {
            'Open': y_next[0],
            'High': y_next[1],
            'Low': y_next[2],
            'Close': close,
            'Volume': y_next[3]
        }

        next_date = current_date + pd.Timedelta(days=1)
        new_row_df = pd.DataFrame([new_row], index=[next_date])
        last_known = pd.concat([last_known, new_row_df])
        current_date = next_date

    return last_known


def plot_forecasted_close(data, mapping_ticker, future_predict_model, close_predict_model, target_date_str, ticker_symbol):
    forecasted_df = forecast_to_date(
        data=data,
        mapping_ticker=mapping_ticker,
        future_predict_model=future_predict_model,
        close_predict_model=close_predict_model,
        target_date_str=target_date_str,
        ticker_symbol=ticker_symbol
    )

    target_date = pd.to_datetime(target_date_str)
    start_date = target_date - pd.Timedelta(days=30)
    plot_df = forecasted_df[(forecasted_df.index >= start_date) & (forecasted_df.index <= target_date)]

    plt.figure(figsize=(10, 5))
    plt.plot(plot_df.index, plot_df['Close'], marker='o', linestyle='-')
    plt.title(f"{ticker_symbol} Close Price Forecast: {start_date.date()} to {target_date.date()}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64


img_url_gradient = 'gradient.png'
img_url_logo_img = 'Logo_image.png'
img_url_logo_only = 'Logo.png'

@app.route("/")
def home():
    return render_template("home_index.html", img_url1 = img_url_gradient, img_url2 = img_url_logo_img, img_url3 = img_url_logo_only)


@app.route("/news")
def news():
    return render_template("news_index.html", img_url1 = img_url_gradient, img_url2 = img_url_logo_img, img_url3 = img_url_logo_only)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    selected_stock = None
    selected_date = None
    plot_img = None

    if request.method == "POST":
        selected_stock = request.form.get("stock")
        selected_date = request.form.get("target_date")

        if selected_stock and selected_date:
            plot_img = plot_forecasted_close(
                data=data,
                mapping_ticker=mapping_ticker,
                future_predict_model=FutureModel,
                close_predict_model=CloseModel,
                target_date_str=selected_date,
                ticker_symbol=selected_stock
            )

    return render_template("predict_index.html",
                           stocks=stocks,
                           selected_stock=selected_stock,
                           selected_date=selected_date,
                           plot_img=plot_img,
                           img_url1 = img_url_gradient, img_url2 = img_url_logo_img, img_url3 = img_url_logo_only)


if __name__ == "__main__":
    app.run(debug=True)