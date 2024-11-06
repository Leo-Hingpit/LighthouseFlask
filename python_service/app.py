from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.json
    if not data or not isinstance(data, list):
        return jsonify({"error": "Invalid data format. Expecting a JSON array."}), 400

    try:
        # Load and preprocess data
        df = pd.DataFrame(data)
        if len(df.dropna()) < 2:
            return jsonify([])

        # Convert 'ds' to datetime and group by month
        df['ds'] = pd.to_datetime(df['ds'])
        df['month'] = df['ds'].dt.to_period("M")
        grouped_df = df.groupby('month').agg({'y': 'sum'}).reset_index()
        grouped_df['ds'] = grouped_df['month'].dt.to_timestamp()
        grouped_df = grouped_df[['ds', 'y']]

        # Fit the Prophet model
        model = Prophet()
        model.fit(grouped_df)

        # Make future predictions
        future = model.make_future_dataframe(periods=90, freq='D')
        forecast = model.predict(future)

        # Aggregate forecasted data by month
        forecast['month'] = forecast['ds'].dt.to_period("M")
        monthly_forecast = forecast.groupby('month')['yhat'].sum().reset_index()
        forecasted_data = [
            {
                "ds": row['month'].start_time.strftime('%Y-%m-%d'),
                "yhat": row['yhat']
            }
            for _, row in monthly_forecast.iterrows()
        ]

        # Prepare historical data
        historical_data = [
            {
                "ds": row['ds'].strftime('%Y-%m-%d'),
                "y": row['y'],
                "isHistorical": True
            }
            for _, row in grouped_df.iterrows()
        ]

        # Find the latest historical date
        latest_date = max(pd.to_datetime(d['ds']) for d in historical_data)
        latest_month = latest_date.to_period("M")

        # Filter forecast data to ensure it starts after the last historical month
        filtered_forecast = [
            item for item in forecasted_data if pd.to_datetime(item['ds']).to_period("M") > latest_month
        ]

        # Combine historical and forecast data
        result = historical_data + filtered_forecast
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
