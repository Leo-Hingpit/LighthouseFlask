from flask import Flask, request, jsonify
from prophet import Prophet
import pandas as pd

app = Flask(__name__)

@app.route('/forecast', methods=['POST'])
def forecast():
    
    try:
        # Parse incoming data
        data = request.json
        if not isinstance(data, list) or not all('ds' in item and 'y' in item for item in data):
            print("Invalid data format received:", data)
            return jsonify({"error": "Invalid data format. Expecting a JSON array with 'ds' and 'y' keys."}), 400

        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df = df.dropna(subset=['ds', 'y'])

        # Log the incoming data
        print("Full historical data:\n", df)

        # Ensure sufficient data for warmup and forecast
        if len(df) < 3:
            print("Insufficient rows for warmup and forecasting.")
            return jsonify({"error": "Insufficient rows in historical data for fitting."}), 400

        # Split warmup and forecast data
        warmup_cutoff = len(df) // 3
        warmup_data = df.iloc[:warmup_cutoff]
        forecast_data = df.iloc[warmup_cutoff:]

        print("Warmup data:\n", warmup_data)
        print("Forecast data:\n", forecast_data)

        # Initialize and fit the model with forecast data
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(forecast_data[['ds', 'y']])

        # Make future predictions for overlapping and extra months
        future = model.make_future_dataframe(periods=90, freq='D')  # Extend by 3 months
        forecast = model.predict(future)

        # Aggregate forecast to monthly predictions
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        monthly_forecast = (
            forecast.set_index('ds')['yhat']
            .resample('MS')
            .mean()
            .reset_index()
            .to_dict(orient='records')
        )

        # Combine historical and forecast data
        result = []
        for item in forecast_data.to_dict(orient='records'):
            result.append({
                'ds': item['ds'].strftime('%Y-%m-%d'),
                'y': item['y'],
                'isHistorical': True
            })

        for forecast_item in monthly_forecast:
            # Check if the forecasted month overlaps with historical data
            forecast_date = forecast_item['ds'].strftime('%Y-%m-%d')
            if forecast_date in forecast_data['ds'].dt.strftime('%Y-%m-%d').values:
                result.append({
                    'ds': forecast_date,
                    'y': forecast_item['yhat'],
                    'isHistorical': False  # Distinguish forecast for overlapping months
                })
            elif forecast_date > df['ds'].max().strftime('%Y-%m-%d'):
                result.append({
                    'ds': forecast_date,
                    'y': forecast_item['yhat'],
                    'isHistorical': False  # Forecast for future months
                })

        print("Final result:\n", result)
        return jsonify(result)

    except Exception as e:
        print(f"Error in forecasting: {str(e)}")
        return jsonify({"error": "Failed to forecast data."}), 500

@app.route('/')
def home():
    return "Flask App is running!", 200

@app.route('/event_forecast', methods=['POST'])
def event_forecast():

    try:
        # Parse incoming data
        data = request.json
        if not isinstance(data, list) or not all('ds' in item and 'y' in item and 'event_type' in item for item in data):
            print("Invalid data format received:", data)
            return jsonify({"error": "Invalid data format. Expecting a JSON array with 'ds', 'y', and 'event_type' keys."}), 400

        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df = df.dropna(subset=['ds', 'y'])

        # Log the incoming data
        print("Full historical data:\n", df)

        # Ensure sufficient data for warmup and forecast
        if len(df) < 3:
            print("Insufficient rows for warmup and forecasting.")
            return jsonify({"error": "Insufficient rows in historical data for fitting."}), 400

        # Split data into warmup and forecast comparison
        warmup_cutoff = len(df) // 3
        warmup_data = df.iloc[:warmup_cutoff]
        forecast_comparison_data = df.iloc[warmup_cutoff:]

        # Initialize and fit the model
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(df[['ds', 'y']])

        # Make future predictions
        future = model.make_future_dataframe(periods=90, freq='D')  # Extend by 3 months
        forecast = model.predict(future)

        # Aggregate forecast to monthly predictions
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        monthly_forecast = (
            forecast.set_index('ds')['yhat']
            .resample('MS')
            .mean()
            .reset_index()
        )

        # Combine warmup, forecast comparison, and future forecasts
        result = []
        for item in data:
            result.append({
                'ds': item['ds'],
                'y': item['y'],
                'event_type': item['event_type'],
                'isHistorical': True
            })

        for _, forecast_item in monthly_forecast.iterrows():
            forecast_date = forecast_item['ds'].strftime('%Y-%m-%d')
            if forecast_date in forecast_comparison_data['ds'].dt.strftime('%Y-%m-%d').values:
                result.append({
                    'ds': forecast_date,
                    'y': forecast_item['yhat'],
                    'event_type': data[0]['event_type'],  # Use the first event_type as Flask processes one at a time
                    'isHistorical': False
                })
            elif forecast_date > df['ds'].max().strftime('%Y-%m-%d'):
                result.append({
                    'ds': forecast_date,
                    'y': forecast_item['yhat'],
                    'event_type': data[0]['event_type'],
                    'isHistorical': False
                })

        print("Final result:\n", result)
        return jsonify(result)

    except Exception as e:
        print(f"Error in forecasting: {str(e)}")
        return jsonify({"error": "Failed to forecast data."}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
