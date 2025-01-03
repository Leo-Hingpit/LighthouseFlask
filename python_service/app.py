from flask import Flask, request, jsonify
from prophet import Prophet
import pandas as pd
import numpy as np
app = Flask(__name__)
from flask_cors import CORS
CORS(app)
@app.route('/electricity-forecast', methods=['POST'])
def electricity_forecast():
    try:
        # Parse incoming data
        data = request.json
        if not isinstance(data, list) or not all('ds' in item and 'y' in item for item in data):
            return jsonify({"error": "Invalid data format. Expecting a JSON array with 'ds' and 'y' keys."}), 400

        # Prepare the data
        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df = df.dropna(subset=['ds', 'y'])

        if len(df) < 3:
            return jsonify({"error": "Insufficient data for forecasting."}), 400

        # Apply adjusted scaling
        scale_factor = 1.2  # Adjust this factor as needed for better scaling
        df['y_scaled'] = df['y'] / scale_factor

        # Train Prophet using scaled data
        model = Prophet(
            seasonality_mode='multiplicative',  # Use multiplicative seasonality for proportional trends
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(df[['ds', 'y_scaled']].rename(columns={'y_scaled': 'y'}))

        # Forecast into the future
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        # Rescale the forecasted values back
        forecast['yhat_rescaled'] = forecast['yhat'] * scale_factor

        # Prepare output
        forecast_data = forecast[['ds', 'yhat_rescaled']].rename(columns={'yhat_rescaled': 'y'}).to_dict(orient='records')

        # Combine historical and forecast data
        combined_data = df[['ds', 'y']].to_dict(orient='records') + [
            {**item, 'isHistorical': False} for item in forecast_data[len(df):]
        ]

        return jsonify(combined_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Parse incoming data
        data = request.json
        if not isinstance(data, list) or not all('ds' in item and 'y' in item for item in data):
            return jsonify({"error": "Invalid data format. Expecting a JSON array with 'ds' and 'y' keys."}), 400

        # Prepare the data
        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df = df.dropna(subset=['ds', 'y'])
        if len(df) < 3:
            return jsonify({"error": "Insufficient data for forecasting."}), 400

        # Initialize and fit the model
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(df[['ds', 'y']])

        # Make future predictions
        future = model.make_future_dataframe(periods=90, freq='D')
        forecast = model.predict(future)

        # Post-process forecast
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        forecast['yhat'] = forecast['yhat'].clip(lower=0)  # Clamp negative values to 0
        monthly_forecast = (
            forecast.set_index('ds')['yhat']
            .resample('MS')
            .mean()
            .reset_index()
            .to_dict(orient='records')
        )

        # Combine historical and forecast data
        result = [
            {"ds": item['ds'].strftime('%Y-%m-%d'), "y": item['y'], "isHistorical": True}
            for item in df.to_dict(orient='records')
        ]
        result += [
            {"ds": item['ds'].strftime('%Y-%m-%d'), "y": item['yhat'], "isHistorical": False}
            for item in monthly_forecast
        ]

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to forecast: {str(e)}"}), 500


@app.route('/event_forecast', methods=['POST'])
def event_forecast():
    try:
        # Parse incoming data
        data = request.json
        warmup_data = data.get('warmupData', [])
        forecast_data = data.get('forecastData', [])
        print("Received data in Flask:", data)

        if not warmup_data:
            return jsonify({"error": "Warmup data missing."}), 400

        # Combine all data for comprehensive analysis
        all_data = warmup_data + forecast_data
        all_data_df = pd.DataFrame(all_data)
        all_data_df['ds'] = pd.to_datetime(all_data_df['ds'], errors='coerce')
        all_data_df = all_data_df.dropna(subset=['ds', 'y'])  # Clean invalid rows
        all_data_df['month'] = all_data_df['ds'].dt.month

        if all_data_df.empty:
            print("All data is empty.")
            return jsonify({"error": "All data is empty."}), 400

        # Calculate monthly averages for each event type
        monthly_avg = all_data_df.groupby(['month', 'event_type'])['y'].mean().to_dict()
        print("Monthly averages (seasonality) calculated from all data:", monthly_avg)

        # Calculate valid months for each event type
        valid_months = all_data_df.groupby('event_type')['month'].unique().to_dict()
        print("Valid months for each event type based on historical data:", valid_months)

        # Calculate overall averages for fallback
        overall_avg = all_data_df.groupby('event_type')['y'].mean().to_dict()
        print("Overall averages for each event type:", overall_avg)

        # Find the latest historical date
        latest_historical_date = all_data_df[all_data_df['isHistorical'] == True]['ds'].max()
        print("Latest historical date across all data:", latest_historical_date)

        # Generate forecasts for the next 3 months
        future_forecasts = []
        for i in range(1, 4):  # Next 3 months
            future_date = latest_historical_date + pd.DateOffset(months=i)
            forecast_month = future_date.month

            for event_type in all_data_df['event_type'].unique():
                # Skip event types if the forecast month is not in valid months
                if forecast_month not in valid_months.get(event_type, []):
                    continue

                # Use seasonality data if available
                predicted_value = monthly_avg.get((forecast_month, event_type), None)

                # Blend seasonality and overall average (50/50 weight)
                if predicted_value is not None:
                    overall_value = overall_avg.get(event_type, 0)
                    predicted_value = 0.5 * predicted_value + 0.5 * overall_value

                # Fallback to overall average if no seasonality data exists
                if predicted_value is None or pd.isna(predicted_value):
                    predicted_value = overall_avg.get(event_type, 0)

                # Cap predictions based on historical peaks
                historical_peak = all_data_df[all_data_df['event_type'] == event_type]['y'].max()
                if predicted_value > historical_peak:
                    predicted_value = historical_peak

                # Eliminate noise for minimal predictions
                if predicted_value < 1.0:
                    predicted_value = 0

                # Add the forecast entry with integer values
                future_forecasts.append({
                    'ds': future_date.strftime('%Y-%m-%d'),
                    'y': int(round(predicted_value)) if predicted_value is not None else 0,
                    'event_type': event_type,
                    'isHistorical': False
                })

        print("Future forecasts for the next 3 months with refined seasonality enforcement:", future_forecasts)

        # Process historical (warmup) and forecast data
        warmup_df = pd.DataFrame(warmup_data)
        forecast_results = []
        for item in forecast_data:
            try:
                forecast_month = pd.to_datetime(item['ds']).month
                event_type = item['event_type']
                if forecast_month in valid_months.get(event_type, []):
                    predicted_value = monthly_avg.get((forecast_month, event_type), 0)
                else:
                    predicted_value = 0
                forecast_results.append({
                    'ds': item['ds'],
                    'y': int(round(predicted_value)),
                    'event_type': event_type,
                    'isHistorical': False
                })
            except Exception as item_error:
                print(f"Error processing forecast item: {item}. Error: {item_error}")
                continue

        # Combine results
        result = []
        if not warmup_df.empty:
            warmup_df['ds'] = pd.to_datetime(warmup_df['ds'])
            for _, row in warmup_df.iterrows():
                result.append({
                    'ds': row['ds'].strftime('%Y-%m-%d'),
                    'y': row['y'],
                    'event_type': row['event_type'],
                    'isHistorical': True
                })
        result.extend(forecast_results)
        result.extend(future_forecasts)

        # Log final results
        print("Final forecast result:\n", result)
        return jsonify(result)

    except Exception as e:
        print(f"Error in forecasting: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500




@app.route('/')
def home():
    return "Flask App is running!", 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
