import datetime
import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Absolute path fix for files so the cloud server can find them instantly
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cities_path = os.path.join(BASE_DIR, "cities.csv")
    country_df = pd.read_csv(cities_path)
except FileNotFoundError as e:
    st.error(f"Critical System File Missing: {e}")
    st.stop()

st.title("Yamaan Faraz YF Weather predictor")

# App Form Selectors
country = st.selectbox("Choose Country", country_df["country"].unique())
filtered_cities = country_df[country_df["country"] == country]
city = st.selectbox("Choose City", filtered_cities["city_name"].unique())

# Caps slider presentation internally to save layout generation buffer memory
days = st.slider("How many days to predict?", 1, 365, value=7)

predict = st.button("Predict")

if predict:
    with st.spinner(f"Training Yamaan's AI Model specifically for {city}..."):

        station_array = filtered_cities.loc[
            filtered_cities["city_name"] == city, "station_id"
        ].values

        if len(station_array) == 0:
            st.error("Station ID not found for the selected city.")
            st.stop()

        target_station = station_array.item() if hasattr(station_array, "item") else station_array

        # 🛠️ THE CRITICAL BULLETPROOF RAM FIX: Stream and chunk parquet row-groups
        parquet_path = os.path.join(BASE_DIR, "daily_weather.parquet")
        try:
            import pyarrow.parquet as pq

            # Connect to parquet file metadata without reading the raw data into RAM
            parquet_file = pq.ParquetFile(parquet_path)
            chunk_list = []

            # Loop through individual internal chunks one by one
            for i in range(parquet_file.num_row_groups):
                # Pull ONLY the exact columns needed for the ML model to save space
                df_chunk = parquet_file.read_row_group(
                    i,
                    columns=['station_id', 'date', 'avg_temp_c', 'max_temp_c', 'maximum_temp', 'temp_max',
                             'precipitation_mm']
                ).to_pandas()

                # Filter rows inside the tiny chunk memory buffer immediately
                filtered_chunk = df_chunk[df_chunk['station_id'] == target_station]
                if not filtered_chunk.empty:
                    chunk_list.append(filtered_chunk)

            if len(chunk_list) > 0:
                asli_df = pd.concat(chunk_list, ignore_index=True)
            else:
                asli_df = pd.DataFrame()

        except Exception as e:
            st.error(f"Error reading dataset or matching columns: {e}")
            st.stop()

        if asli_df.empty:
            st.error(f"No historical weather records found for station {target_station}.")
            st.stop()

        asli_df["date"] = pd.to_datetime(asli_df["date"])

        # Dynamic validation mapping for maximum temperature tracking
        target_temp_col = "avg_temp_c"
        for possible_max_col in ["max_temp_c", "maximum_temp", "temp_max"]:
            if possible_max_col in asli_df.columns:
                target_temp_col = possible_max_col
                break

        asli_df = asli_df.dropna(subset=["date", target_temp_col, "precipitation_mm"])
        asli_df = asli_df.sort_values("date")

        X = pd.DataFrame()
        X["month"] = asli_df["date"].dt.month
        X["day_of_year"] = asli_df["date"].dt.dayofyear

        # Lightweight capped models to prevent memory leaks or kernel drops
        model_temp = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42, n_jobs=1)
        model_temp.fit(X, asli_df[target_temp_col])

        model_rain = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42, n_jobs=1)
        model_rain.fit(X, asli_df["precipitation_mm"])

        # --- FIXED DYNAMIC CLIMATE OFFSET ---
        has_max_col = target_temp_col != "avg_temp_c"
        if not has_max_col:
            city_summer_data = asli_df[asli_df["date"].dt.month.isin([5, 6, 7])]
            if not city_summer_data.empty:
                dynamic_offset = city_summer_data["avg_temp_c"].std() * 1.96
            else:
                dynamic_offset = 6.0
        else:
            dynamic_offset = 0.0

    # Moving prediction steps cleanly outside the data ingestion loop block
    with st.spinner(f"Calculating local trends for {city}..."):
        today = datetime.date.today()
        future_dates = [today + datetime.timedelta(days=i) for i in range(days)]

        future_df = pd.DataFrame({"date": future_dates})
        future_df["date"] = pd.to_datetime(future_df["date"])

        future_X = pd.DataFrame()
        future_X["month"] = future_df["date"].dt.month
        future_X["day_of_year"] = future_df["date"].dt.dayofyear

        predicted_temps_raw = model_temp.predict(future_X)
        predicted_rains = model_rain.predict(future_X)

        predicted_temps = predicted_temps_raw + dynamic_offset

        weather_conditions = []
        for temp, rain in zip(predicted_temps, predicted_rains):
            if rain > 5.0:
                weather_conditions.append("⛈️ Stormy")
            elif rain > 2.5:
                weather_conditions.append("🌧️ Rainy")
            elif temp >= 40.0:
                weather_conditions.append("🔥 Scorching")
            elif temp >= 32.0:
                weather_conditions.append("☀️ Sunny")
            elif temp < 15.0:
                weather_conditions.append("❄️ Cold")
            else:
                weather_conditions.append("🌤️ Pleasant")

        final_table = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
            "Weather": weather_conditions,
            "Temp (°C)": [round(t, 1) for t in predicted_temps],
            "mm": [round(r, 2) for r in predicted_rains],
        })

        st.subheader(f"Weather Forecast Timeline for {city}, {country}")
        st.dataframe(final_table, use_container_width=True, hide_index=True)
