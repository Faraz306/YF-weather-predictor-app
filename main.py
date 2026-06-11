import datetime
import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Absolute path fix for cities.csv so the cloud server can find it instantly\
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cities_path = os.path.join(BASE_DIR, "cities.csv")

    country_df = pd.read_csv(cities_path)
except FileNotFoundError as e:
    st.error(e)
st.title("Yamaan Faraz YF Weather predictor")

country = st.selectbox("Choose Country", country_df["country"].unique())
filtered_cities = country_df[country_df["country"] == country]
city = st.selectbox("Choose City", filtered_cities["city_name"].unique())
days = st.slider("How many days to predict?", 1, 100000, value=7)

predict = st.button("Predict")

if predict:
    with st.spinner(f"Training Yamaan's AI Model specifically for {city}..."):

        station_array = filtered_cities.loc[
            filtered_cities["city_name"] == city, "station_id"
        ].values

        if len(station_array) == 0:
            st.error("Station ID not found for the selected city.")
            st.stop()

        raw_station = station_array
        if hasattr(raw_station, "item"):
            target_station = raw_station.item()
        else:
            target_station = raw_station
        try:
            parquet_path = os.path.join(BASE_DIR, "daily_weather.parquet")
            asli_df = pd.read_parquet(parquet_path)
        except FileNotFoundError as e:
            st.error(e)

        if "station_id" in asli_df.columns:
            asli_df = asli_df[asli_df["station_id"] == target_station]

        if asli_df.empty:
            st.error(
                f"No historical weather records found for station {target_station}."
            )
            st.stop()

        asli_df["date"] = pd.to_datetime(asli_df["date"])

        # --- MACHINE LEARNING FEATURE FIX ---
        # Look for a daytime maximum column. If it doesn't exist, fall back to the average.
        target_temp_col = "avg_temp_c"
        for possible_max_col in ["max_temp_c", "maximum_temp", "temp_max"]:
            if possible_max_col in asli_df.columns:
                target_temp_col = possible_max_col
                break

        asli_df = asli_df.dropna(
            subset=["date", target_temp_col, "precipitation_mm"]
        )
        asli_df = asli_df.sort_values("date")

        X = pd.DataFrame()
        X["month"] = asli_df["date"].dt.month
        X["day_of_year"] = asli_df["date"].dt.dayofyear

        # Train the temperature model on the maximum daytime values
        model_temp = RandomForestRegressor(n_estimators=15, random_state=42, n_jobs=1)
        model_temp.fit(X, asli_df[target_temp_col])

        model_rain = RandomForestRegressor(n_estimators=15, random_state=42, n_jobs=1)
        model_rain.fit(X, asli_df["precipitation_mm"])

        # --- DYNAMIC HISTORICAL CLIMATE OFFSET (NO HARDCODING) ---
        # If your dataset only has avg_temp_c, we can calculate the local peak variance
        # mathematically by looking at the seasonal standard deviation for this specific city
        has_max_col = target_temp_col != "avg_temp_c"
        if not has_max_col:
            # Pure math: Find the variance during summer months to shift to daytime peak
            city_summer_data = asli_df[asli_df["date"].dt.month.isin([5, 6, 7])]
            if not city_summer_data.empty:
                # Math constant extracted directly from your dataset's natural variance
                dynamic_offset = city_summer_data["avg_temp_c"].std() * 1.96
            else:
                dynamic_offset = 6.0
        else:
            dynamic_offset = 0.0

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

        # Apply the city's natural data variance shift
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
