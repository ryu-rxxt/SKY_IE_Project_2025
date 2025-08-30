from io_utils import *

def holiday_flag(date, city, path = "data/holiday_lookup.csv"):
    country = city_to_country(city)
    holiday = pd.read_csv(path, parse_dates = ["date"])
    holiday = holiday[["country", "date"]]
    date = pd.to_datetime(date)
    
    holiday = holiday.loc[
        (holiday["country"] == country) &
        (holiday["date"] == date)
    ]

    return not bool(holiday.empty)

def weather_flag(date, city, path = "data/weather.csv"):
    country = city_to_country(city)
    weather = pd.read_csv(path, parse_dates = ["date"])
    weather = weather[["date", "country", "rain_mm", "snow_mm", "wind_speed_avg", "cloud_cover"]]
    date = pd.to_datetime(date)
    
    subset = weather.loc[
        (weather["date"] == date) & 
        (weather["country"] == country)
    ]
    rec = subset.iloc[0]
    
    return bool(rec["rain_mm"] >= 45.7 or 
        rec["snow_mm"] >= 3.85 or 
        rec["wind_speed_avg"] >= 13.46 or 
        rec["cloud_cover"] == 100
    )
    
def oil_flag(date, path = "data/oil_price.csv"):
    oil_data = fill_na(path)
    date = pd.to_datetime(date)
    if date < pd.Timestamp("2018-01-08"): # 비교군이 존재하지 않을 경우 함수 종료
        return False

    curr_mon = date - timedelta(days = date.weekday())
    curr_sun = curr_mon + timedelta(days = 6)
    prev_mon = curr_mon - timedelta(weeks = 1)
    prev_sun = curr_sun - timedelta(weeks = 1)

    curr_week = oil_data[(oil_data["date"] >= curr_mon) & (oil_data["date"] <= curr_sun)]
    prev_week = oil_data[(oil_data["date"] >= prev_mon) & (oil_data["date"] <= prev_sun)]
    this_avg = curr_week["brent_usd"].mean()
    prev_avg = prev_week["brent_usd"].mean()

    return bool(this_avg >= prev_avg * 1.05)

def machine_failure_flag(date, city, path = "data/machine_failure_log.csv"):
    factory = city_to_id(city, "factory")
    log = pd.read_csv(path, parse_dates = ["start_date", "end_date"])
    log = log[["factory", "start_date", "end_date"]]
    date = pd.to_datetime(date)
    
    mask = (
        (log["factory"]   == factory) &
        (log["start_date"] <= date) &
        (log["end_date"]   >= date)
    )

    return bool(mask.any())

def month_end_flag(dt: pd.Timestamp) -> bool:
    return (dt + pd.Timedelta(days=1)).month != dt.month
