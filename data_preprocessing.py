import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['booking_date', 'appointment_date'])

    # Feature engineering
    df['lead_days'] = (df['appointment_date'] - df['booking_date']).dt.days
    df['day_of_week'] = df['appointment_date'].dt.day_name()

    # Encode categorical columns
    le_gender = LabelEncoder()
    le_type = LabelEncoder()
    le_weather = LabelEncoder()
    le_day = LabelEncoder()

    df['gender'] = le_gender.fit_transform(df['gender'])
    df['appointment_type'] = le_type.fit_transform(df['appointment_type'])
    df['weather'] = le_weather.fit_transform(df['weather'])
    df['day_of_week'] = le_day.fit_transform(df['day_of_week'])

    df['no_show'] = df['no_show'].map({'Yes': 1, 'No': 0})

    # Features and target
    X = df[['age', 'gender', 'lead_days', 'appointment_type', 'previous_no_shows', 'weather', 'day_of_week']]
    y = df['no_show']

    return train_test_split(X, y, test_size=0.3, random_state=42)