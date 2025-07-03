# 🏥 Appointment No-Show Predictor (with Streamlit Interface)

This project is an AI/ML-based system that predicts whether a patient is likely to miss a scheduled medical appointment, and suggests cost-effective intervention strategies to help reduce no-shows.

The project also includes a **Streamlit-based web interface** to interactively explore the data, view model predictions, and analyze intervention strategies.

---

## 📌 Problem Statement

Missed medical appointments lead to wasted time, increased costs, and reduced care efficiency in hospitals. This system helps hospitals:

- Predict high-risk no-show patients in advance
- Take preventive actions like reminders or rescheduling
- Visualize data insights interactively

---

## 📊 Data Features Used

The model is trained on synthetic data designed with realistic correlations between:

- **Demographics**: Age, Gender
- **Appointment Info**: Type, Gap between booking & appointment
- **History**: Count of previous no-shows
- **External Factors**: Weather on appointment day
- **Temporal Patterns**: Day of the week

---

## 📁 Dataset Structure

File: `appointments_high_accuracy.csv`  
Contains 200 records and the following fields:

| Column               | Description                              |
|----------------------|------------------------------------------|
| `patient_id`         | Unique identifier                        |
| `age`                | Age of the patient                       |
| `gender`             | Gender (M/F)                             |
| `booking_date`       | Date the appointment was booked          |
| `appointment_date`   | Date of the appointment                  |
| `appointment_type`   | General, Dental, Eye, or Cardiology      |
| `previous_no_shows`  | Count of past missed appointments        |
| `weather`            | Weather on appointment day               |
| `no_show`            | Target label (Yes/No)                    |

---

## 🧠 Model Features

These features are used for training:

- `age`
- `gender` (encoded)
- `lead_days` (days between booking and appointment)
- `appointment_type` (encoded)
- `previous_no_shows`
- `weather` (encoded)
- `day_of_week` (encoded)

---

## ⚙️ System Workflow

1. **Data Preprocessing** (`data_preprocessing.py`)
   - Reads and encodes the dataset
   - Derives features like lead days and weekday

2. **Model Training** (`model.py`)
   - Trains a Random Forest Classifier
   - Evaluates using accuracy on test data

3. **Intervention Logic** (`intervention.py`)
   - Suggests actions based on no-show probability:
     - 🔔 High Risk: Call & reschedule
     - 📩 Medium Risk: Send SMS reminder
     - ✅ Low Risk: No action needed

4. **Streamlit Web App** (`streamlit_app.py`)
   - Shows dataset preview
   - Displays predictions with intervention suggestions
   - Summarizes total high/medium/low-risk patients

---
