from data_preprocessing import load_and_preprocess_data
from model import train_model, evaluate_model
from intervention import suggest_intervention

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data("appointments.csv")
    model = train_model(X_train, y_train)
    probs = evaluate_model(model, X_test, y_test)

    for i, prob in enumerate(probs):
        action = suggest_intervention(prob)
        print(f"Patient {i+1}: No-show probability = {prob:.2f} → {action}")

if __name__ == "__main__":
    main()