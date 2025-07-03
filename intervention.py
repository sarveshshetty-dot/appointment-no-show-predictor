def suggest_intervention(probability):
    if probability > 0.7:
        return "🔔 Call and reschedule or offer remote consult"
    elif probability > 0.4:
        return "📩 Send SMS reminder"
    else:
        return "✅ No action needed"