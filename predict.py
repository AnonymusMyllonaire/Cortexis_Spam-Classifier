import joblib

# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("📩 Spam SMS Classifier")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("Enter a message: ")
    
    if user_input.lower() in ["quit", "exit"]:
        print("👋 Exiting... Goodbye!")
        break
    
    # Transform input and predict
    transformed_input = vectorizer.transform([user_input])
    prediction = model.predict(transformed_input)[0]
    
    if prediction == 1:
        print("🚨SPAM Message\n")
    else:
        print("✅NOT SPAM Message\n")
