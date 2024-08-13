import pickle
from sklearn.preprocessing import StandardScaler
from logisticreg import load_data, LogisticRegression

X, y = load_data("Heart Attack.csv")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_scaled, y)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(logistic_model, model_file)
