🚗 Used Car Price Prediction

A machine learning web app that predicts the second-hand price of a car based on its features. The user selects details such as brand, model year, mileage, and fuel type, and the model instantly estimates the market price.

🚀 Live Demo

👉 Try the app here

📸 Screenshot

Show Image

🛠️ Tech Stack


Python – core programming language
scikit-learn – machine learning (Random Forest + Pipeline & OneHotEncoder)
pandas – data processing and cleaning
Streamlit – web interface and deployment


📊 About the Project

The model is trained on real used-car market data from Türkiye (~53,500 listings, April 2026). It predicts prices using features such as city, brand, fuel type, transmission, body type, drivetrain, model year, mileage, engine size, and horsepower.

Categorical text features (brand, fuel type, etc.) are transformed into numerical form using a scikit-learn Pipeline with OneHotEncoder, combining preprocessing and the model into a single, reproducible workflow. Extreme (erroneous) price values are filtered out before training to keep the model robust.

The model is trained in memory when the app starts and cached afterwards, making the app fast and independent of library versions.

💻 Run Locally

bashpip install -r requirements.txt
streamlit run app.py

📁 Data Source

Turkey Used Car Prices (Kaggle) – used-car listing data from Türkiye.

👤 Author

Yiğit Efe USTA – Computer Engineering Student
