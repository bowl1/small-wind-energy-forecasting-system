The Orkney Islands, located in Northern Scotland, have significant wind and marine energy resources. Local farms can utilize wind power for energy generation. This app aims to use weather forecasting data to predict energy production for Orkney.

- How to run: 
  - The program includes two main functionalities: model generation and prediction using the model.
  - First run: "pip install -r requirements.txt" to build the enviromant after download the code. 
  - Model generation: Run the following command in the terminal:  mlflow run . --experiment-name windModel
  - Prediction using the model: Run the following command in the terminal:  python Assignment2-train_model.py
  
- Data Sources:
  - Orkney’s renewable power generation: Sourced from Scottish and Southern Electricity Networks (SSEN).
  - Weather forecasts for Orkney: Sourced from the UK Met Office.

- Model Training:
The project uses traditional models from scikit-learn and deep learning models like RNN to train wind power prediction models. Specifically, it includes:
  - Polynomial Regression (degree = 2)
  - XGBoost
  - Long Short-Term Memory (LSTM)

- Training Features:
  - Windspeed
  - Wind direction
  - Month
  - Day of the week
  - Hour

- Observations from Model Training:
After several rounds of training by adjusting data intervals (90 to 365 days) and window sizes (30 to 90 days):
  - LSTM performs better with longer data intervals and larger window sizes (365 days/90 days).
  - XGBoost performs better with shorter data intervals (180 days).

- Prediction Model:
The selected model for predictions is XGBoost, configured with:
  - 180-day interval
  - 30-day window size
Each run automatically retrieves the past 180 days of Orkney’s renewable power generation and weather forecast data for training. Using data from the past 30 days, it predicts wind power generation for the next 30 days.

- Main Results:

![wind_speed_and_power_generation_2023](https://github.com/user-attachments/assets/5534e353-5278-4ef7-ba49-02d32434e6a4)
![wind_direction and power generation in 2023](https://github.com/user-attachments/assets/76e77e65-d599-46cc-9088-756b6a512066)
![train_model_Best(Fetch180d + window30d)](https://github.com/user-attachments/assets/b7e2e06b-a2a3-4922-9674-c07cdb36f543)
![wind_power_prediction -XGBOOST](https://github.com/user-attachments/assets/04ab38f1-7f47-4c7b-bdb1-f10a1ed1a83f)




