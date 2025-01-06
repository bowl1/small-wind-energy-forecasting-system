The Orkney Islands, located in Northern Scotland, have significant wind and marine energy resources. Local farms can utilize wind power for energy generation. This app aims to use weather forecasting data to predict energy production for Orkney.

- How to run: 
  - The program includes two main functionalities: model generation and prediction using the model.
  - First run: "conda env create -f conda_env.yaml" to build the enviromant after download the code.<br/>
  Then run "conda activate windpower_prediction_Orkney" to activate the environment.
  - Model generation: Run the following command in the terminal:  "mlflow run . --experiment-name windModel" or "python trainModel.py"
  - Prediction using the model: Run the following command in the terminal:  python predictWindpower.py
  
- Data Sources:
  - Orkney’s renewable power generation: Sourced from Scottish and Southern Electricity Networks (SSEN).
  - Weather forecasts for Orkney: Sourced from the UK Met Office.

- Model Training:
The project uses traditional models from scikit-learn to train wind power prediction models. Specifically, it includes:
  - Linear Regression
  - XGBoost
  - Random Forest

- Training Features:
  - Windspeed
  - Wind direction
  - Month
  - Day of the week
  - Hour

- Observations from Model Training:
After training 6 models for several times by adjusting data interval among 30 to 365 days, timeseries split from 3 to 10 folds, the Random Forest Model preforms better in 180 days and 5 folds.

- Prediction Model:
The selected model for predictions is Random Forest, configured with:
  - 180-day interval
  - 5 folds

Each run automatically retrieves the past 180 days of Orkney’s renewable power generation and weather forecast data for training. Using data from the past 180 days, it predicts wind power generation for the next 30 days.

- Main Results:

<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
windspeed and windpower visualizaion <br/>
<img src="draw/wind_speed_and_power_generation_2023.png" alt="Description" width="400"> 
  
wind direction and windpower visualizaion <br/>
<img src="draw/wind_direction and power generation in 2023.png" alt="Description" width="400">   

metrics <br/>
<img src="draw/metrics.png" alt="Description" width="400">  

model trained <br/>
<img src="draw/model_predictions-180d&5f.png" alt="Description" width="400">  

prediction with the best model<br/>
<img src="draw/wind_power_prediction_30days.png" alt="Description" width="400">    
</div>



