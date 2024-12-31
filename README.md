# PulseRun
The goal of this project is to create a program that can predict a user's predicted biometric features and average/max heart rate for a future run based on these route features: Distance, Pace, Elevation gained, Elevation lost. This prediction is made using a machine learning model trained on the user's historical running data through Garmin Connect.
## Authors
Anil Chintapalli, Manya Nallagangu
## Libraries and Equipment
* Tensorflow - This machine learning library allows us to train and run models. 
* pandas - Library for managing dataframes
* numpy - Library for array manipulation
* Flask - This framework will be used to build the API.
* Streamlit - This framework will be used to build the UI
* Keras - Library for neural network implementation
* Garmin Connect- This API wrapper allows access to a user's Garmin Connect activity data given they provide their associated email, password, start date, and end date. 
* SDV - Library to generate synthetic and evaluate the quality of that data
* Scikit Learn - Library for standardization of features before training
# Feature List
- [ ] Deployment on desktop UI
- [ ] Model makes synthetic data from historical running data
- [ ] Machine learning model preforms on four inputted individual run parameters
- [ ] Outputs future run characteristics for the user (predicted biometric features and average/max heart rate)
# Priority List
**1. Obtain and format data**
| Status | Date | Description |
| :----: | :-----: | :---------: |
| Completed | 9/11/2024 | Download personal Garmin connect data to local machine
| Completed | 9/18/2024 | Format the necessary running activity data as a CSV

**2. Use sdv to generate personal synthetic data**
| Status | Date | Description |
| :----: | :-----: | :---------: |
| Completed | 11/3/2024 | Install sdv
| Completed | 11/4/2024 | Use sdv to generate 1000 runs for regression

**4. Train/test machine learning model**
| Status | Date | Description |
| :----: | :-----: | :---------: |
| Completed | 12/6/2024 | Neural Network with seperately optimized hyperparameters for best prediction. Solid predictions (within 10 MAE)

**5. API/UI Development**
| Status | Date | Description |
| :----: | :-----: | :---------: |
| Completed | 11/27/2024 | Include way to connect with another user's garmin data (Garmin Connect API Wrapper)
| Completed | 11/28/2024 | Create UI and API
| Completed | 12/9/2024 | Integrated machine learning model with the API (didn't work extremely well? Problems with Flask disconnecting)
| Completed | 12/12/2024 | Workaround that removes API dependancy to have the user's credentials go directly to Garmin Connect from the UI

**6. API/App Development**
| Status | Date | Description |
| :----: | :-----: | :---------: |
| In progress | 12/20/2024 | Begin development of App in Dart
| In progress | 12/20/2024 | Reinstate backend API for remote usage
| In progress | 12/22/2024 | Explore methods for reliable, secure remotely-hosted API
