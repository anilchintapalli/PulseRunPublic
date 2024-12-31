'''
This streamlit application works without an API to 
connect to Garmin Connect, allowing for heart rate/biometric data predicton.

The steps are:
    User connects to Garmin Connect with their credentials
    The valid activities are obtained based on the specificed time range
    The user inputs their desired feature data
    After running a neural network to predict the heart rate/biometric features, these predictions are returned
'''
import streamlit as st
from garmin_connect import GarminConnector
from neural_network import prepare_new_route,train_and_predict
from datetime import date, timedelta
#UI setup
st.title("Running Performance Predictor")
with st.form("prediction_form"):
    st.subheader("Garmin Connect Credentials")
    username = st.text_input("Garmin Connect Email")
    password = st.text_input("Garmin Connect Password", type="password")
    st.subheader("Activity Date Range")
    col1, col2 = st.columns(2)
    with col1:
        default_start = date.today() - timedelta(days=365)
        start_date = st.date_input('Start Date', value=default_start)
    with col2:
        end_date = st.date_input("End Date", value=date.today())
    
    st.header("Route Features")
    col1,col2 = st.columns(2)
    with col1:
        distance = st.number_input('Distance (mi)', min_value=0.0, step=0.1)
        average_pace = st.number_input('Average Pace (min/mi)', min_value=0.0, step=0.1)
    with col2:
        elevation_gain = st.number_input('Elevation Gain (ft)', min_value=0.0, step=1.0)
        elevation_loss = st.number_input('Elevation Loss (ft)', min_value=0.0, step=1.0)
    
    submit_button = st.form_submit_button("Predict Performance")

if submit_button:
    
    #Upon the user clicking the submit button, the program obtains the Garmin connect 
    #activities if this possible.
    
    if not username or not password:
        st.error("Enter both a username and password")
        st.stop()
    
    status_placeholder = st.empty()
    with status_placeholder:
        with st.spinner("Connecting to Garmin..."):
            try:
                garmin = GarminConnector(username, password)
                status_placeholder.empty()
                st.success("Logged in to Garmin Connect successfully")
            except ValueError as e:
                status_placeholder.empty()
                st.error(f"Error logging in to Garmin Connect: {str(e)}")
                st.stop()
    

    with st.spinner("Obtaining Garmin Connect data and predicting"):
        try:
            
            activities_df = garmin.fetch_running_activities(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
                    
                    
            new_route = prepare_new_route(
                distance,
                average_pace,
                elevation_gain,
                elevation_loss)
                        
            predicted_biometrics, predicted_avg_hr, predicted_max_hr = train_and_predict(
                    activities_df,
                    new_route)
            st.success("Prediction successful!")

            st.subheader("Predicted Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Average Heart Rate", f"{predicted_avg_hr:.1f} BPM")
            with col2:
                st.metric("Predicted Maximum Heart Rate", f"{predicted_max_hr:.1f} BPM")
            st.subheader("Predicted Biometrics")
            biometric_names = [
                'Calories', 
                'Activity Training Load', 
                'Avg Power (W)', 
                'Avg Cadence (steps/min)', 
                'Avg Vertical Oscillation (cm)', 
                'Avg Stride Length (cm)', 
                'Avg Ground Contact Time (ms)'
            ]
            for name, value in zip(biometric_names, predicted_biometrics.tolist()):
                st.write(f"{name}: {value:.2f}")
            st.write(f"{len(activities_df)} activities used") 
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")