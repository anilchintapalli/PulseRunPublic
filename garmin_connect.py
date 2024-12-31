import pandas as pd
from datetime import datetime
from garminconnect import (
    Garmin,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)
from clean_synthesize import filter_incomplete_runs, remove_outliers, generate_synthetic_data 

class GarminConnector:
    def __init__(self,username:str,password:str):
        '''
        Initializes the user's connection to Garmin Connect

        Args:
            username: str that represents the user's Garmin Connect email
            password: str that represents the user's Garmin Connect password
        
        Errors it raises (which are reflected in the streamlit UI):
            ValueError if failed connection:
                Internet connection problems, too many requests, invalid credentials
        
        '''
        self.username = username
        self.password = password
        try:
            self.client = Garmin(self.username,self.password)
            self.client.login()
        except GarminConnectConnectionError:
            raise ValueError("Unable to connect to Garmin Connect. Check internet connection.")
        except GarminConnectTooManyRequestsError:
            raise ValueError("Too many requests to Garmin Connect; try again in a bit.")
        except Exception as e:
            raise ValueError("Your Garmin Connect credentials were invalid. Check for typos.")
    def fetch_running_activities(self,start_date:str,end_date:str) -> pd.DataFrame:
        '''
        Obtain the running activities from Garmin Connect and process them

        Args:
            start_dart: str that represents the start date for activity retrieval (YYYY-MM-DD)
            end_dart: str that represents the end date for activity retrieval (YYYY-MM-DD)
        Returns:
            pandas dataframe: Processed dataframe with a default 1000 rows of synthetic data
        Raises:
            ValueError:
                -If end date is in the future
                -If start data if after end date
                -If no activities were found in the desired time range        
        
        '''
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
            
        if end > datetime.now():
            raise ValueError("End date cannot be in the future")
        if start > end:
            raise ValueError("Start date must be before end date")
    # get activities
        activities = self.client.get_activities_by_date(start, end)
        if not activities:
            raise ValueError("No activities found in that date range")
    
    
    #filter out non running activities
        running_activities = [
            activity for activity in activities
            if activity["activityType"]["typeKey"] == "running"]

    # convert df to proper format
        df = pd.DataFrame(running_activities)
        df = filter_incomplete_runs(df)
        df = remove_outliers(df)
        df = generate_synthetic_data(df)

        return df