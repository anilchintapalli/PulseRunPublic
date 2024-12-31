#Developed in conjunction with: https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
import os
#supress tensorflow logs for readability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings

#supress other warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

#import machine learning/data handling modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BiometricPredictor:
    def __init__(self,route_features_number=4,biometric_features_number=7):
        '''
        Initializes the BiometricPredictor based on the number of route and biometric features

        Args:
            route_features_number: An optional argument that represents the number of route features (int; defaults to 4)
            biometric_features_number: An optional argument that represents the number of biometric features (int; defaults to 7)
        
        '''
        self.route_scaler = StandardScaler()
        self.biometric_scaler = StandardScaler()
        self.route_feature_count = route_features_number
        self.biometric_feature_count = biometric_features_number
        self.model = None
    def _create_model(self):
        route = Input(shape=(self.route_feature_count,),name='route')
        #optimized hyperparameters found through previous tuning
        # use functional API because this model takes multiple inputs/outputs
        x = Dense(192, activation='relu')(route)
        x = Dropout(0.0)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)

        predicted_biometric_features = Dense(self.biometric_feature_count,activation='linear',
                                             name='predicted_biometric_features')(x)
        model = Model(inputs=route,outputs=predicted_biometric_features)
        #compile model using Adam; mse for training loss and mae to understand performance
        model.compile(optimizer=Adam(learning_rate=0.008569905521819712),
            loss='mean_squared_error',
            metrics=['mae'])
        return model
    
    #train with optimized number of epochs
    def train(self,route_features,biometric_features,epochs=6):
        '''
        Train the model for biometric prediction.
        Args:
            route_features: numpy array of the route features for training
            biometric_features: numpy array of the target biometric features for training
            epochs: number of training epochs; initially set to 6 (int)
        
        Returns:
            None: this method trains the neural network
        
        
        '''
        scaled_route = self.route_scaler.fit_transform(route_features)
        scaled_biometrics = self.biometric_scaler.fit_transform(biometric_features) 

        #split the data for training and testing
        route_training, route_testing, biometric_training, biometric_testing = train_test_split(scaled_route,scaled_biometrics,test_size=0.2,random_state=42)
        self.model = self._create_model()
        self.model.fit(route_training,biometric_training,
                       validation_data=(route_testing,biometric_testing),epochs=epochs,verbose=0)
    
    def biometric_prediction(self,route):
        '''
        Predicts the biometric features for a given route

        Args:
            route: numpy array of the route features to predict the biometrics for

        Returns:
            numpy array of the predicted biometric features (reversely scaled after being normalized)
        
        
        '''
        scaled_route_prediction = self.model.predict(self.route_scaler.transform(route),verbose=0)
        return self.biometric_scaler.inverse_transform(scaled_route_prediction)


class HeartRatePredictor:
    def __init__(self,route_features_number=4,biometric_features_number=7):
        '''
        Initializes the HeartRatePredictor based on the number of route and biometric features

        Args:
            route_features_number: An optional argument that represents the number of route features (int; defaults to 4)
            biometric_features_number: An optional argument that represents the number of biometric features (int; defaults to 7)
        
        '''
        self.route_scaler = StandardScaler()
        self.biometric_scaler = StandardScaler()
        self.avg_hr_scaler = StandardScaler()
        self.max_hr_scaler = StandardScaler()
        self.route_feature_count = route_features_number
        self.biometric_feature_count = biometric_features_number
        self.model = None

    def _create_model(self):
        route = Input(shape=(self.route_feature_count,),name='route')
        biometrics = Input(shape=(self.biometric_feature_count,),name=
                           'biometrics')

        #combine the route and biometric data
        all_features = tf.keras.layers.concatenate([route,biometrics])
        #optimal hyperparameters found through previous tuning
        x = Dense(32, activation='relu')(all_features)
        x = Dropout(0.2)(x)
        x = Dense(160, activation='relu')(x)
        x = Dropout(0.0)(x)
        x = Dense(256, activation='relu')(x) 
        x = Dropout(0.3)(x)  
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.0)(x)
        x = Dense(192, activation='relu')(x)
        x = Dropout(0.0)(x)
        #final prediction layer
        predicted_avg_hr = Dense(1,activation='linear',name='predicted_avg_hr')(x)
        predicted_max_hr = Dense(1,activation='linear',name='predicted_max_hr')(x)
        #create model
        model = Model(inputs=[route,biometrics],outputs=[predicted_avg_hr,predicted_max_hr])
        model.compile(optimizer=Adam(learning_rate=0.002613807343189224),loss='mean_squared_error',
                      metrics={'predicted_avg_hr':'mae','predicted_max_hr':'mae'})
        
        return model
    
    def train(self,route_features,biometric_features,avg_heart_rates,max_heart_rates,epochs=150):
        '''
        Trains the heart rate prediction model

        Args:
            route_features: numpy array of the input route features for training
            biometric_features: numpy array of the input biometric features for training
            avg_heart_rates: numpy array of the avg heart rate targets for training
            max_heart_rates: numpy array of the max heart rate targets for training
            epochs: Optional argument which specifies the number of training epochs (int; defults to 150)
        
        Returns:
            dict: the performance metrics as a result of training (avg/max HR MAE)
        '''
        scaled_route=self.route_scaler.fit_transform(route_features)
        scaled_biometrics=self.biometric_scaler.fit_transform(biometric_features)
        
        #split into training and testing
        (route_training, route_testing, biometric_training, 
        biometric_testing,avg_hr_training,avg_hr_testing,
        max_hr_training,max_hr_testing) = train_test_split(scaled_route,scaled_biometrics,avg_heart_rates,
                                                           max_heart_rates,test_size=0.2,random_state=42)
        
        #create the model
        self.model=self._create_model()
        results = self.model.fit(
            {
                'route': route_training,
                'biometrics': biometric_training
            },
            {
                'predicted_avg_hr': avg_hr_training,
                'predicted_max_hr': max_hr_training
            },
            validation_data=(
                {
                    'route': route_testing,
                    'biometrics': biometric_testing
                },
                {
                    'predicted_avg_hr': avg_hr_testing,
                    'predicted_max_hr': max_hr_testing
                }
            ),
            epochs=epochs,
            verbose=0
        )

        return {
            'avg_hr_mae': results.history['predicted_avg_hr_mae'][-1],
            'max_hr_mae': results.history['predicted_max_hr_mae'][-1]
        }
    
    def heart_rate_prediction(self,route,biometrics):
        '''
        Predicts the avg and max heart rategiven route and biometric features

        Args:
            route: numpy array of the route features to predict heart rates for
            biometrics: numpy array of the biometric features to predict heart rates for

        Returns:
            tuple: displaying the predicted avg and max heart rates


        
        '''
        route_scaled = self.route_scaler.transform(route)
        biometric_scaled = self.biometric_scaler.transform(biometrics)
        
        return self.model.predict({
            'route': route_scaled, 
            'biometrics': biometric_scaled
        }, verbose=0)

def prepare_data(df):
    '''
    Prepares the data for training by extracting the necessary features from the formatted dataframe

    Args:
        df: pandas dataframe containing the user's running data
    
    Returns:
        tuple: the extracted route/biometric features, average and max heart rates

    
    
    '''
    route_features_columns = [
        'distance', 
        'averagePace', 
        'elevationGain', 
        'elevationLoss'
    ]
    
    biometric_features_columns = [
        'calories', 
        'activityTrainingLoad', 
        'avgPower', 
        'averageRunningCadenceInStepsPerMinute', 
        'avgVerticalOscillation', 
        'avgStrideLength', 
        'avgGroundContactTime'
    ]

    route_features = df[route_features_columns].values
    biometric_features=df[biometric_features_columns].values
    average_hrs = df['averageHR'].values.reshape(-1,1)
    max_hrs = df['maxHR'].values.reshape(-1,1)
    return route_features,biometric_features,average_hrs,max_hrs

def prepare_new_route(distance, average_pace, elevation_gain, elevation_loss):
    '''
    Prepares a numpy array (new route for prediction)

    Args:
        distance: float that represents the distance of the route in miles
        average_pace: float the represents the average pace of the route in min/mile
        elevation_gain: float that represents the elevation gain associated with the route (ft)
        elevation_loss: float that represents the elevation lost along the route (ft)
    
    Returns:
        numpy array: A 2D array of the route features
    
    
    '''
    new_route = np.array([[distance, average_pace, elevation_gain, elevation_loss]])
    return new_route

def train_and_predict(df,new_route):
    '''
    Trains biometric and heart rate neural networks and predicts for the prepared route

    Args:
        df: pandas dataframe that represents the training data (after being formatted by SDV)
        new_route: numpy array that represents the route features the user wishes to predict for

    Returns:
        tuple: returns the predicted biometrics, avg hear rate, and max heart rate associated with those route features
    
    '''
    route_features,biometric_features,average_hrs,max_hrs = prepare_data(df)

    biometric_predictor = BiometricPredictor(route_features_number=route_features.shape[1],
                                             biometric_features_number=biometric_features.shape[1])
    biometric_predictor.train(route_features,biometric_features)

    hr_predictor = HeartRatePredictor(route_features_number=route_features.shape[1],
                                      biometric_features_number=biometric_features.shape[1])
    
    hr_performance = hr_predictor.train(route_features,biometric_features,average_hrs,max_hrs)

    print(f"Average HR MAE: {hr_performance['avg_hr_mae']}")
    print(f"Max HR MAE: {hr_performance['max_hr_mae']}")

    final_biometric_prediction = biometric_predictor.biometric_prediction(new_route)


    final_avg_hr, final_max_hr = hr_predictor.heart_rate_prediction(new_route, final_biometric_prediction)
    print("\nPredicted Heart Rates:")
    print(f"Average HR: {final_avg_hr[0][0]}")    
    print(f"Max HR: {final_max_hr[0][0]}")        

    return final_biometric_prediction[0], final_avg_hr[0][0], final_max_hr[0][0] 