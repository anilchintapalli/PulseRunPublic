import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class BiometricPredictorNet(nn.Module):
    def __init__(self, route_features_number=4):
        super(BiometricPredictorNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(route_features_number, 192),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(192, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 7)  # 7 biometric features
        )
    
    def tune_hyperparameters(self, route_features_number, biometric_features_number, max_trials=50, epochs=100):
        '''
        comments
        '''
        #scaled_route
        pass

    def forward(self, x):
        return self.network(x)

class HeartRatePredictorNet(nn.Module):
    def __init__(self, route_features_number=4, biometric_features_number=7):
        super(HeartRatePredictorNet, self).__init__()
        total_features = route_features_number + biometric_features_number
        
        self.network = nn.Sequential(
            nn.Linear(total_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 160),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(160, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(32, 192),
            nn.ReLU(),
            nn.Dropout(0.0),
        )
        
        self.avg_hr_head = nn.Linear(192, 1)
        self.max_hr_head = nn.Linear(192, 1)
    def tune_hyperparameters(self):
        pass

    def forward(self, route, biometrics):
        x = torch.cat([route, biometrics], dim=1)
        shared = self.network(x)
        avg_hr = self.avg_hr_head(shared)
        max_hr = self.max_hr_head(shared)
        return avg_hr, max_hr

class BiometricPredictor:
    def __init__(self, route_features_number=4, biometric_features_number=7):
        self.route_scaler = StandardScaler()
        self.biometric_scaler = StandardScaler()
        self.route_feature_count = route_features_number
        self.biometric_feature_count = biometric_features_number
        self.model = BiometricPredictorNet(route_features_number)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, route_features, biometric_features, epochs=6):
        scaled_route = self.route_scaler.fit_transform(route_features)
        scaled_biometrics = self.biometric_scaler.fit_transform(biometric_features)

        route_training, route_testing, biometric_training, biometric_testing = train_test_split(
            scaled_route, scaled_biometrics, test_size=0.2, random_state=42
        )

        # Convert to PyTorch tensors
        route_training = torch.FloatTensor(route_training).to(self.device)
        biometric_training = torch.FloatTensor(biometric_training).to(self.device)
        route_testing = torch.FloatTensor(route_testing).to(self.device)
        biometric_testing = torch.FloatTensor(biometric_testing).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=0.008569905521819712)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(route_training)
            loss = criterion(outputs, biometric_training)
            loss.backward()
            optimizer.step()

    def biometric_prediction(self, route):
        self.model.eval()
        with torch.no_grad():
            scaled_route = self.route_scaler.transform(route)
            route_tensor = torch.FloatTensor(scaled_route).to(self.device)
            predictions = self.model(route_tensor)
            return self.biometric_scaler.inverse_transform(predictions.cpu().numpy())

    def export_onnx(self, filepath):
        dummy_input = torch.randn(1, self.route_feature_count).to(self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['route'],
            output_names=['biometrics'],
            dynamic_axes={
                'route': {0: 'batch_size'},
                'biometrics': {0: 'batch_size'}
            }
        )

class HeartRatePredictor:
    def __init__(self, route_features_number=4, biometric_features_number=7):
        self.route_scaler = StandardScaler()
        self.biometric_scaler = StandardScaler()
        self.route_feature_count = route_features_number
        self.biometric_feature_count = biometric_features_number
        self.model = HeartRatePredictorNet(route_features_number, biometric_features_number)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, route_features, biometric_features, avg_heart_rates, max_heart_rates, epochs=150):
        scaled_route = self.route_scaler.fit_transform(route_features)
        scaled_biometrics = self.biometric_scaler.fit_transform(biometric_features)

        (route_training, route_testing, biometric_training, biometric_testing,
         avg_hr_training, avg_hr_testing, max_hr_training, max_hr_testing) = train_test_split(
            scaled_route, scaled_biometrics, avg_heart_rates, max_heart_rates,
            test_size=0.2, random_state=42
        )

        # Convert to PyTorch tensors
        route_training = torch.FloatTensor(route_training).to(self.device)
        biometric_training = torch.FloatTensor(biometric_training).to(self.device)
        avg_hr_training = torch.FloatTensor(avg_hr_training).to(self.device)
        max_hr_training = torch.FloatTensor(max_hr_training).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=0.002613807343189224)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            avg_pred, max_pred = self.model(route_training, biometric_training)
            loss = criterion(avg_pred, avg_hr_training) + criterion(max_pred, max_hr_training)
            loss.backward()
            optimizer.step()

    def heart_rate_prediction(self, route, biometrics):
        self.model.eval()
        with torch.no_grad():
            route_scaled = self.route_scaler.transform(route)
            biometric_scaled = self.biometric_scaler.transform(biometrics)
            
            route_tensor = torch.FloatTensor(route_scaled).to(self.device)
            biometric_tensor = torch.FloatTensor(biometric_scaled).to(self.device)
            
            avg_hr, max_hr = self.model(route_tensor, biometric_tensor)
            return avg_hr.cpu().numpy(), max_hr.cpu().numpy()

    def export_onnx(self, filepath):
        dummy_route = torch.randn(1, self.route_feature_count).to(self.device)
        dummy_biometrics = torch.randn(1, self.biometric_feature_count).to(self.device)
        torch.onnx.export(
            self.model,
            (dummy_route, dummy_biometrics),
            filepath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['route', 'biometrics'],
            output_names=['avg_hr', 'max_hr'],
            dynamic_axes={
                'route': {0: 'batch_size'},
                'biometrics': {0: 'batch_size'},
                'avg_hr': {0: 'batch_size'},
                'max_hr': {0: 'batch_size'}
            }
        )

# Helper functions remain largely the same
def prepare_data(df):
    route_features_columns = [
        'distance', 'averagePace', 'elevationGain', 'elevationLoss'
    ]
    
    biometric_features_columns = [
        'calories', 'activityTrainingLoad', 'avgPower',
        'averageRunningCadenceInStepsPerMinute', 'avgVerticalOscillation',
        'avgStrideLength', 'avgGroundContactTime'
    ]

    route_features = df[route_features_columns].values
    biometric_features = df[biometric_features_columns].values
    average_hrs = df['averageHR'].values.reshape(-1, 1)
    max_hrs = df['maxHR'].values.reshape(-1, 1)
    return route_features, biometric_features, average_hrs, max_hrs

def prepare_new_route(distance, average_pace, elevation_gain, elevation_loss):
    return np.array([[distance, average_pace, elevation_gain, elevation_loss]])

def train_and_predict(df, new_route):
    route_features, biometric_features, average_hrs, max_hrs = prepare_data(df)

    # Train biometric predictor
    biometric_predictor = BiometricPredictor(
        route_features_number=route_features.shape[1],
        biometric_features_number=biometric_features.shape[1]
    )
    biometric_predictor.train(route_features, biometric_features)
    
    # Export biometric predictor to ONNX
    biometric_predictor.export_onnx('biometric_predictor.onnx')

    # Train heart rate predictor
    hr_predictor = HeartRatePredictor(
        route_features_number=route_features.shape[1],
        biometric_features_number=biometric_features.shape[1]
    )
    hr_predictor.train(route_features, biometric_features, average_hrs, max_hrs)
    
    # Export heart rate predictor to ONNX
    hr_predictor.export_onnx('heart_rate_predictor.onnx')

    # Make predictions
    final_biometric_prediction = biometric_predictor.biometric_prediction(new_route)
    final_avg_hr, final_max_hr = hr_predictor.heart_rate_prediction(new_route, final_biometric_prediction)

    print("\nPredicted Heart Rates:")
    print(f"Average HR: {final_avg_hr[0][0]}")
    print(f"Max HR: {final_max_hr[0][0]}")

    return final_biometric_prediction[0], final_avg_hr[0][0], final_max_hr[0][0]

def main():
    # Load your training data
    data = pd.read_csv('synthetic_running_data9.csv')
    route = prepare_new_route(5,420,200,10)
    # Train and predict
    predicted_biometrics, predicted_avg_hr, predicted_max_hr = train_and_predict(data,route)

if __name__ == "__main__":
    main()