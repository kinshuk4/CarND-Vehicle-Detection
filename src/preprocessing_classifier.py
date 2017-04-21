import glob
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import src.feature_extractor as fext

DEFAULT_TEST_SIZE = 0.2


class PreProcessingClassifier:
    def __init__(self):
        self.svc = None
        self.X_test = None
        self.y_test = None
        self.X_scaler = None

    def train(self, vehicle_files, non_vehicle_files, test_size=DEFAULT_TEST_SIZE):
        print("Extracting the features....")
        vehicle_features = fext.extract_features_from_file_list(vehicle_files)
        non_vehicle_features = fext.extract_features_from_file_list(non_vehicle_files)
        print("Features extracted.")

        X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
        X_scaler = StandardScaler().fit(X)
        scaled_X = X_scaler.transform(X)

        y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=test_size, random_state=rand_state)

        svc = LinearSVC()
        svc.fit(X_train, y_train)
        print("Training complete")

        self.X_test = X_test
        self.y_test = y_test
        self.X_scaler = X_scaler
        self.svc = svc

        return svc

    def test(self, n_predict=15):
        print('ACCURACY:     ', round(self.svc.score(self.X_test, self.y_test), 4))
        print('SVC PREDICTS: ', self.svc.predict(self.X_test[0:n_predict]))
        print('LABELS:       ', self.y_test[0:n_predict])

    def dump(self, model_path, scaler_path):
        print("Starting the dump")
        pickle.dump(self.svc, open(model_path, "wb"))
        pickle.dump(self.X_scaler, open(scaler_path, "wb"))
        print("Ending the dump")

    def get(self, model_path, scaler_path):
        svc_trained = pickle.load(open(model_path, "rb"))
        xscaler = pickle.load(open(scaler_path, "rb"))
        return svc_trained, xscaler


def main():
    data_folder = "../../DataSets/carnd-vehicle-detection-p5-data/data1/"

    vehicle_files = glob.glob(data_folder + 'vehicles/**/*.png', recursive=True)
    non_vehicle_files = glob.glob(data_folder + 'non-vehicles/**/*.png', recursive=True)

    print('Total Vehicle and non Vehicle files : {} and {} '.format(len(vehicle_files), len(non_vehicle_files)))
    classifier = PreProcessingClassifier()
    classifier.train(vehicle_files, non_vehicle_files)


if __name__ == '__main__':
    main()
