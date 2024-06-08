import math
import numpy as np
import matplotlib.pyplot as plt


def plot_data_with_best_features(data, class_labels, best_features):
    num_instances, num_features = data.shape

    # Extract the features and class labels
    features = data[:, 1:]

    # Plot each pair of best features
    for i in range(len(best_features)):
        for j in range(i + 1, len(best_features)):
            plt.figure()
            for label in np.unique(class_labels):
                indices = class_labels == label
                plt.scatter(features[indices, best_features[i] - 1], features[indices, best_features[j] - 1], label=f'Class {label}')
            plt.xlabel(f'Feature {best_features[i]}')
            plt.ylabel(f'Feature {best_features[j]}')
            plt.title(f'Features {best_features[i]} vs {best_features[j]}')
            plt.legend()
            plt.show()

def normalize(data, num_features, num_instances, feature1_index, feature2_index):


    # Calculate mean and standard deviation for the selected features
    mean_feature1 = sum(row[feature1_index] for row in data) / num_instances
    mean_feature2 = sum(row[feature2_index] for row in data) / num_instances

    variance_feature1 = sum(pow((row[feature1_index] - mean_feature1), 2) for row in data) / num_instances
    variance_feature2 = sum(pow((row[feature2_index] - mean_feature2), 2) for row in data) / num_instances

    std_feature1 = math.sqrt(variance_feature1)
    std_feature2 = math.sqrt(variance_feature2)

    # Normalize the selected features
    for i in range(num_instances):
        data[i][feature1_index] = (data[i][feature1_index] - mean_feature1) / std_feature1
        data[i][feature2_index] = (data[i][feature2_index] - mean_feature2) / std_feature2

    return data

def read_data(file):
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
            instances = [list(map(float, line.split())) for line in f.readlines()]
            num_instances = len(instances)
            data = np.array([list(map(float, line.split())) for line in lines])
            class_labels = data[:, 0].astype(int)

            return data, class_labels, num_instances
    except FileNotFoundError:
        raise FileNotFoundError(f'The file {file} does not exist. Exiting program.')

def main():
    file = input('Type in the name of the file to read data from: ')
    data, class_labels, num_instances = read_data(file)
    data = normalize(data, 2, num_instances, 20, 33)
    best_features = [20, 33]  # Replace this with your best features
    plot_data_with_best_features(data, class_labels, best_features)

if __name__ == '__main__':
	main()