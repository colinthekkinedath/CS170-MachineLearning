import math
import copy

# Function to find the nearest neighbor for a given data point
def nearest_neighbor_classifier(data, point, feature_subset, num_instances):
    nearest_neighbor = None
    shortest_distance = float('inf')

    # Iterate through all instances to find the nearest neighbor
    for i in range(num_instances):
        if i == point:
            continue
        
        # Calculate the distance between the points
        distance = calculate_distance(data, point, i, feature_subset)

        # Update nearest neighbor if the distance is shorter
        if distance < shortest_distance:
            nearest_neighbor = i
            shortest_distance = distance

    return nearest_neighbor

# Function to calculate the distance between two points based on selected features
def calculate_distance(data, point1, point2, feature_subset):
    distance = 0
    for feature in feature_subset:
        distance += (data[point1][feature] - data[point2][feature]) ** 2
    return math.sqrt(distance)

# Function to perform one-out validation for a given feature subset
def one_out_validator(data, feature_subset, num_instances):
    correct = 0.0
    # Iterate through all instances
    for i in range(num_instances):
        # Find the nearest neighbor using the feature subset
        nearest_neighbor = nearest_neighbor_classifier(data, i, feature_subset, num_instances)
        
        # Check if the predicted class matches the actual class
        if data[nearest_neighbor][0] == data[i][0]:
            correct = correct + 1

    # Calculate accuracy
    accuracy = (correct / num_instances) * 100
    return accuracy

# Function to perform forward selection algorithm
def forward_selection(data, num_instances, num_features):
    feature_subset = []
    final_set = []
    top_accuracy = 0.0

    # Iterate through all features
    for _ in range(num_features):
        best_feature_to_add = None
        local_best_feature = None
        local_best_accuracy = 0.0

        # Iterate through all features to find the best feature to add
        for feature in range(1, num_features + 1):
            if feature not in feature_subset:
                temp_subset = copy.deepcopy(feature_subset)
                temp_subset.append(feature)

                # Calculate accuracy with the current feature subset
                accuracy = one_out_validator(data, temp_subset, num_instances)
                print(f'\tUsing feature(s) {temp_subset} accuracy is {accuracy}%')

                # Update top accuracy and best feature to add if accuracy improves
                if accuracy > top_accuracy:
                    top_accuracy = accuracy
                    best_feature_to_add = feature

                # Update local best accuracy and feature
                if accuracy > local_best_accuracy:
                    local_best_accuracy = accuracy
                    local_best_feature = feature

        # Add the best feature to the feature subset
        if best_feature_to_add is not None:
            feature_subset.append(best_feature_to_add)
            final_set.append(best_feature_to_add)
            print(f'\n\nFeature set {feature_subset} was best, accuracy is {top_accuracy}%\n\n')
        else:
            print('\n\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)')
            feature_subset.append(local_best_feature)
            print(f'Feature set {feature_subset} was best, accuracy is {local_best_accuracy}%\n\n')

    print(f'Finished search!! The best feature subset is {final_set} which has an accuracy of {top_accuracy}%')

# Function to perform backward elimination algorithm
def backward_elimination(data, num_instances, num_features, initial_accuracy):
    feature_subset = list(range(1, num_features + 1))
    final_set = list(range(1, num_features + 1))
    top_accuracy = initial_accuracy

    # Iterate through all features
    for _ in range(num_features):
        best_feature_to_remove = None
        local_best_feature = None
        local_best_accuracy = 0.0

        # Iterate through all features to find the best feature to remove
        for feature in range(1, num_features + 1):
            if feature in feature_subset:
                temp_subset = copy.deepcopy(feature_subset)
                temp_subset.remove(feature)

                # Calculate accuracy with the current feature subset
                accuracy = one_out_validator(data, temp_subset, num_instances)
                print(f'\tUsing feature(s) {temp_subset} accuracy is {accuracy}%')

                # Update top accuracy and best feature to remove if accuracy improves
                if accuracy > top_accuracy:
                    top_accuracy = accuracy
                    best_feature_to_remove = feature

                # Update local best accuracy and feature
                if accuracy > local_best_accuracy:
                    local_best_accuracy = accuracy
                    local_best_feature = feature

        # Remove the best feature from the feature subset
        if best_feature_to_remove is not None:
            feature_subset.remove(best_feature_to_remove)
            final_set.remove(best_feature_to_remove)
            print(f'\n\nFeature set {feature_subset} was best, accuracy is {top_accuracy}%\n\n')
        else:
            print('\n\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)')
            feature_subset.remove(local_best_feature)
            print(f'Feature set {feature_subset} was best, accuracy is {local_best_accuracy}%\n\n')

    print(f'Finished search!! The best feature subset is {final_set} which has an accuracy of {top_accuracy}%')

# Function to calculate the mean of each feature
def mean(data, num_features, num_instances):
    mean = []
    for i in range(0, num_features):
        mean.append((sum(row[i + 1] for row in data)) / num_instances)
    return mean

# Function to calculate the standard deviation of each feature
def std(data, num_features, num_instances):
    standardDeviation = []
    avg = mean(data, num_features, num_instances)
    for i in range(0, num_features):
        variance = sum(pow((row[i + 1] - avg[i]), 2) for row in data) / num_instances
        standardDeviation.append(math.sqrt(variance))
    return standardDeviation

# Function to normalize the data
def normalize(data, num_features, num_instances):
    avg = mean(data, num_features, num_instances)
    standardDeviation = std(data, num_features, num_instances)
    for i in range(0, num_instances):
        for j in range(1, num_features + 1):
            # Normalize each feature in the data
            data[i][j] = ((data[i][j] - avg[j - 1]) / standardDeviation[j - 1])
    return data

# Function to read data from a file
def read_data(file):
    try:
        with open(file, 'r') as data:
            # Read the first line to determine the number of features
            first_line = data.readline()
            num_features = len(first_line.split()) - 1
            # Read the instances from the file
            instances = [list(map(float, line.split())) for line in data.readlines()]
            num_instances = len(instances)
        return instances, num_instances, num_features
    except FileNotFoundError:
        raise FileNotFoundError(f'The file {file} does not exist. Exiting program.')

# Function to select the algorithm
def algorithm_selection(num_features, num_instances):
    print('Type the number of the algorithm you want to run.')
    print('1. Forward Selection')
    print('2. Backward Elimination')
    print('3. Special algorithm')
    choice = int(input())
    while choice not in range(1, 4):
        print('Invalid choice, please try again.')
        choice = int(input())
    print(f'This dataset has {num_features} features (not including the class attribute), with {num_instances + 1} instances.')
    return choice

# Function to print accuracy
def print_accuracy(accuracy, num_features):
    formatted_accuracy = "{:.1f}".format(accuracy)
    print(f'Running nearest neighbor with no features (default rate), using "leaving-one-out" evaluation, I get an accuracy of {formatted_accuracy}%.')

# Main function to run the program
def main():
    print("Welcome to Keshav, Colin, Vijay, and David's Feature Selection Algorithm.\n")

    file = input('Type in the name of the file to test: ')
    instances, num_instances, num_features = read_data(file)
    choice = algorithm_selection(num_features, num_instances)

    print('Please wait while I normalize the data... Done!')
    normalized_instances = normalize(instances, num_features, num_instances)

    accuracy = one_out_validator(normalized_instances, [], num_instances)
    print_accuracy(accuracy, num_features)

    print('Beginning search.\n\n')

    if choice == 1:
        forward_selection(normalized_instances, num_instances, num_features)
    elif choice == 2:
        backward_elimination(normalized_instances, num_instances, num_features, accuracy)
    elif choice == 3:
        print('Not implemented yet. Exiting program.')


if __name__ == '__main__':
    main()
