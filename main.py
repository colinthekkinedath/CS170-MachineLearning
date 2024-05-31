import random 
import copy

def main():
    print("Welcome to Keshav, Colin, Vijay, and David's Feature Selection Algorithm.\n")

    inputfeatures = input('Please enter total number of features:')

    featuresnum = int(inputfeatures)

    inputalgo = input('\nType the number of the algorithm you want to run. \n1. Forward Selection \n2. Backward Elimination \n3. Special Algorithm.\n')

    algonum = int(inputalgo)

    print(greedy_forward_feature_selection(featuresnum))


def greedy_forward_feature_selection(num_features):
    # Start with an empty subset
    feature_subset = []
    final_set = []
    topAccuracy = 0.0

    # Loop a maximum of num_features times
    for i in range(num_features):
        add_this = -1
        local_add = -1
        localAccuracy = 0.0
        
        for j in range(1, num_features + 1):
            if j not in feature_subset:
                # Copy current subset into temp_subset
                temp_subset = copy.deepcopy(feature_subset)
                # Add feature j to temp_subset and check accuracy
                temp_subset.append(j)

                accuracy = random.uniform(0.0, 100.0)
                print(f'\tUsing feature(s) {temp_subset} accuracy is {accuracy:.2f}%')

                if accuracy > localAccuracy:
                    localAccuracy = accuracy
                    local_add = j

                # Update the topAccuracy and feature to add
                if accuracy > topAccuracy:
                    topAccuracy = accuracy
                    add_this = j
        
        # Update feature subset based on the highest accuracy found in this iteration
        if add_this >= 0:
            feature_subset.append(add_this)
            final_set = copy.deepcopy(feature_subset)
            print(f'\n\nFeature set {feature_subset} was best, accuracy is {topAccuracy:.2f}%\n\n')
        else:
            # Even if accuracy doesn't improve, we should add the best local feature
            feature_subset.append(local_add)
            print(f'\n\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)')
            print(f'Feature set {feature_subset} was best, accuracy is {localAccuracy:.2f}%\n\n')

    print(f'Finished search!! The best feature subset is {final_set}, which has an accuracy of {topAccuracy:.2f}%')


if __name__ == "__main__":
    main()