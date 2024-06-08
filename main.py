import random 
import copy

def main():
    print("Welcome to Keshav, Colin, Vijay, and David's Feature Selection Algorithm.\n")
    
    filename = input('Type the name of the file to test: ')

    dataVals = {} # dictionary that will hold the data

    file = open(filename, 'r')
    data = file.readlines() # read all lines into a list

    for row in data: # parse the row
        row = row.split('\n')
        row = row[0].split(' ')
        row.remove('')

        classVal = int(row[0][0]) # get the instance class

        for i in row[1:]:
            for j in i.split(): # takes care of any whitespace that got through
                instances = dataVals.get(classVal, [])
                instances.append(float(j)) # python float() converts IEEE to double automatically
                dataVals[classVal] = instances

    file.close()

    inputfeatures = input('Please enter total number of features:')

    featuresnum = int(inputfeatures)

    inputalgo = input('\nType the number of the algorithm you want to run. \n1. Forward Selection \n2. Backward Elimination \n3. Special Algorithm.\n')

    if (inputalgo == "1"):
        print(greedy_forward_feature_selection(featuresnum))
    else:
        print(backward_elimination(featuresnum))

    


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

                accuracy = oneOut_Validation()
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

def backward_elimination(num_features):
    # Start with a full subset of features
    feature_subset = list(range(1, num_features + 1))
    final_set = copy.deepcopy(feature_subset)
    topAccuracy = 0.0

    # Initial accuracy with all features
    accuracy = random.uniform(0.0, 100.0)
    topAccuracy = accuracy
    print(f'Using feature(s) {feature_subset} initial accuracy is {topAccuracy:.2f}%')

    # Loop until we have one feature left
    while len(feature_subset) > 1:
        remove_this = -1
        localAccuracy = -1.0
        
        for j in feature_subset:
            # Copy current subset into temp_subset
            temp_subset = copy.deepcopy(feature_subset)
            # Remove feature j from temp_subset and check accuracy
            temp_subset.remove(j)

            accuracy = oneOut_Validation()
            print(f'\tUsing feature(s) {temp_subset} accuracy is {accuracy:.2f}%')

            if accuracy > localAccuracy:
                localAccuracy = accuracy
                remove_this = j

        # Update feature subset based on the highest accuracy found in this iteration
        if remove_this >= 0:
            feature_subset.remove(remove_this)
            print(f'\n\nFeature set {feature_subset} was best, accuracy is {localAccuracy:.2f}%\n\n')
            if localAccuracy > topAccuracy:
                topAccuracy = localAccuracy
                final_set = copy.deepcopy(feature_subset)
    
    print(f'Finished search!! The best feature subset is {final_set}, which has an accuracy of {topAccuracy:.2f}%')

def oneOut_Validation():
    return random.uniform(0.0, 100.0)


if __name__ == "__main__":
    main()