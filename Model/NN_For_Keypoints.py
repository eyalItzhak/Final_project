import tensorflow as tf
import json
import numpy as np
import os
import pathlib
import pandas
from sklearn import preprocessing
TRAIN_INPUT = "..\Yoga_output_train"
TEST_INPUT = "..\Yoga_output_test"
MAX_UNRECOGNIZED_KEY_POINTS = 10
def get_position_data(folder_name):
    data = []
    labels = []
    position_name = os.path.basename(folder_name)
    if(os.path.exists(os.path.join(folder_name,"Normalized_JSON"))):
        files = os.scandir(os.path.join(folder_name,"Normalized_JSON"))
        for file in filter(lambda x : x.name.endswith("_normalized.json"),files):
            file_data = json.load(open(file)) 
            for person in file_data["people"]: #iterate over all people in json file
                key_points = person["pose_keypoints_2d"]
                person_data  = []
                unknown_points = 0
                for key_point in key_points:
                    person_data.append(key_point["coordinate"][0])
                    person_data.append(key_point["coordinate"][1])
                    person_data.append(key_point["probability"])
                    if(key_point["probability"] == 0):
                        unknown_points+= 1
                if(unknown_points<MAX_UNRECOGNIZED_KEY_POINTS):
                    data.append(person_data)
                    labels.append(position_name)
    return data,labels


def get_input_data(input):
    data = []
    labels = []
    path = os.path.join(pathlib.Path(__file__).parent.resolve(),input)
    for folder_name in os.scandir(path):
        if(os.path.exists(os.path.join(folder_name,"Normalized_JSON"))):
            position_data,position_labels = get_position_data(folder_name)
            for i,position in enumerate(position_data):
                data.append(position)
                labels.append(position_labels[i])
    return data,labels

def validate_samples_size(train_labels,test_labels):
    train_classes = np.unique(train_labels)
    test_classes = np.unique(test_labels)
    if len(train_classes) > len(test_classes):
        for train_sample in train_classes:
            if train_sample not in test_classes:
                print( train_sample +"not found in test!")
    elif len(train_classes) < len(test_classes):
        for test_sample in test_classes:
            if test_sample not in train_classes:
                print( test_sample +"not found in train!")
    else:
        print("Valid sample size: " + str(len(train_classes)))
        



# Define input data and labels
train_input_data,train_labels = get_input_data(TRAIN_INPUT)
test_input_data, test_labels = get_input_data(TEST_INPUT)
validate_samples_size(train_labels,test_labels)
#####TRAIN#############
train_df = pandas.DataFrame(train_input_data)
label_df = pandas.DataFrame(train_labels)


# Convert labels to one-hot encoding
classes = np.unique(train_labels)
num_classes = len(classes)
label_to_int = {label: i for i, label in enumerate(classes)}
int_to_label = {i: label for label, i in label_to_int.items()}
labels_int = np.array([label_to_int[label] for label in train_labels])
labels_one_hot = np.zeros((len(train_labels), num_classes))
labels_one_hot[np.arange(len(train_labels)), labels_int] = 1



#####TEST#############
test_df = pandas.DataFrame(test_input_data)
test_label_df = pandas.DataFrame(test_labels)


# Convert labels to one-hot encoding
classes = np.unique(test_labels)
num_classes = len(classes)
label_to_int = {label: i for i, label in enumerate(classes)}
int_to_label = {i: label for label, i in label_to_int.items()}
labels_int = np.array([label_to_int[label] for label in test_labels])
test_labels_one_hot = np.zeros((len(test_labels), num_classes))
test_labels_one_hot[np.arange(len(test_labels)), labels_int] = 1




# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(75,)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1,75)))
# model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(100, activation='relu'))
# model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_df, labels_one_hot,batch_size=32, epochs=30)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_df, test_labels_one_hot)

print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')


