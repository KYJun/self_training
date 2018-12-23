import tensorflow as tf
import numpy as np

# load mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# convert label to one-hot vector
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# shuffle train data
np.random.seed(0)
shuffled_indices = np.random.permutation(np.arange(60000))
x_train_shuffled = x_train[shuffled_indices, :, :]
y_train_shuffled = y_train[shuffled_indices, :]

# split train data into labeled / unlabeled data
x_labeled = x_train_shuffled[:50000, :, :]
y_labeled = y_train_shuffled[:50000, :]

x_unlabeled = x_train_shuffled[50000:, :, :]
y_unlabeled = y_train_shuffled[50000:, :]

# build simple nerual network
# 2 layers (256, 10 for nodes each, relu and softmax for activation, adam optimizer)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train model (about 3 epochs)
model.fit(x_labeled, y_labeled, epochs=3)

# obtain current accuracy
loss, acc = model.evaluate(x_test, y_test)
print("Current Accuracy: ", acc)

# 3 methods possible
methods = ["all", "top", "weight"]

# set method
method = methods[2]

# add prediction of current model inferred from unlabeled data
if method == "all":

    # epochs
    for _ in range(2):
        # obtain prediction
        predictions = model.predict(x_unlabeled)

        # switch prediction to one-hot label
        pseudo_labels = np.argmax(predictions, axis=1)
        pseudo_labels = np.eye(10)[pseudo_labels]

        # concatenate all data
        all_x = np.concatenate([x_labeled, x_unlabeled], axis=0)
        all_y = np.concatenate([y_labeled, pseudo_labels], axis=0)

        # do fitting
        model.fit(all_x, all_y, epochs=1)

    # calculate test loss
    t_loss, t_acc = model.evaluate(x_test, y_test)

    print("Accuracy changed to : ", t_acc)

# add prediction of top 10% (1000 instances) inferred from unlabeled data
elif method == "top":

    # epochs
    for _ in range(2):

        # obtain prediction
        predictions = model.predict(x_unlabeled)

        # get prediction probability and its predicted label
        prediction_prob = np.amax(predictions, axis=1, keepdims=True)
        prediction_labels = np.argmax(predictions, axis=1)

        # sort logit into order
        prediction_sort = np.argsort(prediction_prob, axis=0)

        # get 1000 high probability instances
        index_target = np.where(prediction_sort < 1000)[0]

        # convert 1000 instances prediction to one-hot labels
        pseudo_labels_topten = prediction_labels[index_target]
        pseudo_labels_topten = np.eye(10)[pseudo_labels_topten]

        # top 1000 instances of x data
        x_unlabeled_topten = x_unlabeled[index_target, :, :]

        # concatenate data
        new_x = np.concatenate((x_labeled, x_unlabeled_topten), axis=0)
        new_y = np.concatenate((y_labeled, pseudo_labels_topten), axis=0)

        # do fitting
        model.fit(new_x, new_y, epochs=1)

    # calculate test loss
    t_loss, t_acc = model.evaluate(x_test, y_test)

    print("Accuracy changed to : ", t_acc)


# apply weights for each predicted instances
elif method == "weight":

    # epochs
    for _ in range(2):

        # obtain predictions
        predictions = model.predict(x_unlabeled)

        # treat predicted logit as weight value
        pseudo_weights = np.amax(predictions, axis=1, keepdims=True)

        # convert weight as y vectors
        pseudo_labels = np.where(predictions == pseudo_weights, pseudo_weights, 0)

        # concatenate all data
        all_x = np.concatenate((x_labeled, x_unlabeled), axis=0)
        all_y = np.concatenate((y_labeled, pseudo_labels), axis=0)

        # do fitting
        model.fit(all_x, all_y, epochs=1)

    # calculate loss
    t_loss, t_acc = model.evaluate(x_test, y_test)

    print("Accuracy changed to : ", t_acc)

# for comparison
else:

    # train with all the train data
    model.fit(x_train, y_train, epochs=2)

    base_loss, base_acc = model.evaluate(x_test, y_test)

    print("Baseline loss : ", base_loss)







