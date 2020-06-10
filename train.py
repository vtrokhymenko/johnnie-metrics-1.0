import tensorflow as tf
import datetime
import yaml
import json
import time

params = yaml.safe_load(open('params.yaml'))
epochs = params['epochs']
log_file = params['log_file']
dense = params['dense']

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(dense, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

model = create_model()
model.compile(optimizer='RMSprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

csv_logger = tf.keras.callbacks.CSVLogger(log_file)

start_real = time.time()
start_process = time.process_time()
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    callbacks=[csv_logger, tensorboard_callback])
end_real = time.time()
end_process = time.process_time()

with open("summary.json", "w") as fd:
    json.dump({
        "accuracy": float(history.history["accuracy"][-1]),
        "loss": float(history.history["loss"][-1]),
        "time_real" : end_real - start_real,
        "time_process": end_process - start_process
    }, fd)

