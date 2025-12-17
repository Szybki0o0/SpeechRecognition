import tensorflow as tf

model = tf.keras.models.load_model("kws_cnn_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("kws_cnn_model_dynamic.tflite", "wb") as f:
    f.write(tflite_model)

print("Zapisano: kws_cnn_model_dynamic.tflite")
