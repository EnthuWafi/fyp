import tensorflow as tf
import tf2onnx

# The path to the stubborn FaceChannel model
h5_path = "/home/pasokon/anaconda3/envs/facechannel_env/lib/python3.9/site-packages/FaceChannel/FaceChannelV1/TrainedNetworks/DimensionalFaceChannel.h5"
onnx_path = "facechannel.onnx"

print("[1/3] Loading Keras model...")
# The magic argument: compile=False completely ignores the missing 'ccc' metric!
model = tf.keras.models.load_model(h5_path, compile=False)

print("[2/3] Converting to ONNX format...")
# tf2onnx automatically figures out the input shapes
model_proto, _ = tf2onnx.convert.from_keras(model, output_path=onnx_path)

print(f"[3/3] Success! ONNX model saved to: {onnx_path}")