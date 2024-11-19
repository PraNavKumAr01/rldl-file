import tensorflow as tf
import torch
import keras

def test_installations():
    print("TensorFlow version:", tf.__version__)
    print("GPU available for TensorFlow:", tf.config.list_physical_devices('GPU'))
    print("\nPyTorch version:", torch.__version__)
    print("GPU available for PyTorch:", torch.cuda.is_available())
    tf_tensor = tf.constant([[1., 2.], [3., 4.]])
    print("\nTensorFlow tensor:\n", tf_tensor)
    torch_tensor = torch.tensor([[1., 2.], [3., 4.]])
    print("\nPyTorch tensor:\n", torch_tensor)
if __name__ == "__main__":
    test_installations()