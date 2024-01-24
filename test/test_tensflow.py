import tensorflow as tf

print("Built with CUDA: ", tf.test.is_built_with_cuda())
print("GPUs Available:", tf.config.list_physical_devices('GPU'))