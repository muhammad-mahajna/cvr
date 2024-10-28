
# ------------------------------------------
import os
import tensorflow as tf

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sep = "=" * 80
# ------------------------------------------
print(sep)
print("Checking for available GPUs...\n")
gg = tf.config.list_physical_devices("GPU")
print("GPUs found: %d" % len(gg))

for g in gg:
    print("\t%s: %s" % (g.device_type, g.name))

# ------------------------------------------
print(sep)
print("Testing Tensorflow...\n")
try:
    x = tf.reduce_sum(tf.random.normal([1000, 1000]))
except:
    print("FAILED!")
    sys.exit()

print("Tensorflow works properly. SUCCESS!")
# ------------------------------------------
print()
