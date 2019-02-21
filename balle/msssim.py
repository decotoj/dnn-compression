import tensorflow as tf

original = tf.image.decode_png(tf.read_file("/home/jake/Desktop/CS230/Project/results/2092.png"))

decompressed = tf.image.decode_png(tf.read_file("/home/jake/Desktop/CS230/Project/results/2092_balle_AWS.png"))
#decompressed = tf.image.decode_png(tf.read_file("/home/jake/Desktop/CS230/Project/results/2092_balle_decoto.png"))
#decompressed = tf.image.decode_png(tf.read_file("/home/jake/Desktop/CS230/Project/results/2092_jpeg_30.png"))

ssim = tf.image.ssim_multiscale(original,decompressed, 255)

sess = tf.Session()

print = tf.print("The error metric is: ", ssim)

sess.run(print)