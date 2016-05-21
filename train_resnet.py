import tensorflow as tf
from tensorflow.contrib.learn.python.learn.ops.conv_ops import conv2d
from batchnorm import batch_norm
import data_iterator
import time
import os.path

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', None, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('iterations', 2000, 'Number of epochs to train (1M examples per epoch).')
flags.DEFINE_integer('num_modules', 16, 'Number of resnet modules.')
flags.DEFINE_integer('num_features', 192, 'Feature map depth.')
flags.DEFINE_string('summary_dir', None, 'Where to put the summary logs')
flags.DEFINE_string('checkpoint_dir', None, 'Where to put checkpoint files')
flags.DEFINE_string('games_list', None, 'file with list of HDF5s to train from')

def conv2d_bn(data, depth, filter_size, training_switch, **kwargs):
    kwargs['bias'] = False  # batch normalization obviates biases
    return batch_norm(conv2d(data, depth, filter_size, **kwargs),
                      depth, training_switch)

def res_3x3_pair(data, depth, scope_name, training_switch, keep_prob_var):
    # following He 2016, we do BN and activation on the input
    with tf.variable_scope(scope_name):
        # BN and activation on input
        normed = tf.nn.relu(batch_norm(data, depth, training_switch))
        with tf.variable_scope("layer_1"):
            conv1 = conv2d_bn(normed, depth, (3, 3), training_switch, activation=tf.nn.relu)
            conv1 = tf.nn.dropout(conv1, keep_prob_var)
        with tf.variable_scope("layer_2"):
            conv2 = conv2d(conv1, depth, (3, 3), bias=True)
    return conv2 + data

def model(input_data, training_switch, keep_prob_var, num_modules=20, depth=64):
    with tf.variable_scope("input"):
        # [filter_height, filter_width, in_channels, out_channels]
        model = conv2d(tf.to_float(input_data), depth, (5, 5), bias=True)

    print("{} modules".format(num_modules))
    for idx in range(num_modules):
        model = res_3x3_pair(model, depth, "resnet_module_{}".format(idx + 1), training_switch, keep_prob_var)

    with tf.variable_scope("down8"):
        model = conv2d_bn(model, 8, (3, 3), training_switch, activation=tf.nn.relu)
    with tf.variable_scope("down2"):
        model = conv2d_bn(model, 2, (1, 1), training_switch, activation=tf.nn.relu)

    # output is 362 softmax: 361 possible moves, plus pass move
    with tf.variable_scope("output"):
        W = tf.Variable(tf.truncated_normal([19 * 19 * 2, 362], stddev=(1.0 / 361 * 2)), name='W')
        B = tf.Variable(tf.constant(0.0, shape=[362]))
        model = tf.matmul(model.reshape([-1, 19 * 19 * 2]), W) + B

    return model

def run_training():
    data_set = data_iterator.HDF5Iterator([f.strip() for f in open(FLAGS.games_list)],
                                          batch_size=FLAGS.batch_size)

    sess = tf.Session()
    summaries = []

    in_data = tf.placeholder(tf.uint8, (FLAGS.batch_size,) + data_set.lshape, name='input')
    in_training = tf.Variable(True, name='in_training')
    keep_prob = tf.constant(0.9, name='keep_prob')

    predictions = model(in_data, in_training, keep_prob, num_modules=FLAGS.num_modules)

    label = tf.placeholder(tf.uint64, (FLAGS.batch_size,))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(predictions, label))
    accuracy = tf.reduce_mean(tf.argmax(label, 1) == tf.argmax(predictions, 1))
    top5 = tf.nn.in_top_k(predictions, label, 5)

    summaries.append(tf.scalar_summary("loss", loss))
    summaries.append(tf.scalar_summary("accuracy", accuracy))
    summaries.append(tf.scalar_summary("top5", top5))

    step_var = tf.Variable(0)
    inc_step = step_var.assign_add(1)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.95)
    train_op = opt.minimize(loss, colocate_gradients_with_ops=True, aggregation_method=2)
    summary_op = tf.merge_summary(summaries)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    print(sess.graph)
    summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir, sess.graph)

    checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if checkpoint:
        print("restoring from checkpoint", checkpoint)
        saver.restore(sess, checkpoint)
    else:
        print("Couldn't find checkpoint to restore from.  Starting fresh.")

    while True:
        batch_input, batch_labels = data_set.batch(FLAGS.batch_size)
        feed_dict = {in_data: batch_input, label: batch_labels, in_training: True}

        start_time = time.time()
        _, step, loss_val, accuracy_val, top5_val, summary_str = sess.run([train_op, inc_step, loss, accuracy_val, top5, summary_op],
                                                                          feed_dict=feed_dict)

        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, "checkpoint"), global_step=step)

        duration = time.time() - start_time

        print('Step %d: loss = %.3f, acc = %.2f, top5 = %.2f (%.1f sec)' %
              (step, loss_val, accuracy_val, top5_val, duration))

        if step >= int(FLAGS.iterations):
            break

    # End train loop
    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, "final-checkpoint"), global_step=step)

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()
