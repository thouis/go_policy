import tensorflow as tf
from tensorflow.contrib.learn.python.learn.ops.conv_ops import conv2d
from batchnorm import batch_norm
import data_iterator
import time
import os.path
from math import log, exp
from clipopt import ClippedGDOptimizer

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
        normed = tf.nn.elu(batch_norm(data, depth, training_switch))
        with tf.variable_scope("layer_1"):
            conv1 = conv2d_bn(normed, depth, (3, 3), training_switch, activation=tf.nn.elu)
            conv1 = tf.nn.dropout(conv1, keep_prob_var)
        with tf.variable_scope("layer_2"):
            conv2 = conv2d(conv1, depth, (3, 3), bias=True)
        return conv2 + data


def model(input_data, training_switch, keep_prob_var, num_modules=20, depth=64):
    with tf.variable_scope("input"):
        # [filter_height, filter_width, in_channels, out_channels]
        model = conv2d(tf.to_float(input_data), depth, (5, 5), bias=True)

    for idx in range(num_modules):
        model = res_3x3_pair(model, depth, "resnet_module_{}".format(idx + 1), training_switch, keep_prob_var)

    with tf.variable_scope("down8"):
        model = conv2d_bn(model, 8, (3, 3), training_switch, activation=tf.nn.elu)
    with tf.variable_scope("down2"):
        model = conv2d_bn(model, 2, (1, 1), training_switch, activation=tf.nn.elu)

    # output is 362 softmax: 361 possible moves, plus pass move
    with tf.variable_scope("output"):
        W = tf.Variable(tf.truncated_normal([19 * 19 * 2, 362], stddev=(1.0 / 361 * 2)), name='W')
        B = tf.Variable(tf.constant(0.0, shape=[362]))
        model = tf.matmul(tf.reshape(model, [-1, 19 * 19 * 2]), W) + B

    return model


def run_training():
    train_set = data_iterator.HDF5Iterator([f.strip() for f in open(FLAGS.games_list)],
                                           batch_size=FLAGS.batch_size, ndata=1024 * 1024)
    validation_set = data_iterator.HDF5Iterator([f.strip() for f in open(FLAGS.games_list)],
                                                batch_size=FLAGS.batch_size, ndata=1024, validation=True)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    summaries = []

    step_var = tf.Variable(0, trainable=False, name="step")
    inc_step = step_var.assign_add(1)
    in_training = tf.Variable(True, name='in_training', trainable=False)
    label = tf.placeholder(tf.int64, (FLAGS.batch_size,), name="label")
    in_data = tf.placeholder(tf.uint8, (FLAGS.batch_size,) + train_set.lshape, name='input')

    with tf.device('/gpu:0'):
        keep_prob = tf.Variable(0.9, name='keep_prob', trainable=False)

        predictions = model(in_data, in_training, keep_prob,
                            num_modules=FLAGS.num_modules,
                            depth=FLAGS.num_features)


        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(predictions, label), name='lossmean')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), label), "float"))

        lr_var = tf.Variable(FLAGS.learning_rate, name='lr', trainable=False)
        opt = ClippedGDOptimizer(lr_var, 0.9)

        train_op = opt.minimize(loss, colocate_gradients_with_ops=True, aggregation_method=2)

    top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions, label, 5), "float"))
    summaries.append(tf.scalar_summary("loss", loss))
    summaries.append(tf.scalar_summary("accuracy", accuracy))
    summaries.append(tf.scalar_summary("top5", top5))

    summary_op = tf.merge_summary(summaries)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir, sess.graph)

    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if checkpoint:
        print("restoring from checkpoint", checkpoint)
        saver.restore(sess, checkpoint)
        print("Starting from {}".format(sess.run(step_var)))
    else:
        print("Couldn't find checkpoint to restore from.  Starting fresh.")

    while True:
        step_val, = sess.run([inc_step])
        total_loss = 0.0

        # halve learning rate every 50M examples
        cur_learning_rate = FLAGS.learning_rate * (exp(step_val * log(0.5) / ((50 * 1024 * 1024) / train_set.ndata)))

        for batchidx, (batch_input, batch_labels) in enumerate(train_set.minibatches()):
            feed_dict = {in_data: batch_input, label: batch_labels,
                         in_training: True, keep_prob: 0.9,
                         lr_var: cur_learning_rate}

            start_time = time.time()
            _, loss_val, accuracy_val, top5_val, summary_str = sess.run([train_op, loss, accuracy, top5, summary_op],
                                                                        feed_dict=feed_dict)
            total_loss += loss_val
            duration = time.time() - start_time

            print('Train %d: %d/%d: loss = %.3f (avg: %.3f), acc = %.2f, top5 = %.2f, lr = %.3f (%.1f sec)' %
                  (step_val, batchidx, train_set.ndata // train_set.batch_size, loss_val, total_loss /
                   (batchidx + 1), accuracy_val, top5_val, cur_learning_rate, duration))

            summary_idx = train_set.ndata * (step_val - 1) + batchidx * train_set.batch_size + 1
            summary_writer.add_summary(summary_str, summary_idx)
            summary_writer.flush()

        total_loss = 0.0
        for batchidx, (batch_input, batch_labels) in enumerate(validation_set.minibatches()):
            feed_dict = {in_data: batch_input, label: batch_labels, in_training: False, keep_prob: 1.0}

            start_time = time.time()
            loss_val, accuracy_val, top5_val, summary_str = sess.run([loss, accuracy, top5, summary_op],
                                                                     feed_dict=feed_dict)
            total_loss += loss_val
            duration = time.time() - start_time

            print('Validation %d: %d/%d: loss = %.3f (avg: %.3f), acc = %.2f, top5 = %.2f (%.1f sec)' %
                  (step_val, batchidx, validation_set.ndata // validation_set.batch_size, loss_val, total_loss /
                   (batchidx + 1), accuracy_val, top5_val, duration))

        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, "checkpoint"), global_step=step_val)

        if step_val > int(FLAGS.iterations):
            break

    # End train loop
    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, "final-checkpoint"), global_step=step_val)


def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()
