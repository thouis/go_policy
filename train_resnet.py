import tensorflow as tf
from tensorflow.python import control_flow_ops
from batchnorm import batch_norm
import training_set
import time
import os.path

from malis import malis_node

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', None, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('iterations', 2000, 'Number of epochs to train (1M examples per epoch).')
flags.DEFINE_integer('num_modules', None, 'Number of resnet modules.')
flags.DEFINE_integer('num_features', None, 'Feature map depth.')
flags.DEFINE_string('summary_dir', None, 'Where to put the summary logs')
flags.DEFINE_string('checkpoint_dir', None, 'Where to put checkpoint files')
flags.DEFINE_string('games_list', None, 'file with list of HDF5s to train from')

def conv2d_bn(data, depth, filter_size, training_switch, **kwargs):
    kwargs['bias'] = False  # batch normalization obviates biases
    return batch_norm(conv2d(data, depth, filter_size, **kwargs),
                      depth, training_switch)

def res_3x3_pair(data, depth, scope_name, training_switch, keep_prob):
    # following He 2016, we do BN and activation on the input
    with tf.variable_scope(scope_name):
        # BN and activation on input
        normed = tf.nn.relu(batch_norm(data, depth, training_switch))
        with tf.variable_scope("layer_1"):
            conv1 = conv2d_bn(normed, depth, (3, 3), training_switch, activation=tf.nn.relu)
            conv1 = tf.dropout(conv1, keep_prob)
        with tf.variable_scope("layer_2"):
            conv2 = conv2d(conv1, depth, (3, 3), bias=True)
    return conv2 + data

def model(input_data, training_switch, num_modules=20, depth=64, keep_prob):
    with tf.variable_scope("input"):
        # [filter_height, filter_width, in_channels, out_channels]
        model = conv2d(image, depth, (5, 5), bias=True)

    print("{} modules".format(num_modules))
    for idx in range(num_modules):
        model = res_3x3_pair(model, depth, "resnet_module_{}".format(idx + 1), training_switch)

    with tf.variable_scope("down8"):
        model = conv2d_bn(model, 8, (3, 3), training_switch, activation=tf.nn.relu)
    with tf.variable_scope("down2"):
        model = conv2d_bn(model, 2, (1, 1), training_switch, activation=tf.nn.relu)

    # output is 362 softmax: 361 possible moves, plus pass move
    with tf.variable_scope("output"):
        W = tf.Variable(tf.truncated_normal([19 * 19 * 2, 362], stddev=(1.0 / 361 * 2)), name='W')
        B = tf.Variable(tf.Constant(0.0, shape=[362]))
        model = tf.softmax(tf.matmul(model.reshape([-1, 19 * 19 * 2]), W) + B)

    return model

def run_training():
    data_set = training_set.Training(FLAGS.hdf5, im_size, use_malis=FLAGS.malis)
    sess = tf.Session()
    summaries = []

    im = tf.placeholder(tf.float32, [FLAGS.batch_size, im_size, im_size, 1])
    in_training = tf.Variable(True, name='in_training')

    predictions = model(im, in_training, num_modules=FLAGS.num_modules, malis=FLAGS.malis, shape=[FLAGS.batch_size, im_size, im_size])

    if not FLAGS.malis:
        pos_labels = tf.placeholder(tf.float32, [FLAGS.batch_size, im_size, im_size, 1])
        neg_labels = tf.placeholder(tf.float32, [FLAGS.batch_size, im_size, im_size, 1])
        log_pos = tf.log(predictions + 0.01)
        log_neg = tf.log(1.0 - predictions + 0.01)

        loss = -(tf.reduce_sum(pos_labels * log_pos) / tf.reduce_sum(pos_labels) +
                 tf.reduce_sum(neg_labels * log_neg) / tf.reduce_sum(neg_labels)) / 2.0

        accuracy = (tf.reduce_sum(tf.round(pos_labels * predictions)) / tf.reduce_sum(pos_labels) +
                    tf.reduce_sum(tf.round(neg_labels * (1 - predictions))) / tf.reduce_sum(neg_labels)) / 2.0

        num_pos = tf.reduce_sum(pos_labels)
        num_neg = tf.reduce_sum(neg_labels)

        summaries.append(tf.scalar_summary("loss", loss))
        summaries.append(tf.scalar_summary("accuracy", accuracy))
        summaries.append(tf.image_summary("input", im))
        summaries.append(tf.image_summary("pred", predictions))
        summaries.append(tf.image_summary("pos", pos_labels))
        summaries.append(tf.image_summary("neg", neg_labels))
    else:
        gt = tf.placeholder(tf.int32, [FLAGS.batch_size, im_size, im_size, FLAGS.depth])
        malis_loss_image = - malis_node(gt, predictions)
        loss = tf.reduce_mean(tf.reduce_sum(malis_loss_image, [1, 2])) / (im_size ** 2)
        summaries.append(tf.scalar_summary("loss", loss))
        summaries.append(tf.image_summary("input", im))
        summaries.append(tf.image_summary("pred", tf.reshape(predictions, [FLAGS.batch_size, im_size, im_size, 3])))

    step_var = tf.Variable(0)
    inc_step = step_var.assign_add(1)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9)
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
        if not FLAGS.malis:
            batch_ims, batch_pos_labels, batch_neg_labels = data_set.batch(FLAGS.batch_size)
            feed_dict = {im: batch_ims, pos_labels: batch_pos_labels, neg_labels: batch_neg_labels}

            start_time = time.time()
            _, step, loss_value, accuracy_value, pn, nn, summary_str = sess.run([train_op, inc_step,
                                                                                 loss, accuracy, num_pos, num_neg, summary_op], feed_dict=feed_dict)
        else:
            batch_ims, batch_gt = data_set.batch(FLAGS.batch_size)
            feed_dict = {im: batch_ims, gt: batch_gt}

            start_time = time.time()
            _, step, loss_value, summary_str = sess.run([train_op, inc_step, loss, summary_op], feed_dict=feed_dict)

        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, "checkpoint"), global_step=step)

        duration = time.time() - start_time

        if not FLAGS.malis:
            print('Step %d: loss = %.3f, acc = %.2f (pos,neg labels: %d, %d, %.1f sec)' %
                  (step, loss_value, accuracy_value, pn, nn, duration))
        else:
            print('Step %d: loss = %.3f (%.1f sec)' %
                  (step, loss_value, duration))

        if step >= int(FLAGS.iterations):
            break

    # End train loop
    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, "final-checkpoint"), global_step=step)

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()
