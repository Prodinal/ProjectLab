
import functools
import glob
import itertools
import multiprocessing
import random
import sys
import time

import cv2
import numpy
import tensorflow as tf

import model
import common

# Creates vector from given code (i.e. Malignant or Benign)
# Todo later
def code_to_vec(code):
    def label_to_vec(c):
        y = numpy.zeros((len(common.OUTCOMES),))
        y[common.OUTCOMES.index(c)] = 1.0
        return y

    c = numpy.vstack([label_to_vec(code)])
    return c.flatten()
    # return numpy.reshape(c.flatten(), (-1,2))


def read_data(img_glob_positive, img_glob_negative):

    for fname in sorted(glob.glob(img_glob_positive)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255
        # Todo: figure out how to code label in name
        code = "Benign"
        yield im, code_to_vec(code)
    for fname in sorted(glob.glob(img_glob_negative)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255
        # Todo: figure out how to code label in name
        code = "Malignant"
        yield im, code_to_vec(code)


def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


def batch(it, batch_size):
    out = []
    for x in it:
        out.append(x)
        if len(out) == batch_size:
            yield out
            out = []
    if out:
        yield out

def mpgen(f):
    def main(q, args, kwargs):
        try:
            for item in f(*args, **kwargs):
                q.put(item)
        finally:
            q.close()

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        q = multiprocessing.Queue(3) 
        proc = multiprocessing.Process(target=main,
                                       args=(q, args, kwargs))
        proc.start()
        try:
            while True:
                item = q.get()
                yield item
        finally:
            proc.terminate()
            proc.join()

    return wrapped

@mpgen
def read_batches(batch_size):
    g = gen.generate_ims()
    def gen_vecs():
        for im, c, p in itertools.islice(g, batch_size):
            yield im, code_to_vec(p, c)
    

# Todo
def get_loss(y, y_):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                                        labels=y_, 
                                                        logits=y))


def read_training_batch(batch_size):
    train_ims, train_ys = unzip(list(read_data("Nodules/Positive/*.jpg", "Nodules/Negative/*.jpg"))[common.TEST_NUM+1:])	#Remaining images for training
    index_shuf = range(len(train_ims))
    shuffleable = [[i] for i in range(10)]
    random.shuffle(shuffleable)
    for i in itertools.islice(shuffleable, batch_size):
        # print(train_ys[i])
        yield train_ims[i], train_ys[i]


def train(learn_rate, report_steps, batch_size, initial_weights=None):
    x,y,params = model.get_training_model()

    y_ = tf.placeholder(tf.float32, [None, len(common.OUTCOMES)])

    loss = get_loss(y, y_)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    best = tf.argmax(tf.reshape(y[:,:], [-1, 1, len(common.OUTCOMES)]), 2)
    correct = tf.argmax(tf.reshape(y_[:,1:], [-1, 1, len(common.OUTCOMES)]), 2)

    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

    init = tf.initialize_all_variables()

    def vec_to_result(v):
        return "".join(common.OUTCOMES[i] for i in v)

    def do_report():
        #r = sess.run([best,
        #                correct,
        #                tf.greater(y[:, 0], 0),
        #                y_,
        #                loss],
        #                feed_dict = {x: test_xs, y_: test_ys})
        r = sess.run(y, feed_dict = {x: test_xs, y_: test_ys})
        print('Outcome for testing images: {}'.format(r))

        #num_correct = numpy.sum(r[0] == r[1], axis=0)

        #r_short = (r[0][:190], r[1][:190], r[2][:190], r[3][:190])
        #for b, c, pb, pc in zip(*r_short):
        #    print("{} {} <-> {} {}".format(vec_to_result(c), pc,
        #                                   vec_to_result(b), float(pb)))

        #num_p_correct = numpy.sum(r[2] == r[3])

        #print ('B{:3d} {:2.02f}% {:02.02f}% loss: {} (digits: {}, presence: {}) |{}|').format(
        #    batch_idx,
        #    100. * num_correct / (len(r[0])),
        #    100. * num_p_correct / len(r[2]),
        #    r[6],
        #    r[4],
        #    r[5],
        #    "".join("X "[numpy.array_equal(b, c) or (not pb and not pc)]
        #                                   for b, c, pb, pc in zip(*r_short)))
        
    def do_batch():
        # print(batch_xs.shape)
        # print(batch_ys.shape)
        sess.run(train_step,
                 feed_dict={x: batch_xs, y_:batch_ys})
        if batch_idx % report_steps == 0:
            do_report()
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        if initial_weights is not None:
            sess.run(assign_ops)

        test_xs, test_ys = unzip(list(read_data("Nodules/Positive/*.jpg", "Nodules/Negative/*.jpg"))[:common.TEST_NUM])		#First some images for testing
        
        try:
            while True:
                last_batch_idx = 0
                last_batch_time = time.time()
                batch_iter = enumerate(read_training_batch(batch_size))

                for batch_idx, (batch_xs, batch_ys) in batch_iter:
                    do_batch()
                    if batch_idx % report_steps == 0:
                        batch_time = time.time()
                        if last_batch_idx != batch_idx:
                            print("time for 60 batches {}".format(
                                60*(last_batch_time - batch_time) /
                                            (last_batch_idx - batch_idx)))
                            last_batch_idx = batch_idx
                            last_batch_time = batch_time
        
        except KeyboardInterrupt:
            last_weights = [p.eval() for p in params]
            numpy.savez("weights.npz", *last_weights)
            return last_weights

if __name__ == '__main__':
    if len(sys.argv) > 1:
        f = numpy.load(sys.argv[1])
        initial_weights = [f[n] for n in sorted(f.files,
                                                key=lambda s: int(s[4:]))]
    else:
        initial_weights = None

    train(learn_rate=0.001,
          report_steps=20,
          batch_size=50,
          initial_weights=initial_weights)

# if __name__ == "__main__":
#	a=open ('Nodules/Positive/PlsShowUp.txt' , 'w')