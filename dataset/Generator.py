import numpy as np
from ImageGT import ImageGT


class GenerateBatch:
    def __init__(self, dataset, label, batch_size, epoch_num=None, img_op=None, label_op=None, both_op=None):
        self.dataset = dataset
        self.label = label
        self.batch_size = batch_size
        self.start = 0
        self.epoch_num = epoch_num
        self.img_op = img_op
        self.label_op = label_op
        self.both_op = both_op

    def generate_next_batch(self):
        if self.epoch_num is not None:
            for i in range(self.epoch_num):
                print 'Epoch: %d / %d' % (i, self.epoch_num)
                while self.start < len(self.dataset):
                    cur_image_batch = self.dataset[self.start: self.start + self.batch_size]
                    cur_label_batch = self.label[self.start: self.start + self.batch_size]
                    self.start += self.batch_size
                    if self.img_op is not None:
                        cur_image_batch = [np.expand_dims(np.expand_dims(self.img_op(cur_image), axis=2), axis=3) for cur_image in
                                           cur_image_batch]
                        cur_image_batch = np.concatenate(cur_image_batch, axis=3)
                    if self.label_op is not None:
                        cur_label_batch = [np.expand_dims(np.expand_dims(self.label_op(cur_label), axis=2), axis=3) for cur_label in
                                           cur_label_batch]
                        cur_label_batch = np.concatenate(cur_label_batch, axis=3)
                    if self.both_op is not None:
                        cur_image_batch, cur_label_batch = self.both_op(cur_image_batch, cur_label_batch)
                    yield ImageGT(cur_image_batch, cur_label_batch)
        else:
            while True:
                cur_image_batch = self.dataset[self.start: self.start + self.batch_size]
                cur_label_batch = self.label[self.start: self.start + self.batch_size]
                self.start = (self.start + self.batch_size) % len(self.dataset)
                if self.img_op is not None:
                    cur_image_batch = [np.expand_dims(np.expand_dims(self.img_op(cur_image), axis=2), axis=3) for
                                       cur_image in
                                       cur_image_batch]
                    cur_image_batch = np.concatenate(cur_image_batch, axis=3)
                if self.label_op is not None:
                    cur_label_batch = [np.expand_dims(np.expand_dims(self.label_op(cur_label), axis=2), axis=3) for
                                       cur_label in
                                       cur_label_batch]
                    cur_label_batch = np.concatenate(cur_label_batch, axis=3)
                if self.both_op is not None:
                    cur_image_batch, cur_label_batch = self.both_op(cur_image_batch, cur_label_batch)
                yield ImageGT(cur_image_batch, cur_label_batch)
