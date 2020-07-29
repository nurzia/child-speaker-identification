"""
Train a simple, supervised model
"""

import argparse
import os
import random
random.seed(343455)

from babble import loading2
from babble import modelling
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def main():
    parser = argparse.ArgumentParser(description='Trains a simple supervised model')
    parser.add_argument('--folder', type=str,
                        default='data/unannotated',
                        help='Location of the annotated data folder')
    parser.add_argument('--seed', type=int, default=874873,
                        help='Random seed')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Random seed')
    parser.add_argument('--pad', default='interval', type=str,
                        help='decides whether to mean-pad or max-pad')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--min_red', type=float, default=None,
                        help='minimum length of utterances included in seconds')
    parser.add_argument('--max_red', type=float, default=None,
                        help='maximum length of utterances included in seconds')
    parser.add_argument('--model_file', type=str,
                        default='new_model.h5',
                        help='file to save the trained model on')
    parser.add_argument('--cutoff', type=int, default=750,
                        help='number of utterances per child')
    parser.add_argument('--threshold', type=int, default=None,
                        help='age threshold')
    parser.add_argument('--below_over', type=str,
                        default=None,
                        help='below or over threshold')

    kids_list = ['att', 'max', 'oon',
                 'mad', 'her', 'vic',
                 'bra', 'wou', 'chl',
                 'lot']

    args = parser.parse_args()
    print(args)
    if args.min_red:
        min_red = int((args.min_red * 44100. - 1024.)/(1024. - 360.) + 1.)
        print('min_red: ', min_red)
    else:
        min_red = None
    if args.max_red:
        max_red = int((args.max_red * 44100. - 1024.)/(1024. - 360.) + 1.)
        print('max_red: ', max_red)
    else:
        max_red = None

    streamer = loading2.DataGenerator(batch_size=args.batch_size,
                                      folder=args.folder,
                                      seed=args.seed,
                                      pad=args.pad,
                                      cutoff=args.cutoff,
                                      CI_exclude=True,
                                      oldcam=False,
                                      width=1,
                                      min_reduction=min_red,
                                      max_reduction=max_red,
                                      kids_list=kids_list,
                                      threshold=args.threshold,
                                      below_over=args.below_over)
    streamer.fit_scaler()

    # shape info from streamer:
    num_classes = len(streamer.encoder.classes_)
    print('num classes:', num_classes)
    feat_dim = streamer.scaler.mean_.shape[0]
    spec_len = streamer.pad_length
    print('spectrogram length:', spec_len)

    if os.path.isfile(args.model_file):
        os.remove(args.model_file)

    model = modelling.build_model_(num_classes=num_classes,
                                   spec_len=spec_len, feat_dim=feat_dim)

    checkpoint = ModelCheckpoint(args.model_file, monitor='val_loss',
                                 verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                  patience=1, min_lr=0.00001,
                                  verbose=1, min_delta=0.001)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    adam = optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999,
                           epsilon=None, decay=0.0, amsgrad=False,
                           clipvalue=0.5, clipnorm=1.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['categorical_accuracy'])

    # fit model to data
    model.fit_generator(streamer.get_batches(stream='train', endless=True),
                        steps_per_epoch=streamer.get_num_batches('train'),
                        epochs=args.num_epochs,
                        validation_data=streamer.get_batches(stream='dev', endless=True),
                        validation_steps=streamer.get_num_batches('dev'),
                        callbacks=[checkpoint, reduce_lr, early_stopping])

    print('>>> evaluating the model:')

    # shape info from streamer:
    num_classes = len(streamer.encoder.classes_)
    print('num classes:' + str(num_classes))
    feat_dim = streamer.scaler.mean_.shape[0]
    spec_len = streamer.pad_length
    print('spectrogram length:' + str(spec_len) + '\n')

    results = model.evaluate_generator(streamer.get_batches(stream='test', endless=True),
                                       steps=streamer.get_num_batches('test'), verbose=1)
    print(results)

    occurrences_array=np.zeros(num_classes)
    total_utt = 0
    for X, lab_batch in streamer.get_batches(stream='train', endless=False):
        for Y in lab_batch:
            occurrences_array[np.argmax(Y)] += 1
            total_utt += 1
    print('percentages: ', occurrences_array/total_utt)

    # mount test-data onto two arrays for confusion matrix and classification report
    labels = []
    batches = []
    for batch, batch_labels in streamer.get_batches(stream='test'):
        for label in batch_labels:
            labels.append(label)
        for b in batch:
            batches.append(b)
    batches = np.array(batches)
    predictions = model.predict(batches)
    labels = np.array(labels)
    uncat_labels = []
    uncat_predictions = []
    for label, prediction in zip(labels, predictions):
        uncat_labels.append(np.argmax(label))
        uncat_predictions.append(np.argmax(prediction))

    # show classification report
    print('classification report\n' +
                  str(classification_report(uncat_labels, uncat_predictions)) + '\n')
    print('confusion matrix\n' +
                  str(confusion_matrix(uncat_labels, uncat_predictions)) + '\n')
    print('accuracy:' + str(accuracy_score(uncat_labels, uncat_predictions)) + '\n')


if __name__ == '__main__':
    main()
