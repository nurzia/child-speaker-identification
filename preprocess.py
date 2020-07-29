"""
e.g.
> python utils.py --reparse
"""

import argparse
import json

from babble import utils


def main():
    parser = argparse.ArgumentParser(description='utilses the data and makes splits')
    parser.add_argument('--annot', type=str,
                        default='data/annotated',
                        help='Location of the annotated data folder')
    parser.add_argument('--unannot', type=str,
                        default='data/unannotated',
                        help='Location of the unannotated data folder')
    parser.add_argument('--CI_unannot', type=str,
                        default='data/CI_unannotated',
                        help='Location of the CI data folder')
    parser.add_argument('--min_len', type=int, default=.2,
                        help='Minimum length for an utterance (default: .2 second)')
    parser.add_argument('--max_len', type=int, default=5,
                        help='Maximum length for an utterance (in seconds)')
    parser.add_argument('--seed', type=int, default=873873,
                        help='Random seed')
    parser.add_argument('--train_size', type=float, default=.8,
                        help='Proportion of train data in splits')
    parser.add_argument('--mel_freq', type=int,
                        default=64,
                        help='Number of frequencies in the MEL-spectrogram. Use zero for no MEL')
    parser.add_argument('--fft_freq', type=int,
                        default=1024,
                        help='Number of frequencies used for the STFT')
    parser.add_argument('--duration', type=int,
                        default=None,
                        help='Only load this many seconds from beginning of file (for dev purposes)')
    parser.add_argument('--hop_length', type=int,
                        default=1024-360,
                        help='Number of elements that one hops by at every iteration')
    parser.add_argument('--win_length', type=int,
                        default=1024,
                        help='number of elements in one STFT frame')
    parser.add_argument('--reparse', default=False, action='store_true',
                        help='Whether to reparse data from scratch')
    parser.add_argument('--CI_reparse', default=False, action='store_true',
                        help='Whether to reparse CI data from scratch')
    parser.add_argument('--viz_annot', default=False, action='store_true',
                        help='Visualize the spectrograms')
    parser.add_argument('--viz_unannot', default=False, action='store_true',
                        help='Visualize the spectrograms of the unannotated files')
    parser.add_argument('--viz_CI_unannot', default=False, action='store_true',
                        help='Visualize the spectrograms of the CI files')
    parser.add_argument('--pad', default=3,
                        help='decides whether to use the "mean" or "max" or specify the exact number of seconds')

    args = parser.parse_args()
    print(args)
    print ('->  parsing CI data')
    if args.reparse:
        print('reparse')
        utils.extract_utterances(dir=args.unannot, min_len=args.min_len,
                                  max_len=args.max_len, duration=args.duration,
                                  annotated=False)
        stats = utils.extract_spectrograms(dir=args.unannot, mel_freq=args.mel_freq,
                                           win_length=args.win_length, fft_freq=args.fft_freq,
                                           hop_length=args.hop_length)
        with open(f'{args.unannot}/stats.json', 'w') as f:
            f.write(json.dumps(stats, indent=4))

    print('-> padding')
    try:
        args.pad = int(args.pad)
        length = int(44100 * args.pad / args.win_length)
        print(length)
    except ValueError:
        with open(f'{args.annot}/stats.json') as stats:
            stats_data = json.load(stats)
            if args.pad == 'max':
                length = int(stats_data['max_len'])
            elif args.pad == 'mean':
                length = int(stats_data['mean_len'])

    # if args.viz_annot:
    #     utils.viz_spectograms(dir=args.annot, length=length)
    # if args.viz_unannot:
    #     utils.viz_spectograms(dir=args.unannot, length=length)
    # if args.viz_CI_unannot:
    #     utils.viz_spectograms(dir=args.CI_unannot, length=length)

    # print('-> splitting annotated data')
    # utils.metadata_split(dir=args.annot, train_size=args.train_size, seed=args.seed)
    # print('-> splitting unannotated data')
    # utils.metadata_split(dir=args.unannot, train_size=args.train_size, seed=args.seed)


if __name__ == '__main__':
    main()
