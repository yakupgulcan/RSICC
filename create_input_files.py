from utils import create_input_files
import time
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare LEVIR-CC-format training data.')
    parser.add_argument('--captions', default='./Levir_CC_dataset/LevirCCcaptions.json')
    parser.add_argument('--images', default='./Levir_CC_dataset/images')
    parser.add_argument('--output_folder', default='./data')
    parser.add_argument('--captions_per_image', type=int, default=5)
    parser.add_argument('--min_word_freq', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=50)
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    print('create_input_files START at: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

    create_input_files(dataset='LEVIR_CC',
                       karpathy_json_path=args.captions,
                       image_folder=args.images,
                       captions_per_image=args.captions_per_image,
                       min_word_freq=args.min_word_freq,
                       output_folder=args.output_folder,
                       max_len=args.max_len)

    print('create_input_files END at: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
