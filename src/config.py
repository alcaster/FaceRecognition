import argparse

parser = argparse.ArgumentParser()

############################
#   training setting    #
############################
parser.add_argument('--train', type=bool, default=True, help='Is training?')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--test_size', type=float, default=0.2, help='Percentage for test set. F.ex 0.2')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')

############################
#   Image preprocessing    #
############################
parser.add_argument('--img_size', type=int, default=220, help='Size of each image.')
parser.add_argument('--num_parallel', type=int, default=8, help='Amount of paralleling processed pictures.')
parser.add_argument('--buffer_size', type=int, default=100, help='Buffer size for image processing.')

############################
#   environment setting    #
############################
parser.add_argument('--data_path', type=str, default='../data/train',
                    help='The path to the train dataset.\n Each class should be in separate directory.')
parser.add_argument('--log_dir', type=str, default='../out/2', help='Directory where saved checkpoints will be stored.')

FLAGS, unparsed = parser.parse_known_args()
