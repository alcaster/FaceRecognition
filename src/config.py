import argparse
from utils.argparse_utils import SmartFormatter

parser = argparse.ArgumentParser(description='', formatter_class=SmartFormatter)

############################
#   training setting    #
############################
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--test_size', type=float, default=0.2, help='Percentage for test set. F.ex 0.2')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')

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
                    help="""R|Directory where train set is stored. Format of directory:
data_path
│   class1
│   │   example1.png
│   │   example2.png
│   class2
│   │   example1.png""")
parser.add_argument('--log_dir', type=str, default='../out/1', help='Directory where saved checkpoints will be stored.')

############################
#   testing setting    #
############################
parser.add_argument('--test', action='store_true', help="Test? Default->training.")
parser.add_argument('--test_set_path', type=str, default='../data/test',
                    help='Same format as data_path')
parser.add_argument('--checkpoint_path', type=str, default='../out/1', help='')
parser.add_argument('--meta_path', type=str, default='../out/1/model-7401.meta', help='')

FLAGS, unparsed = parser.parse_known_args()
