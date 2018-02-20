from glob import glob
import shutil
from optparse import OptionParser
from utils import CheckOrCreate
from os.path import join
from skimage.io import imread, imsave

parser = OptionParser()
parser.add_option("--path", dest="path", type="string",
                   help="Where to find the path")

(options, args) = parser.parse_args()

dst = 'ImageFolder'
# CheckOrCreate(dst)
shutil.copytree(options.path, dst, symlinks=False, ignore=None)

FILES = glob(join(dst, "GT_*", "GT_*.png"))
AND_TEST = glob(join(dst, "GT_test", "test_*.png"))
for f in FILES + AND_TEST:
    img = imread(f)
    img[img > 0] = 1
    imsave(f, img.astype('int8'))
