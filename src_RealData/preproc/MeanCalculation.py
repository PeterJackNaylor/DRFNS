import numpy as np
from Data.DataGenClass import DataGenMulti
from Data.ImageTransform import ListTransform
from os.path import join
from optparse import OptionParser


if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option("--path", dest="path",type="string",
                      help="path to annotated dataset")
    parser.add_option("--output", dest="out",type="string",
                      help="out path")

    (options, args) = parser.parse_args()

    path = options.path
    transf, transf_test = ListTransform()

    size = (1000, 1000)
    size_test = (512, 512)
    crop = 1
    DG = DataGenMulti(path, crop=crop, size=size, transforms=transf_test,
                 split="train", num="test")
    DG_test = DataGenMulti(path, crop=crop, size=size_test, transforms=transf_test,
                 split="test", num="test")
    res = np.zeros(shape=3, dtype='float')
    count = 0
    for i in range(DG.length):
        key = DG.NextKeyRandList(0)
        res += np.mean(DG[key][0], axis=(0, 1))
        count += 1
    for i in range(DG_test.length):
        key = DG_test.NextKeyRandList(0)
        res += np.mean(DG_test[key][0], axis=(0, 1))
        count += 1
    mean = res / count
    np.save(join(options.out, "mean_file.npy"), mean)

