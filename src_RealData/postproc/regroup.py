#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
from glob import glob
from optparse import OptionParser


parser = OptionParser()
parser.add_option('--store_best', dest='store_best',type='str')
parser.add_option('--output', dest="output", type="str")

(options, args) = parser.parse_args()

CSV = glob('*.csv')
df_list = []
for f in CSV:
    df = pd.read_csv(f, index_col=False)
    name = f.split('.cs')[0]
    df.index = [name]
    df_list.append(df)
table = pd.concat(df_list)
best_index = table['AJI'].argmax()
table.to_csv(options.output, header=True, index=True)
tmove_name = "{}".format(best_index)
model = "_".join(best_index.split('_')[:-2])
n_feat = model.split('__')[1].split('_')[0]
lr = model.split('__')[1].split('_')[2]
wd = model.split('__')[1].split('_')[1]
name = options.store_best
os.mkdir(name)
os.rename(model, os.path.join(name, model))

p1 = table.ix[best_index, 'p1']
p2 = table.ix[best_index, 'p2']

f1 = open('p1_val', 'w')
f1.write('{}'.format(p1))
f1.close()

f2 = open('p2_val', 'w')
f2.write('{}'.format(p2))
f2.close()

f_feat = open('feat_val', 'w')
f_feat.write('{}'.format(n_feat))
f_feat.close()

f_feat = open('wd_val', 'w')
f_feat.write('{}'.format(wd))
f_feat.close()

f_feat = open('lr_val', 'w')
f_feat.write('{}'.format(lr))
f_feat.close()
