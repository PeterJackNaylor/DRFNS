#!/bin/sh

wget https://zenodo.org/record/1175282/files/TNBC_NucleiSegmentation.zip && \
unzip TNBC_NucleiSegmentation.zip -d TNBC_NucleiSegmentation && \
rm TNBC_NucleiSegmentation.zip

wget http://members.cbio.mines-paristech.fr/~pnaylor/Downloads/pretrained.zip && \
unzip pretrained.zip -d pretrained && \
rm pretrained.zip

wget http://members.cbio.mines-paristech.fr/~pnaylor/Downloads/ForDataGenTrainTestVal.zip && \
unzip ForDataGenTrainTestVal.zip -d ForDataGenTrainTestVal && \
rm ForDataGenTrainTestVal.zip