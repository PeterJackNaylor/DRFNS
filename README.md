[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1175282.svg)](https://doi.org/10.5281/zenodo.1175282)
# Deep Regression For Nuclei Segmentation

In this repository, we've implemented the code resulting in the submitted paper "Segmentation of Nuclei in Histopathology Images by deep regression of the distance map." in TMI, by P. Naylor, M. Laé, F. Reyal and T. Walter. The code in this repository is made out of nextflow for pipeline management. For more information about nextflow please refer to [Nextflow's documentation](https://www.nextflow.io/). Each process of this pipeline is made out of python and called via nextflow. Nextflow was very handy to take advantage of a cluster with multi-queues, in particular for a pipeline using CPU's and GPU's.

## Description
We tackle the task of nuclei segmentation within histopathology tissue. Many methods have been proposed in the past but relatively few of them try to handle the wide hetereogenity that one can typically encounter with such data. As is the current trend, we apply state of the art algorithm technics based on CNN but also our novel nuclei segmentation framework. We compare each method with two metrics, the first pixel based and the second objected based. We elaborate a benchmark of fully convolutionnal algorithm applied to these datasets.  In particular we compare ourselves to [\[Neeraj et al\]](https://nucleisegmentationbenchmark.weebly.com/) and show that our deep regression method outperforms previous state of the art method for separating cluttered objects. 

# Setup
To setup, please [install](https://www.nextflow.io/docs/latest/getstarted.html) nextflow and configure it to your setup by configuring the nextflow.config file. 
This code works for python 2.7.11 and tensorflow 1.5. They may be other requirements in terms of python packages but nothing too specific. Install with ```conda install ``` or ```pip install```. 
In addition, if one wishes to reproduce the results achieved with FCN, please download and add to the PYTHONPATH this [directory](https://github.com/warmspringwinds/tf-image-segmentation).
# Hardware
This code was run on a K80 GPU. A K80 has a bi-heart and that is why ```maxForks 2``` in the nextflow files. Also do not hesitate to modify the processes environnement. For instance, to assure that jobs were running on seperate nodes, I used the options ```beforeScript``` and ```afterScript```. The scripts called at these moment are just locks. If one job launches first it will create a lock (for example for GPU number 0) to alert other jobs that he is using a GPU (number 0). Removes these lines if you do not need them.
# Data 
The data made publicaly available by our institute can be found [here](https://zenodo.org/record/1175282/files/TNBC_NucleiSegmentation.zip). If you want to run the code by yourself please run the file download_metadata.sh, ```bash download_metadata.sh```. This will download *DS1* and *DS2* as described in the paper, moreover it will download the pretrained weights for the FCN model. 
By running this script, the image file will be subdivided into groups as described in the paper. 
One could also find the data annouced publicaly available by checking out the website created by the authors [\[Neeraj et al\]](https://nucleisegmentationbenchmark.weebly.com/).
# Running the pipeline with synthetic data
To run this pipeline you will have to run the following command: ```nextflow run dummydataset.nf --epoch 10 -c nextflow.config -resume```
This command will call the nextflow pipeline script ```dummydataset.nf``` which is subdivided into 3 process: 
1) Elaboration of the dummy data 
2) Creating tensorflow records
3) Training the different designs with fixed hyper parameters.
Would be nice to insert diagram picture of the pipeline.
# Running the pipeline with the real data
To run this pipeline you will have to run the following command: ```nextflow run realdataset.nf --epoch 80 -c nextflow.config -resume```
This command will call the nextflow pipeline script ```realdataset.nf``` which is subdivided into 4 steps: 
1) Preparing the data for the experiments, processes: *ChangeInput*, *BinToDistance*, *Mean* and *CreateRecords*.
2) Training
3) Validating the different models, processes: *Testing*, *GetBestPerKey*, *Validation* and *plot*.
Would be nice to insert diagram picture of the pipeline.
# Running with your own data
If you wish to run the pipeline with your own data please specify your data folder so they follow the same rules as those found in ```./datafolder/ ```. Further more, you can tweak how the model loads the data by modifying the data generator class that can be found in ```./DataGen/ ```.