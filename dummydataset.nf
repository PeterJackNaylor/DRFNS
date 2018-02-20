#!/usr/bin/env nextflow

params.toannotate = file('datafolder/TNBC_NucleiSegmentation')

PYDS = file('src_DummyDataSet/DDSCreation.py')
TFRecordPY = file('src_DummyDataSet/Records.py')

UNET_NORMALIZED_LOGIT = file('src_DummyDataSet/UNet_Logit_Normalized.py')
UNET_NORMALIZED = file('src_DummyDataSet/UNet_Normalized.py')
UNET_UNNORMALIZED = file('src_DummyDataSet/UNet_UNNormalized.py')

target = Channel.from( [0, UNET_NORMALIZED_LOGIT], [0, UNET_NORMALIZED], [1, UNET_UNNORMALIZED])

def getFileName( file ) {
      if ( file.name == "Normalized" ){
      	  0
      } else {
      	  1
      }
}

VAL_NAME = [[0, "Normalized"], [1, "UNNormalized"]]

process DummyDataSet {
	publishDir "./out_DDS/Data"
	input:
	file path from params.toannotate
	file py from PYDS
	each pair from VAL_NAME
	output:
	file "./${pair[1]}" into PATHS
	"""
	python $py --path $path --output  ./${pair[1]} --test 10 --mu 127 --sigma 100 --sigma2 10 --normalized ${pair[0]}
	"""
}

process CreateTFRecords {
	publishDir "./out_DDS/Records"
	input:
	file py from TFRecordPY
	file path from PATHS
	output:
	set file("$path"), file("${path}.tfrecords") into PATH_RECORDS
	"""
	python $py --output ${path}.tfrecords --path $path --crop 4 --UNet --size 212 --seed 42 --epoch 20 --type JUST_READ --train
	"""
}

PATH_RECORDS  .map { folder, rec -> tuple(getFileName(folder) ,folder, rec) } .cross(target) .set { PATH_RECORDS_FILE }
// PATH_RECORDS_FILE .subscribe{first, second -> println("New parameter:\n") println(first[0]) println(first[1]) println(first[2]) println(second[0]) println(second[1]) println(second[1].baseName) println("\n")}
process Training {
	publishDir "./out_DDS/${second[1].baseName}"
	maxForks 1
	input:
	set first, second from PATH_RECORDS_FILE // first[1] is the path, first[2] is the record, second[1] is the python file
	output:
	file "step_*"


	"""
	python -W ignore ${second[1]} --tf_record ${first[2]} --path ${first[1]} --log . --learning_rate 0.001 --batch_size 4 --epoch 100 --n_features 2 --weight_decay 0.005 --dropout 0.5 --n_threads 50
	"""
}


