#!/usr/bin/env nextflow

// General parameters
params.image_dir = './datafolder'
params.epoch = 1
IMAGE_FOLD = file(params.image_dir + "/Normalized_ForDataGenTrainTestVal")
/*          0) a) Resave all the images so that they have 1 for label instead of 255 
            0) b) Resave all the images so that they are distance map
In outputs:
newpath name
*/

CHANGESCALE = file('src_RealData/preproc/changescale.py')
NAMES = ["FCN", "UNet"]

process ChangeInput {
    input:
    file path from IMAGE_FOLD
    file changescale from CHANGESCALE
    each name from NAMES
    output:
    set val("$name"), file("ImageFolder") into IMAGE_FOLD2, IMAGE_FOLD3 
    """
    python $changescale --path $path

    """
}

BinToDistanceFile = file('src_RealData/preproc/BinToDistance.py')

process BinToDistance {
    input:
    file py from BinToDistanceFile
    file path from IMAGE_FOLD
    output:
    set val("DIST"), file("ToAnnotateDistance") into DISTANCE_FOLD, DISTANCE_FOLD2

    """
    python $py $path
    """
}

/*          1) We create all the needed records 
In outputs:
a set with the name, the split and the record
*/

TFRECORDS = file('src_RealData/TFRecords.py')
IMAGE_FOLD2 .concat(DISTANCE_FOLD) .into{FOLDS;FOLDS2}
UNET_REC = ["UNet", "--UNet", 212]
FCN_REC = ["FCN", "--no-UNet", 224]
DIST_REC = ["DIST", "--UNet", 212]

RECORDS_OPTIONS = Channel.from(UNET_REC, FCN_REC, DIST_REC)
FOLDS.join(RECORDS_OPTIONS) .set{RECORDS_OPTIONS_v2}
RECORDS_HP = [["train", "16", "0"], ["fulltrain", "16", "0"], ["validation", "1", 996], ["test", "1", 996]]

process CreateRecords {
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    set name, file(path), unet, size_train from RECORDS_OPTIONS_v2
    each op from RECORDS_HP
    output:
    set val("${name}"), val("${op[0]}"), file("${op[0]}_${name}.tfrecords") into NSR0, NSR1, NSR2, NSR3
    """
    python $py --tf_record ${op[0]}_${name}.tfrecords --split ${op[0]} --path $path --crop ${op[1]} $unet --size_train $size_train --size_test ${op[2]} --seed 42 --epoch 1 --type JUST_READ 
    """
}

NSR0.filter{ it -> it[1] == "train" }.set{TRAIN_REC}
NSR1.filter{ it -> it[1] == "validation" }.set{VAL_REC}
NSR2.filter{ it -> it[1] == "test" }.set{TEST_REC}
NSR3.filter{ it -> it[1] == "fulltrain"}.set{FULLTRAIN_REC}
/*          2) We create the mean
In outputs:
a set with the name, the split and the record
*/

MEANPY = file('src_RealData/preproc/MeanCalculation.py')

process Mean {
    input:
    file py from MEANPY
    set val(name), file(toannotate) from FOLDS2
    output:
    set val("$name"), file("mean_file.npy"), file("$toannotate") into MeanFile, Meanfile2, Meanfile2VAL, Meanfile3, Meanfile3VAL, MeanFileFull
    """
    python $py --path $toannotate --output .
    """
}

/*          3) We train
In inputs: Meanfile, name, split, rec
In outputs:
a set with the name, the parameters of the model
*/

ITERVAL = 4
ITER8 = 10800
LEARNING_RATE = [0.001, 0.0001, 0.00001]//, 0.000001]
FEATURES = [16, 32, 64]
WEIGHT_DECAY = [0.000005, 0.00005, 0.0005]
BS = 10

Unet_file = file('src_RealData/UNet.py')
Fcn_file = file('src_RealData/FCN.py')
Dist_file = file('src_RealData/Dist.py')

UNET_TRAINING = ["UNet", Unet_file, 212, 0]
FCN_TRAINING  = ["FCN", Fcn_file, 224, ITER8]
DIST_TRAINING = ["DIST", Dist_file, 212, 0]

Channel.from(UNET_TRAINING, FCN_TRAINING, DIST_TRAINING) .into{ TRAINING_CHANNEL; TRAINING_CHANNEL2; TRAINING_CHANNELFULL}
PRETRAINED_8 = file(params.image_dir + "/pretrained/checkpoint16/")
TRAIN_REC.join(TRAINING_CHANNEL).join(MeanFile) .set {TRAINING_OPTIONS}

process Training {
    beforeScript "source \$HOME/CUDA_LOCK/.whichNODE"
    afterScript "source \$HOME/CUDA_LOCK/.freeNODE"
    input:
    set name, split, file(rec), file(py), size, iters, file(mean), file(path) from TRAINING_OPTIONS
    val bs from BS
    each feat from FEATURES
    each lr from LEARNING_RATE
    each wd from WEIGHT_DECAY   
    file __ from PRETRAINED_8
    val epoch from params.epoch
    output:
    set val("$name"), file("${name}__${feat}_${wd}_${lr}"), file("$py"), feat, wd, lr into RESULT_TRAIN, RESULT_TRAIN2, RESULT_TRAIN_VAL, RESULT_TRAIN_VAL2
    when:
    "$name" != "FCN" || ("$feat" == "${FEATURES[0]}" && "$wd" == "${WEIGHT_DECAY[0]}")
    script:
    """
    python $py --tf_record $rec --path $path  --log ${name}__${feat}_${wd}_${lr} --learning_rate $lr --batch_size $bs --epoch $epoch --n_features $feat --size_train $size --weight_decay $wd --mean_file ${mean} --n_threads 100 --restore $__  --split $split --iters $iters
    """
} 

/*          4) a) We choose the best hyperparamter with respect to the test data set

In inputs: Meanfile, image_path resp., split, rec, model, python, feat
In outputs: a set with the name and model or csv
*/
// a)
P1 = [0, 1, 10, 11]//[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
P2 = [0.5, 1.0] //, 1.5, 2.0]
VAL_REC.cross(RESULT_TRAIN).map{ first, second -> [first, second.drop(1)].flatten() } .set{ VAL_OPTIONS_pre }
Meanfile2.cross(VAL_OPTIONS_pre).map { first, second -> [first, second.drop(1)].flatten() } .into{VAL_OPTIONS;VAL_OPTIONS2}

process Validation {
    beforeScript "source \$HOME/CUDA_LOCK/.whichNODE"
    afterScript "source \$HOME/CUDA_LOCK/.freeNODE"
    input:
    set name, file(mean), file(path), split, file(rec), file(model), file(py), feat, wd, lr from VAL_OPTIONS    
    each p1 from P1
    each p2 from P2
    val iters from ITERVAL
    output:
    set val("$name"), file("${name}__${feat}_${wd}_${lr}_${p1}_${p2}.csv") into RESULT_VAL
    set val("$name"), file("$model") into MODEL_VAL
    when:
    ("$name" =~ "DIST" && p1 < 6) || ( !("$name" =~ "DIST") && p2 == P2[0] && p1 > 5)
    script:
    """
    python $py --tf_record $rec --path $path  --log $model --batch_size 1 --n_features $feat --mean_file ${mean} --n_threads 100 --split $split --size_test 996 --p1 ${p1} --p2 ${p2} --restore $model --iters $iters --output ${name}__${feat}_${wd}_${lr}_${p1}_${p2}.csv
    """  

}


/*          5) We regroup a) the test on dataset 1
In inputs: name, all result_test.csv per key
In outputs: name, best_model, p1, p2
*/
// a)
REGROUP = file('src_RealData/postproc/regroup.py')
RESULT_VAL  .groupTuple() 
             .set { KEY_CSV }
RESULT_TRAIN2.map{name, model, py, feat, wd, lr -> [name, model]} .groupTuple() . set {ALL_MODELS}
KEY_CSV .join(ALL_MODELS) .set {KEY_CSV_MODEL}

process GetBestPerKey {
    publishDir "./out_RDS/Validation_tables/" , pattern: "*.csv"
    input:
    file py from REGROUP
    set name, file(csv), file(model) from KEY_CSV_MODEL

    output:
    set val("$name"), file("best_model") into BEST_MODEL_VAL
    set val("$name"), 'feat_val', 'wd_val', 'lr_val', 'p1_val', 'p2_val' into PARAM
    file "${name}_validation.csv"
    """
    python $py --store_best best_model --output ${name}_validation.csv
    """
}

N_FEATS .map{ it.text } .set {FEATS_}
WD_VAL  .map{ it.text } .set {WD_}
LR_VAL  .map{ it.text } .set {LR_}
P1_VAL  .map{ it.text } .set {P1_}
P2_VAL  .map{ it.text } .set {P2_}

FULLTRAIN_REC. join(TRAINING_CHANNELFULL) .join(MeanFileFull) .set{FULL_RECORD}
FULL_RECORD.join( PARAM ).set{FTRAINING_PARAM}


process FullTraining {
    beforeScript "source \$HOME/CUDA_LOCK/.whichNODE"
    afterScript "source \$HOME/CUDA_LOCK/.freeNODE"
    input:
    set name, split, file(rec), file(py), size, iters, file(mean), file(path), feat, wd, lr, p1, p2 from FTRAINING_PARAM
    val bs from BS
    file __ from PRETRAINED_8
    val epoch from params.epoch
    output:
    set val("$name"), file("${name}__${feat.text}_${wd.text}_${lr.text}"), "${feat.text}", "${wd.text}", "${lr.text}", "p1.text", "p2.text" into RESULT_FULLTRAIN
    script:
    """
    python $py --tf_record $rec --path $path  --log ${name}__${feat.text}_${wd.text}_${lr.text} \\ 
               --learning_rate ${lr.text} --batch_size $bs --epoch $epoch --n_features ${feat.text} \\
               --size_train $size --weight_decay ${wd.text} --mean_file ${mean} --n_threads 100 --restore $__  --split $split --iters $iters
    """
} 

/*
Compute test score on test set
a) Test with hyper parameter choosen on validation dataset

*/
// a)
BEST_MODEL_VAL.join(TRAINING_CHANNEL2).join(Meanfile3) .set{ TEST_OPTIONS}
N_FEATS .map{ it.text } .set {FEATS_}
P1_VAL  .map{ it.text } .set {P1_}
P2_VAL  .map{ it.text } .set {P2_}

process Test {
    beforeScript "source \$HOME/CUDA_LOCK/.whichNODE"
    afterScript "source \$HOME/CUDA_LOCK/.freeNODE"
    publishDir "./out_RDS/Test/"
    input:
    set name, file(best_model), file(py), _, __, file(mean), file(path) from TEST_OPTIONS
    val feat from FEATS_ 
    val p1 from P1_
    val p2 from P2_
    output:
    file "./$name"
    file "${name}.csv" into CSV_TEST
    """
    python $py --mean_file $mean --path $path --log $best_model --restore $best_model --batch_size 1 --n_features ${feat} --n_threads 100 --split test --size_test 996 --p1 ${p1} --p2 ${p2} --output ${name}.csv --save_path $name
    """
}

PLOT = file('src_RealData/postproc/plot.py')

process Plot {
    publishDir "./out_RDS/Validation/"
    input:
    file _ from CSV_TEST .collect()
    file py from PLOT
    output:
    file "BarResult_train_test_val.png"
    """
    python $py --output BarResult_train_val_test.png --output_csv Result_train_val_test.csv
    """
}
