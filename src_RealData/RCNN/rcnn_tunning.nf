

training_data = file("/mnt/data40T_v2/Peter/DRFNS/src_RealData/RCNN/SmallerDG")
validation_data = file("/mnt/data40T_v2/Peter/DRFNS/src_RealData/RCNN/validation") 
test_data = file("/mnt/data40T_v2/Peter/DRFNS/src_RealData/RCNN/test")
LEARNING_RATE = [0.1, 0.01, 0.001, 0.0001]
DMC = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
RPN_NMS_THRESHOLD = [0.5, 0.6, 0.7, 0.8, 0.9]
FORCOCO = file("Mask_RCNN")
PY = file('NucleiSeg.py')

process training {
    queue 'gpu-cbio'
    clusterOptions '--exclude=node[27-28]'
    maxForks 3
    tag { lr + "_" + dmc + "_" + rpn_nms_threshold}
    input:
    file FORCOCO
    file DS from training_data
    each lr from LEARNING_RATE
    each dmc from DMC
    each rpn_nms_threshold from RPN_NMS_THRESHOLD
    output:
    set file("nucleiseg*"), lr, dmc, rpn_nms_threshold into WEIGHTS
    """
    module load cuda90
    source \$HOME/applications_slurm/init_py3
    python $PY train --dataset=$DS --subset=train --weight=coco --lr=$lr --DMC $dmc --RPN_NMS_THRESHOLD $rpn_nms_threshold --logs=. 
    """
}

process validation {
    maxForks 2
    queue 'gpu-cbio'
    clusterOptions '--exclude=node[24-26]'
    input:
    file DS from validation_data
    set file(w), _1, _2, _3 from WEIGHTS
    each dmc from DMC
    each rpn_nms_threshold from RPN_NMS_THRESHOLD
    output:
    file "Mask_RCNN/results/nucleus/*" into VAL_RESULTS
    set _1, _2, _3, w, dmc, rpn_nms_threshold, file("Mask_RCNN/results/nucleus/*") into VAL_PARAM
    """
    module load cuda90
    source \$HOME/applications_slurm/init_py3
    export CUDA_VISIBLE_DEVICES=1
    python $PY detect --dataset=$DS --subset=val --weight=$w/mask_rcnn_nucleiseg_0020.h5 --logs=. --DMC $dmc --RPN_NMS_THRESHOLD $rpn_nms_threshold
    """
}

/*
process find_best {
    input:
    file _ from CSV .collect()
    output:
    file "best_*.csv" into best
    """

    """
}

process test {




}
*/
