

training_data = file("/mnt/data40T_v2/Peter/DRFNS/src_RealData/RCNN/SmallerDG")
validation_data = 
test_data = 
LEARNING_RATE = [0.1, 0.01, 0.001, 0.0001]
DMC = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
RPN_NMS_THRESHOLD = [0.5, 0.6, 0.7, 0.8, 0.9]

PY = file('NucleiSeg.py')

process training {

    input:
    file DS from training_data
    each lr from LEARNING_RATE
    each dmc from DMC
    each rpn_nms_threshold from RPN_NMS_THRESHOLD
    output:
    file "nucleiseg*", lr, dmc, rpn_nms_threshold into WEIGHTS
    """
    python $PY train --dataset=$DS --subset=train --weight=coco --lr=$lr --DMC $dmc --RPN_NMS_THRESHOLD $rpn_nms_threshold --logs=. 
    """
}

process validation {
    input:
    file DS from validation_data
    file w, _1, _2, _3 from WEIGHTS
    each dmc from DMC
    each rpn_nms_threshold from RPN_NMS_THRESHOLD
    output:
    file file("*.csv") into CSV
    set file("*.csv"), w into test
    """
    python $PY detect --dataset=$DS --subset=val --weight=$w/*20.h5 --logs=. --DMC $dmc --RPN_NMS_THRESHOLD $rpn_nms_threshold
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