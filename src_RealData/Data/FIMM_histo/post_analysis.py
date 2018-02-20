import os, sys

import cellh5

from optparse import OptionParser
import numpy as np
import operator

import pdb

class SimpleAnalyzer(object):
    def __init__(self, input_folder, output_folder):
        print 'SimpleAnalyzer'
        self.input_folder = input_folder
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print 'made %s' % self.output_folder
        return

    def get_positions(self, in_folder):
        filenames = filter(lambda x: os.path.splitext(x)[-1].lower() == '.ch5' and x[0] != '_',
                           os.listdir(in_folder))
        if len(filenames) < 1:
            raise ValueError('no ch5 files found in %s' % in_folder)
            return None
        positions = {}
        for fn in filenames:    
            well, pos = os.path.splitext(fn)[0].split('_')
            if not well in positions:
                positions[well] = []
            positions[well].append(pos)            
        return positions
        
    def get_first_pos(self, in_folder):
        filenames = filter(lambda x: os.path.splitext(x)[-1].lower() == '.ch5' and x[0] != '_',
                           os.listdir(in_folder))
        if len(filenames) > 0:
            return filenames[0]
        else:
            raise ValueError('no ch5 files found in %s' % in_folder)

        return None

    def get_counts(self, plate=None, channel='merged', region='histonucgrey-expanded-expanded',
                   result_folder=None):

        if result_folder is None:
            result_folder = self.input_folder        
        if plate is None:
            plate_folder = result_folder
        else:
            plate_folder = os.path.join(result_folder, plate)

        ch5_folder = os.path.join(plate_folder, 'hdf5')
        if not os.path.isdir(ch5_folder):
            print 'problem with folder settings: hdf5-folder for plate %s not found' % plate
            print 'given folder name: %s' % ch5_folder

        cm_general = cellh5.CH5MappedFile(os.path.join(ch5_folder, self.get_first_pos(ch5_folder)))            
        positions = self.get_positions(ch5_folder)
        
        classes = cm_general.class_definition('%s__%s' % (channel, region))
        class_colors = [x[-1] for x in classes]
        class_names = [x[1] for x in classes]
        class_labels = [x[0] for x in classes]

        res = {}
        
        #when _allpositions workes: for well, poslist in cm_general.positions.iteritems():
        for well, poslist in positions.iteritems():
            res[well] = dict(zip(class_names, [0 for x in classes]))    
                        
            for pos in poslist:
                cm = cellh5.CH5MappedFile(os.path.join(ch5_folder, '%05i_%02i.ch5' % (int(well), int(pos)) ))
                
                # dirty hack:
                #pdb.set_trace()
                lw = cm.positions.keys()[0]
                lpl = cm.positions[lw]
                #pos_obj = cm.get_position(well, pos)
                
                pos_obj = cm.get_position(lw, lpl[0])

                if len(pos_obj['object']['%s__%s' % (channel, region)]) == 0:
                    cm.close()
                    continue
                
                # attention: the function get_class_prediction gives back an index, not a label.
                # the labels start with 1 and the index starts with 0. 
                # but the labels can also be in completely different order.
                predictions = [class_names[x[0]] for x in pos_obj.get_class_prediction('%s__%s' % (channel, region))]
                
                for pred in predictions:
                    res[well][pred] += 1
                                    
                cm.close()
        
        cm_general.close()
        
        return res

    def get_classes(self, plate=None):
        if plate is None:
            plate_folder = result_folder
        else:
            plate_folder = os.path.join(result_folder, plate)
        ch5_folder = os.path.join(plate_folder, 'hdf5')
        if not os.path.isdir(ch5_folder):
            print 'problem with folder settings: hdf5-folder for plate %s not found' % plate
                
        cm = cellh5.CH5MappedFile(os.path.join(ch5_folder, self.get_first_pos(ch5_folder)))

        classes = cm.class_definition('merged__histonucgrey-expanded-expanded')
        class_colors = [x[-1] for x in classes]
        class_names = [x[1] for x in classes]
        class_labels = [x[0] for x in classes]

        cm.close()
        
        return classes

    def export_predictions(self, predictions, filename):
        print self.output_folder
        print filename
        fp = open(os.path.join(self.output_folder, filename), 'w')
        exp_id = predictions.keys()[0]
        phenos = sorted(predictions[exp_id].keys())
        title = '\t'.join([exp_id] + phenos)
        fp.write(title + '\n')
        for exp_id in predictions:
            temp_str = '\t'.join([exp_id] + ['%i' % predictions[exp_id][pheno] for pheno in phenos])
            fp.write(temp_str + '\n')
        fp.close()
        return
    
def mitocheck_analysis(resD=None):
    phenos = ['max_Prometaphase', 'max_Metaphase', 'max_MetaphaseAlignment']
    
    if resD is None:
        filename = '/Users/twalter/data/mitocheck_results_primary_screen/meta_2007_11_06/id_result_file_essential.pickle'
        fp = open(filename, 'r')
        resD = pickle.load(fp)
        fp.close()
    
    gene_list = ['ATM','BARD1','BRCA1','BRCA2','BRIP1','CASP8','CDH1','CHEK2',
                 'CTLA4','CYP19A1','FGFR2','H19','LSP1','MAP3K1','MRE11A','NBN',
                 'PALB2','PTEN','RAD51','RAD51C','STK11','TERT','TOX3','TP53','XRCC2','XRCC3']
    for gene in gene_list:
        print
        print ' **************************** '
        print gene
        id_list = filter(lambda x: resD[x]['geneName'].lower()==gene.lower(), resD.keys())
        for exp_id in id_list:
            tempStr = '%s\t%s\t%s: ' % (gene, resD[exp_id]['sirnaId'], exp_id)
            for pheno in phenos:
                tempStr += '   %s: %.4f' % (pheno, resD[exp_id][pheno])
            print tempStr    
            
    return

def mitocheck_analysis2(resD, filename_hit_table):
    phenos = ['max_Metaphase', 'max_Prometaphase', 'max_Apoptosis', 
              'max_MetaphaseAlignment']
    sirnas = {}
    for exp_id in resD.keys():
        sirna = resD[exp_id]['sirnaId']
        if not sirna in sirnas:
            sirnas[sirna] = {'gene': resD[exp_id]['geneName'], 
                             'idL': []}
        sirnas[sirna]['idL'].append(exp_id)
    
    scores = {}
    for sirna in sirnas:
        scores[sirna] = {'gene': sirnas[sirna]['gene']}
        for pheno in phenos:
            scores[sirna][pheno] = np.median([resD[x][pheno] for x in sirnas[sirna]['idL']])

    # hit lists:
    for pheno in phenos:
        print
        print
        print ' ******************************************************** '
        print pheno
        score_list = [(sirna, scores[sirna]['gene'], scores[sirna][pheno]) for sirna in scores.keys()]
        score_list.sort(key=operator.itemgetter(-1), reverse=True)
        for i in range(10): 
            sirna = score_list[i][0]
            print '%s\t%s\t%.5f' % (scores[sirna]['gene'], sirna, scores[sirna][pheno] )            
            
    threshD = {'max_Prometaphase': 0.06,
               'max_Metaphase': 0.03, 
               'max_MetaphaseAlignment': 0.06}

    hit_table = {}
    for pheno in threshD.keys():
        sirnas = filter(lambda x: scores[x][pheno] > threshD[pheno], scores.keys())
        for sirna in sirnas:
            if sirna in hit_table:
                continue
            hit_table[sirna] = {
                                'gene': scores[sirna]['gene']                                
                                }
            for pheno in phenos:
                hit_table[sirna][pheno] = scores[sirna][pheno]

    hit_table2 = {}
    for sirna in hit_table: 
        gene = hit_table[sirna]['gene']
        if not gene in hit_table2:
            hit_table2[gene] = {}
            for pheno in phenos:
                hit_table2[gene][pheno] = hit_table[sirna][pheno]
        else:
            for pheno in phenos:
                hit_table2[gene][pheno] = max(hit_table2[gene][pheno], 
                                              hit_table[sirna][pheno])
                
    # export hit_table:
    fp = open(filename_hit_table, 'w')
    tempStr = '\t'.join(['gene'] + phenos)
    fp.write(tempStr + '\n')
    for gene in hit_table2:
        tempStr = '\t'.join([gene] + 
                            ['%.5f' % hit_table2[gene][pheno] for pheno in phenos])
        fp.write(tempStr + '\n')
    fp.close()

    return

    
if __name__ ==  "__main__":

    description =\
'''
%prog - running segmentation tool .
'''

    parser = OptionParser(usage="usage: %prog [options]",
                         description=description)

    parser.add_option("-i", "--input_folder", dest="input_folder",
                      help="Input folder (raw data)")
    parser.add_option("-o", "--output_folder", dest="output_folder",
                      help="Output folder (properly adjusted images)")


    (options, args) = parser.parse_args()
    si = SimpleAnalyzer(options.input_folder, options.output_folder)
    predictions = si.get_counts()
    filename = options.input_folder.split('/')[-1] + '.txt'
    si.export_predictions(predictions, filename)
    
    