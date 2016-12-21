"""
TODO: Try to remove the sub-process dependence
Input: GT json files, Pred json files

"""
# coding: utf-8
import multiprocessing
import sys
dataDir = '../../VQA-Evaluation'   #Change this according to the repository name
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import os
import time
import numpy as np

annFile = sys.argv[1]
quesFile = sys.argv[2]
resFile = sys.argv[3]

## number of chunks for splitting the qid_list
CHUNK_SZ = 16
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)
vqaEval = VQAEval(vqa, vqaRes, n=2)
all_qids = vqa.getQuesIds()
binary_qids = vqa.getQuesIds(ansTypes='yes/no')
number_qids = vqa.getQuesIds(ansTypes='number')	
other_qids = vqa.getQuesIds(ansTypes='other')

"""
Slightly more optimized implementation of splitting stuff
Saves ~2 seconds
Flipped the process of computing question-type accuracies. Good Stuff, the chunking idea!
"""
def get_iter_arr(length_qids):
	factor = int(length_qids/CHUNK_SZ)
	remainder = length_qids % CHUNK_SZ
	len_array = np.ones(CHUNK_SZ)
	len_array = factor*len_array
	if remainder != 0:
		len_array[-1] = remainder
	return len_array.tolist()

def vqaeval(qid_list):
	vqaEval.evaluate(qid_list.tolist())
	return vqaEval.accuracy['overall']

def reduce_acc(results_list, length_list, length):
	return float(sum([a*b for a,b in zip(results_list, length_list)])) / length
"""
End 
"""

if __name__ == "__main__":
	t = time.time()
	pool = multiprocessing.Pool(12)
	
	## Binary Accuracies
	binary_qids_split = np.array_split(binary_qids, CHUNK_SZ)
	binary_qids_len = get_iter_arr(len(binary_qids))
	binary_results = pool.map(vqaeval, binary_qids_split)
	binary_acc = reduce_acc(binary_results, binary_qids_len, len(binary_qids))
	print(binary_acc)

	## Number Accuracies
	number_qids_split = np.array_split(number_qids, CHUNK_SZ)
	number_qids_len = get_iter_arr(len(number_qids))
	number_results = pool.map(vqaeval, number_qids_split)
	number_acc = reduce_acc(number_results, number_qids_len, len(number_qids))
	print(number_acc)

	## Other Accuracies
	other_qids_split = np.array_split(other_qids, CHUNK_SZ)
	other_qids_len = get_iter_arr(len(other_qids))
	other_results = pool.map(vqaeval, other_qids_split)
	other_acc = reduce_acc(other_results, other_qids_len, len(other_qids))
	print(other_acc)

	## Overall Accuracy
	overall_acc = float(float(other_acc*len(other_qids)) + float(number_acc*len(number_qids)) + float(binary_acc*len(binary_qids))) / len(all_qids)
	print(overall_acc)

	elapsed = time.time() - t
	print "Elapsed Time: " + str(elapsed)
	
	


