
"""
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

def prepare_objects(annFile, quesFile, resFile, chunk):
	global CHUNK_SZ 
	global vqa 
	global vqaRes
	global vqaEval
	CHUNK_SZ = chunk
	vqa = VQA(annFile, quesFile)
	vqaRes = vqa.loadRes(resFile, quesFile)
	vqaEval = VQAEval(vqa, vqaRes, n=2)
	return CHUNK_SZ, vqa, vqaEval
	
"""
Slightly more optimized implementation of splitting stuff
Saves ~2 seconds
Flipped the process of computing question-type accuracies. Good Stuff, the chunking idea!
"""
def get_iter_arr(qid_split):
	len_array = []
	for i in range(CHUNK_SZ):
		len_array.append(len(qid_split[i]))
	return len_array

def vqaeval(qid_list):
	vqaEval.evaluate(qid_list.tolist())
	return vqaEval.accuracy['overall']

def reduce_acc(results_list, length_list, length):
	return float(sum([a*b for a,b in zip(results_list, length_list)])) / length

"""
End 
"""

def Evaluate(annFile, resFile):
	quesFile = dataDir + '/Data/VQA_jsons/OpenEnded_mscoco_train2014_questions_reduced.json'
	chunk_sz = 16
	N_CORES = 2
	prepare_objects(annFile, quesFile, resFile, chunk_sz)
	all_qids = vqa.getQuesIds()
	binary_qids = vqa.getQuesIds(ansTypes='yes/no')
	number_qids = vqa.getQuesIds(ansTypes='number')	
	other_qids = vqa.getQuesIds(ansTypes='other')

	t = time.time()

	pool = multiprocessing.Pool(N_CORES)
	
	## Binary Accuracies
	binary_qids_split = np.array_split(binary_qids, CHUNK_SZ)
	binary_qids_len = get_iter_arr(binary_qids_split)
	binary_results = pool.map(vqaeval, binary_qids_split)
	binary_acc = reduce_acc(binary_results, binary_qids_len, len(binary_qids))
	print(binary_acc)

	## Number Accuracies
	number_qids_split = np.array_split(number_qids, CHUNK_SZ)
	number_qids_len = get_iter_arr(number_qids_split)
	number_results = pool.map(vqaeval, number_qids_split)
	number_acc = reduce_acc(number_results, number_qids_len, len(number_qids))
	print(number_acc)

	## Other Accuracies
	other_qids_split = np.array_split(other_qids, CHUNK_SZ)
	other_qids_len = get_iter_arr(other_qids_split)
	other_results = pool.map(vqaeval, other_qids_split)
	other_acc = reduce_acc(other_results, other_qids_len, len(other_qids))
	print(other_acc)

	## Overall Accuracy
	overall_acc = float(float(other_acc*len(other_qids)) + float(number_acc*len(number_qids)) + float(binary_acc*len(binary_qids))) / len(all_qids)
	print(overall_acc)


	elapsed = time.time() - t
	print "Elapsed Time: " + str(elapsed)
	return overall_acc, binary_acc, number_acc, other_acc	
