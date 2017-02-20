
"""
Input: GT JSON file, Pred JSON file, Phase-CodeName

############################################################################
Hardcoded path of a split-JSON file which has the following structure
Split JSON File Structure - 
{
	split1 : [list of qids]
	split2 : [list of qids]
	split3 : [list of qids]
	split4 : [list of qids]
} 

A global dict that has the information of the splits associated with each phase.
Each phase has multiple splits associated with it. 
	{
		phase-1 : split1, split2, split3
		phase-2 : split2, split4
		phase-3 : split1, split4, split3
	}

Ques-File is hardcoded
Test on the train split.

"""
# coding: utf-8
import multiprocessing
import sys
dataDir = '../../VQA-Evaluation'   #Change this according to the repository name
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
from contextlib import closing
from pprint import pprint
from tqdm import *
import os
import time
import numpy as np
import json

# phase-split association dict
"""
phase_splits = {
	'OpenEnded' : {
					'test-dev2015' : ['test-dev'],
					'test2015' : ['test-dev', 'test-reserve', 'test-challenge', 'test-standard'],
					'test-challenge2015' : ['test-dev', 'test-reserve', 'test-challenge', 'test-standard']
					}
	'MultipleChoice' : {
						'test2015' : ['test-dev', 'test-reserve', 'test-challenge', 'test-standard'],
						'test-challenge2015' : ['test-dev', 'test-reserve', 'test-challenge', 'test-standard']
						}
				}
"""

# Test on some dummy splits of the VQA-train set
phase_splits = {
	'OpenEnded' : {
					'train-dev2015' : ['split_1'],
					'train2015' : ['split_1', 'split_2', 'split_3', 'split_4'],
					'train-challenge2015' : ['split_3', 'split_2', 'split_4']
					}
				}

# Load the split-qids dict
splitFile = '../Data/VQA_jsons/vqa_train2014_dummysplits.json'
split_qids = json.load(open(splitFile))

# Hard-code question file per-challenge
quesFile = '../Data/VQA_jsons/OpenEnded_mscoco_train2014_questions.json'
questions = json.load(open(quesFile))

task_type = 'OpenEnded'
res = VQA()

# Prepare all objects, variables and make them global
def prepare_objects(annFile, resFile, phase_codename):
	print('Preparing global objects..')
	global vqa
	global binary_qids
	global number_qids
	global other_qids
	global all_qids
	global vqaRes
	global vqaEval
	vqa = VQA(annFile, questions)
	binary_qids = vqa.getQuesIds(ansTypes='yes/no')
	number_qids = vqa.getQuesIds(ansTypes='number')
	other_qids = vqa.getQuesIds(ansTypes='other')
	all_qids = vqa.getQuesIds()
	vqaEval = VQAEval(all_qids, n=2)
	vqaRes = vqa.loadRes(res, resFile)
	
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
	vqaEval.evaluate(vqa, vqaRes, qid_list.tolist())
	return vqaEval.accuracy['overall']

def reduce_acc(results_list, length_list, length):
	return float(sum([a*b for a,b in zip(results_list, length_list)])) / length

def eval_split(type_qids):
	"""
	Function to evaluate a particular split associated with a phase 
	"""
	# Type qids is a dict with keys being the answer-types and the values being the list of qids
	print('Evaluating split ..')
	accuracy_dict = {}
	acc = 0.0
	length = 0
	for key, val in type_qids.iteritems():
		if len(val) == 0:
			accuracy_dict[key] = 0.0
		else:
			qid_split = np.array_split(val, CHUNK_SZ)
			qids_len = get_iter_arr(qid_split)
			with closing(multiprocessing.Pool(N_CORES)) as p:
				key_res = p.map(vqaeval, qid_split)
			key_acc = reduce_acc(key_res, qids_len, len(val))
			accuracy_dict[key] = key_acc
			acc += float(key_acc*len(val))
			length += len(val)

	accuracy_dict['overall'] = float(acc)/float(length) 

	return accuracy_dict

def evaluate(annFile, resFile, phase_codename):
	"""
	Function to evaluate the phase submissions 
	"""
	global CHUNK_SZ
	global N_CORES
	CHUNK_SZ = 1000
	N_CORES = 16
	t = time.time()
	prepare_objects(annFile, resFile, phase_codename)

	# Get all the split-keys corresponding to a given phase
	split_keys = phase_splits[task_type][phase_codename]

	# Final accuracies as a dict with the following structure
	"""
	{
               "result":[
                  {
                     "split_codename_1":{
                        "key1":30,
                        "key2":50,
                     }
                  },
                  {
                     "split_codename_2":{
                        "key1":90,
                        "key2":10,
                     }
                  },
                  {
                     "split_codename_3":{
                        "key1":100,
                        "key2":45,
                     }
                  }
               ]
            }
	"""
	result = { 'result' : []}
	print('Evaluating phase..')
	for i in split_keys:
		type_qids = {}
		res_dict = {}
		type_qids['yes/no'] = list(set(split_qids[i]) & set(binary_qids))
		type_qids['number'] = list(set(split_qids[i]) & set(number_qids))
		type_qids['other'] = list(set(split_qids[i]) & set(other_qids))
		res_dict[i] = eval_split(type_qids)
		result['result'].append(res_dict)

	elapsed = time.time() - t
	print "Elapsed Time: " + str(elapsed)
	pprint(result)
	return result
