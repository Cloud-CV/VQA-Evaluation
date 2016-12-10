"""
TODO: Evaluate VQA in parallel for n-instances
Input: GT json files, Pred json files

"""
# coding: utf-8
import multiprocessing
import sys
dataDir = '../../VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import os
import time

annFile = sys.argv[1]
quesFile = sys.argv[2]
resFile = sys.argv[3]
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)
vqaEval = VQAEval(vqa, vqaRes, n=2)
all_qids = vqa.getQuesIds()



def vqaeval(iter):
	qid = all_qids[iter]
	qid_list = []
	qid_list.append(qid)
	vqaEval.evaluate(qid_list)
	qid_acc_dict = {qid:vqaEval.accuracy['overall']}
	return qid_acc_dict

def reduce_acc(results_list):
	result_dict = reduce(lambda r, d: r.update(d) or r, results_list, {})
	# Get question ids corresponding to 3 answer types - yes/no; Number; Others
	binary_qids = vqa.getQuesIds(ansTypes='yes/no')
	number_qids = vqa.getQuesIds(ansTypes='number')
	other_qids = vqa.getQuesIds(ansTypes='other')
	
	overall_acc = float(sum(result_dict[key] for key in all_qids)) / len(all_qids)
	binary_acc = float(sum(result_dict[key] for key in binary_qids)) / len(binary_qids)
	number_acc = float(sum(result_dict[key] for key in number_qids)) / len(number_qids)
	other_acc = float(sum(result_dict[key] for key in other_qids)) / len(other_qids)
	
	print(overall_acc)
	print(binary_acc)
	print(number_acc)
	print(other_acc)
	
if __name__ == "__main__":
	t = time.time()
	pool = multiprocessing.Pool(4)
	results = pool.map(vqaeval, range(len(all_qids)))
	reduce_acc(results)
	elapsed = time.time() - t
	print elapsed
	
	


