"""
Write tests for vqa evaluation code
"""
import os 
import time
import numpy as np
import vqaeval_par as eval_fun
import sys 
import unittest

annFile = '../Data/VQA_jsons/mscoco_train2014_annotations.json'
quesFile = '../Data/VQA_jsons/OpenEnded_mscoco_train2014_questions.json'
resFile = '../Data/VQA_jsons/OpenEnded_mscoco_train2014_fake_results.json'


class EvalTest(unittest.TestCase):

	def test_chunk_size(self):
		"""
		Check chunk size
		"""
		global vqa
		global vqaEval
		global CHUNK_SZ
		CHUNK_SZ, vqa, vqaEval = eval_fun.prepare_objects(annFile, quesFile, resFile, 16)
		self.assertEqual(CHUNK_SZ, 16)

	def test_iter_array(self):
		"""
		Test to check the w-avg coefficients of the chunk sizes
		"""
		binary_qids = vqa.getQuesIds(ansTypes='yes/no')
		number_qids = vqa.getQuesIds(ansTypes='number')	
		other_qids = vqa.getQuesIds(ansTypes='other')
		all_qids = vqa.getQuesIds()

		self.assertEqual(len(binary_qids) + len(number_qids) + len(other_qids) - len(all_qids), 0)

		binary_qids_split = np.array_split(binary_qids, CHUNK_SZ)
		number_qids_split = np.array_split(number_qids, CHUNK_SZ)
		other_qids_split = np.array_split(other_qids, CHUNK_SZ)

		binary_qids_num = []
		for i in range(CHUNK_SZ):
			binary_qids_num.append(len(binary_qids_split[i]))

		number_qids_num = []
		for i in range(CHUNK_SZ):
			number_qids_num.append(len(number_qids_split[i]))

		other_qids_num = []
		for i in range(CHUNK_SZ):
			other_qids_num.append(len(other_qids_split[i]))

		binary_qids_len = eval_fun.get_iter_arr(binary_qids_split)
		number_qids_len = eval_fun.get_iter_arr(number_qids_split)
		other_qids_len = eval_fun.get_iter_arr(other_qids_split)

		self.assertEqual(binary_qids_len, binary_qids_num)
		self.assertEqual(number_qids_len, number_qids_num)
		self.assertEqual(other_qids_len, other_qids_num)


	def test_ques_accuracy(self):
		"""
		Check for accuracies using parallel-vs-sequential computation 	
		"""
		vqaEval.evaluate()
		# Regular accuracies
		reg_ovacc = vqaEval.accuracy['overall']
		reg_binacc = vqaEval.accuracy['perAnswerType']['yes/no']
		reg_numacc = vqaEval.accuracy['perAnswerType']['number']
		reg_otheracc = vqaEval.accuracy['perAnswerType']['other']

		# Parallel-processed accuracies
		ovacc, binacc, numacc, otheracc = eval_fun.Evaluate(annFile, quesFile, resFile, CHUNK_SZ)

		self.assertEqual(round(reg_ovacc, 2), round(ovacc, 2))
		self.assertEqual(round(reg_binacc, 2), round(binacc, 2))
		self.assertEqual(round(reg_numacc, 2), round(numacc, 2))
		self.assertEqual(round(reg_otheracc, 2), round(otheracc, 2))
