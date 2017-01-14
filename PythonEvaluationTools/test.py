"""
Write tests for vqa evaluation code
"""
import os 
import time
import numpy as np
import re
import rstr
import vqaeval_par as eval_fun
import sys 
import unittest

base_dir = os.getcwd()
base_dir = base_dir.rsplit('/',1)[0]
annFile = base_dir + '/Data/VQA_jsons/mscoco_train2014_annotations_reduced.json'
quesFile = base_dir + '/Data/VQA_jsons/OpenEnded_mscoco_train2014_questions_reduced.json'
resFile = base_dir + '/Data/VQA_jsons/OpenEnded_mscoco_train2014_fake_results_reduced.json'

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

	def test_preprocess_text(self):
		"""
		Check question preprocessing in the evaluation API
		"""
		patterns = []
		patterns.append(vqaEval.periodStrip.pattern)
		patterns.append(vqaEval.commaStrip.pattern)
		patterns.append(vqaEval.puncStrip.pattern)
		patterns.append(vqaEval.puncStrip2.pattern)

		pattern = "|".join(patterns)
		print(pattern)

		PatternString = ""
		for x in xrange(1,5):
			PatternString = PatternString + rstr.xeger(pattern)

		print(PatternString)

		stripPattern = ''
		for i in PatternString:
			if i.isdigit() or i.isalpha():
				print(i)
			else:
				stripPattern = stripPattern + i

		stripPattern = re.sub(r'\s+', '', stripPattern)
		print(stripPattern)

		# Create text for pre-processing
		pre_text = "What " +  stripPattern + " is A an THE " + stripPattern + " none ONE " + stripPattern + " image?"
		print(pre_text)
		outText_1 = vqaEval.processPunctuation(pre_text)
		outText_1 = re.sub(r'\s+', ' ', outText_1)
		print(outText_1)
		actText_1 = "What is A an THE none ONE image?"
		print(actText_1)
		self.assertEqual(outText_1, actText_1)

		outText_2 = vqaEval.processDigitArticle(outText_1)
		actText_2 = "what is 0 1 image?"
		self.assertEqual(outText_2, actText_2)

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
		ovacc, binacc, numacc, otheracc = eval_fun.Evaluate(annFile, resFile)

		self.assertEqual(round(reg_ovacc, 2), round(ovacc, 2))
		self.assertEqual(round(reg_binacc, 2), round(binacc, 2))
		self.assertEqual(round(reg_numacc, 2), round(numacc, 2))
		self.assertEqual(round(reg_otheracc, 2), round(otheracc, 2))
