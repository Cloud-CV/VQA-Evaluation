from subprocess import call
import pika
import multiprocessing
import sys
dataDir = '../../VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import os
import time
import yaml

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='hello')

vqa_dict = {}
# Call back function
def callback(ch, method, properties, body):
	print(" [x] Received %r" % body)
	vqa_dict = yaml.safe_load(body)
	arg_string = 'python vqaeval_par.py ' + vqa_dict['anno'] + ' ' + vqa_dict['ques'] + ' ' + vqa_dict['pred']  
	os.system(arg_string)

channel.basic_consume(callback,queue='hello',no_ack=True)
print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()

