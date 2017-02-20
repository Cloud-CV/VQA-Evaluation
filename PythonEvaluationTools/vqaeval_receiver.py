from subprocess import call
import pika
import multiprocessing
import sys
import os
import time
import yaml
import vqaeval_par as eval_fun

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='hello')

vqa_dict = {}
# Call back function
def callback(ch, method, properties, body):
	print(" [x] Received %r" % body)
	vqa_dict = yaml.safe_load(body)
	result = eval_fun.evaluate(vqa_dict['anno'], vqa_dict['pred'], vqa_dict['phase_codename'])

channel.basic_consume(callback,queue='hello',no_ack=True)
print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()

