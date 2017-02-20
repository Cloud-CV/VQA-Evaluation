import pika
import sys

# Add path to annotation File
annFile = '../Data/VQA_jsons/train-challenge2015_train2014_anno.json'
# Add path to question File
# quesFile = '../Data/VQA_jsons/OpenEnded_mscoco_train2014_questions.json'
# Load path to resFile as argument
resFile = '../Data/VQA_jsons/train-challenge2015_train2014_results.json'

vqa_dict = {}
vqa_dict['anno'] = annFile
# vqa_dict['ques'] = quesFile
vqa_dict['pred'] = resFile
vqa_dict['phase_codename'] = 'train-challenge2015'

vqa_str = str(vqa_dict)

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='hello')
channel.basic_publish(exchange='',routing_key='hello',body=vqa_str)
connection.close()