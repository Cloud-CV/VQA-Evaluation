import pika
import sys

# Add path to annotation File
annFile = '../Data/VQA_jsons/mscoco_train2014_annotations.json'
# Add path to question File
quesFile = '../Data/VQA_jsons/OpenEnded_mscoco_train2014_questions.json'
# Load path to resFile as argument
resFile = sys.argv[1]

vqa_dict = {}
vqa_dict['anno'] = annFile
vqa_dict['pred'] = resFile

vqa_str = str(vqa_dict)

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='hello')
channel.basic_publish(exchange='',routing_key='hello',body=vqa_str)
connection.close()