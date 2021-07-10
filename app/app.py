from __future__ import unicode_literals, print_function
from flask import Flask
from flask import request
import json
import requests
app = Flask(__name__)

import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm

import pandas as pd


import csv
#########################################
TRAIN_DATA = []
with open('dataset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            TRAIN_DATA.append(
                            (row[0], {
                                'entities': [(0, len(row[0]), row[1])]
                            }))
            line_count += 1


model = "ner"
output_dir=Path("./ner")
n_iter=100

if model is not None:
    nlp = spacy.load(model)  
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')  
    print("Created blank 'en' model")

#set up the pipeline

if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
else:
    ner = nlp.get_pipe('ner')


for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in tqdm(TRAIN_DATA):
            nlp.update(
                [text],  
                [annotations],  
                drop=0.5,  
                sgd=optimizer,
                losses=losses)
        print(losses)


doc = nlp("JAVA")
print(doc)
print('Entities', [(ent.text, ent.label_) for ent in doc.ents])

if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
#########################################


nlp = spacy.load('./ner')    



@app.route('/')
def hello_world():

    return "Hello"


@app.route('/parse')
def parse_resume():
    from pyresparser import ResumeParser
    data = ResumeParser('/home/rhysha/applicant-scoring/resume.pdf').get_extracted_data()
    print(data['skills'])
    skills = data['skills']

    required_skills = request.args.get('required_still')
    doc = nlp(json.dumps(' '.join([str(elem) for elem in skills])).upper())
    print(doc)
    rv = []
    print('Entities', [rv.append((ent.text, ent.label_)) for ent in doc.ents])
    return json.dumps(rv)
   
    # title = request.args.get('title')
    # url = "http://192.168.1.111:8081/ent"
    # message_text = ' '.join([str(elem) for elem in skills])
    # headers = {'content-type': 'application/json'}
    # d = {'text': message_text, 'model': 'en'}

    # response = requests.post(url, data=json.dumps(d), headers=headers)
    # r = response.json()  
    # print(r,message_text)
    

