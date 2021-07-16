from __future__ import unicode_literals, print_function
from flask import Flask, render_template, request, redirect, url_for
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
nlpg = spacy.load('en_core_web_sm')
from logging.config import dictConfig
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

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
n_iter=3

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

@app.route('/upload',methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)

    app.logger.info('%s logged in successfully', "HELLO")
    print("HELLO")
    return json.dumps(["OK"])

@app.route('/parse')
def parse_resume():
    from pyresparser import ResumeParser
    data = ResumeParser('/home/rhysha/applicant-scoring/resume.pdf').get_extracted_data()
    #print(data['skills'])
    skills = data['skills']

    required_skills = request.args.get('required_skills').upper()
    doc = nlp(json.dumps(' '.join([str(elem) for elem in skills])).upper())
    #print(doc)
    rv = []
    #print('Entities', [rv.append((ent.text, ent.label_)) for ent in doc.ents])
    a=required_skills
    b= json.dumps(' '.join([str(elem) for elem in skills])).upper()
    print(a,b)
    score = ner_similarity(a,b)
    print(score)
    doc = nlp(b)
    rv.append([{"score":score}])
    for ent in doc.ents:
        text,label = ent.text, ent.label_
        rv.append({text:label})
   
    return json.dumps(rv)
    
def ner_similarity(a,b):
    nlpg = spacy.load('en_core_web_sm')
    search_doc = nlpg(a)
    main_doc = nlpg(b)

    #search_doc = nlpg(a)
    #main_doc = nlpg(b)

    search_doc_no_stop_words = nlpg(' '.join([str(t) for t in search_doc if not t.is_stop]))
    main_doc_no_stop_words = nlpg(' '.join([str(t) for t in main_doc if not t.is_stop]))
    print(search_doc_no_stop_words.similarity(main_doc_no_stop_words))
    return json.dumps([search_doc_no_stop_words.similarity(main_doc_no_stop_words)])    

@app.route('/similarity')
def similarity():
    a = request.args.get('a').upper()
    b = request.args.get('b').upper()
    nlpg = spacy.load('en_core_web_sm')
    search_doc = nlpg(a)
    main_doc = nlpg(b)

    #search_doc = nlpg(a)
    #main_doc = nlpg(b)

    search_doc_no_stop_words = nlpg(' '.join([str(t) for t in search_doc if not t.is_stop]))
    main_doc_no_stop_words = nlpg(' '.join([str(t) for t in main_doc if not t.is_stop]))
    print(search_doc_no_stop_words.similarity(main_doc_no_stop_words))
    return json.dumps([search_doc_no_stop_words.similarity(main_doc_no_stop_words)])


    