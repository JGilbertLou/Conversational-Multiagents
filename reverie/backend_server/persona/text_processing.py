"""
Author: José Loucel

File: text_processing.py
Description: Defines all the functions to preprocess, embedd and compare text.


Note (August 7, 2024) -- Preprocessing
"""
import re
import datetime
import sys
import torch

sys.path.append('../../')

from global_methods import *
from persona.prompt_template.gpt_structure import *
from persona.prompt_template.print_prompt import *
from persona.prompt_template.run_gpt_prompt import *

from transformers import BertTokenizer, BertModel, BartTokenizer, BartModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rake_nltk import Rake

rake = Rake()

sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased')

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
bart_model = BartModel.from_pretrained('facebook/bart-large') 

classifier = pipeline('sentiment-analysis', model="nateraw/bert-base-uncased-emotion")


sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_phrase_keywords(phrase):
    rake.extract_keywords_from_text(phrase)
    return rake.get_ranked_phrases()

def generate_sentence_transformer_embedding(phrase):
    return sentence_transformer_model.encode([phrase])[0]

def _calculate_embedding(phrase, tokenizer, model):
    inputs = tokenizer(phrase, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    sentence_embedding = torch.mean(last_hidden_states, dim=1)
    return sentence_embedding

def generate_bert_text_embedding(phrase):
    return _calculate_embedding(phrase, bert_tokenizer, bert_model)

def generate_bart_text_embedding(phrase):
    return _calculate_embedding(phrase, bart_tokenizer, bart_model)

def cosine_sim_sentence(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0]

def matrix_cosine_sim(embedding_vector1, embedding_vector2):
    return cosine_similarity(embedding_vector1, embedding_vector2)

def identify_question(phrase):
  return run_gpt_identify_question(phrase)

def classify_phrase(curr_time, curr_chat):
  return run_gpt_classify_phrase(curr_time, curr_chat)

def identify_emotion(message):
    emotion_results = classifier(message)
    return emotion_results[0]["label"], emotion_results[0]["score"]
    