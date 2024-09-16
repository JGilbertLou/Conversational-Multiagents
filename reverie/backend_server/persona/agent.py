"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: persona.py
Description: Defines the Persona class that powers the agents in Reverie. 

Note (May 1, 2023) -- this is effectively GenerativeAgent class. Persona was
the term we used internally back in 2022, taking from our Social Simulacra 
paper.
"""
import math
import sys
import datetime
import random
sys.path.append('../')

from global_methods import *

from persona.memory_structures.spatial_memory import *
from persona.memory_structures.concomitant_memory import *
from persona.memory_structures.scratch import *

from persona.cognitive_modules.perceive import *
from persona.cognitive_modules.retrieve import *
from persona.cognitive_modules.plan import *
from persona.cognitive_modules.reflect import *
from persona.cognitive_modules.execute import *
from persona.cognitive_modules.converse import *

from persona.text_processing import *

from persona.cognitive_modules.convo.hear import *
from persona.cognitive_modules.convo.understand import *
from persona.cognitive_modules.convo.talk import *

class Agent: 
    def __init__(self, name, folder_mem_saved=False):
        # PERSONA BASE STATE 
        # <name> is the full name of the persona. This is a unique identifier for
        # the persona within Reverie. 
        self.name = name

        # PERSONA MEMORY 
        # If there is already memory in folder_mem_saved, we load that. Otherwise,
        # we create new memory instances. 
        # <s_mem> is the persona's associative memory. 
        f_c_mem_saved = f"{folder_mem_saved}/bootstrap_memory/associative_memory"
        self.c_mem = ConcomitantMemory(f_c_mem_saved)
        # <scratch> is the persona's scratch (short term memory) space. 
        scratch_saved = f"{folder_mem_saved}/bootstrap_memory/scratch.json"
        self.scratch = Scratch(scratch_saved)

        #for stimulus, emotions in self.scratch.stimulus_coef.items():
        #    print(stimulus, emotions.items())

        
        # TODO Revisar Director()
        # TODO Crear funciones para Director()


        '''

    convo_struct = dict()
    convo_struct["theme"] = ""
    convo_struct["conversation"] = []
    convo_struct["keyword_embedding"] = dict()
    

    self.buffer = dict()
    buffer["convo"] = convo_struct
    buffer["h_u"] = False
    buffer["u_t"] = False
    buffer["response_context"] = None
    
    u_t_buffer = dict()
    u_t_buffer["understand"] = dict()
    u_t_buffer["talk"] = dict()
    u_t_buffer["talk"]
    u_t_buffer["convo"] = convo_struct
    

    
    keywords = extract_phrase_keywords(self.convo[0][-1:])
            self.keywords.append(keywords)
            for keyword in keywords:
                if keyword not in self.keywords_embeddings.values():
                    self.keywords_embeddings[keyword] = generate_bert_text_embedding(keyword)
            
            self.current_keywords = {keyword: self.keywords_embeddings[keyword] for keyword in keywords for keywords in self.keywords[-4:]}
            if self.theme not in self.current_keywords.keys():
                self.current_keywords[self.theme] = self.theme_embedding'''

        self.buffer = dict()
        

        self.hear = Hear(self, self.buffer)
        self.understand = Understand(self, self.buffer)
        self.talk = Talk(self, self.buffer)

    
    def execute_fsm(self):
        self.hear.update()
        self.understand.update()
        self.talk.update()


    def save(self, save_folder, c_nodes, c_embedding): 
        """
        Save persona's current state (i.e., memory). 

        INPUT: 
        save_folder: The folder where we wil be saving our persona's state. 
        OUTPUT: 
        None
        """
        # Associative memory contains a csv with the following rows: 
        # [event.type, event.created, event.expiration, s, p, o]
        # e.g., event,2022-10-23 00:00:00,,Isabella Rodriguez,is,idle
        f_a_mem = f"{save_folder}/associative_memory"
        self.c_mem.save(f_a_mem, c_nodes, c_embedding)

    def save_embedding(self, save_folder, c_embedding, file_name):
        f_a_mem = f"{save_folder}/associative_memory"
        self.c_mem.save_embedding(f_a_mem, c_embedding, file_name)

    def get_emotions(self):
        return self.scratch.emotions.copy()

    def update_emotions(self, stimulus, intensity):
        self.scratch.update_emotions(stimulus, intensity)
        # Scratch contains non-permanent data associated with the persona. When 
        # it is saved, it takes a json form. When we load it, we move the values
        # to Python variables. 
        #f_scratch = f"{save_folder}/scratch.json"
        #self.scratch.save(f_scratch)

    def join_conversation(self, theme, orchestrator_id):
        self.hear.set_convo(theme, orchestrator_id)

    def set_relationship_with(self, orchestrator_id, t_agents_name):
        self.understand.set_relationship_with(orchestrator_id, t_agents_name)
    
    def get_relationship_ack(self, orchestrator_id):
        return self.understand.get_relationship_ack(orchestrator_id)
    
    def generate_agent_relationship(self, target_name, retreived_nodes):
        return generate_summarize_agent_relationship_v2(self.name, target_name, retreived_nodes)

    def set_new_message(self, orchestrator_id, message):
        self.hear.set_new_message(orchestrator_id, message)

    def set_retrieve_relevant_context(self, orchestrator_id):
        self.understand.set_retrieve_context(orchestrator_id)

    def get_context_sim(self, orchestrator_id):
        return self.understand.get_cos_sim(orchestrator_id)

    def set_permission_2_talk(self, permission, initial_phrase, question_task):
        return self.understand.set_director_response(permission, initial_phrase, question_task)
    
    def get_last_message(self, orch_id):
        return self.talk.get_last_message(orch_id)
    
# TODO Implementar retrieve
    def ag_retrieve(self, target_agent):
        return ag_retrieve(self, target_agent)
    
    def priority_retrieve(self, time_interval, first_date, last_date, initial_hour, last_hour):
        return priority_retrieve(self, time_interval, first_date, last_date, initial_hour, last_hour)
    
    def sim_retrieve(self, focal_keywords, time_interval, first_date, last_date, initial_hour, last_hour):
        return sim_retrieve(self, focal_keywords, time_interval, first_date, last_date, initial_hour, last_hour)
# TODO Implementar generar texto

    def generate_phrase(self, initial_phrase, relationship, curr_convo, retrieved_nodes, quesiton_task):
        utt_dic = generate_utterance(self, initial_phrase, relationship, curr_convo, retrieved_nodes, quesiton_task)
        return utt_dic["convo"], utt_dic["end"]
    
    def generate_emotional_phrase(self, initial_phrase, relationship, curr_convo, retrieved_nodes, quesiton_task):
        utt_dic = generate_emotional_utterance(self, initial_phrase, relationship, curr_convo, retrieved_nodes, quesiton_task)
        return utt_dic["convo"], utt_dic["end"]
    
    def identify_question(phrase):
        return identify_question(phrase)
    
    def classify_phrase(self, curr_chat):
        return classify_phrase(self.scratch.curr_time, curr_chat)
    
    def identify_emotion(self, message):
        return identify_emotion(message)
    
    def estimate_emotion(self, phrase, emotions):
        return estimate_emotion(phrase, emotions)

  

  