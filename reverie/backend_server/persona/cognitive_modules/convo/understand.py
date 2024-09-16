"""
Author: José Loucel

File: understand.py
Description: This defines the "understand" fsm for the conversational agents. 
"""

import random
import sys
sys.path.append('../../')

from persona.text_processing import *
from persona.cognitive_modules.convo.fsm import *

class Understand:
    def __init__(self, agent, buffer): 
        self.agent = agent
        self.understand = FSM()
        self.ack = dict()
        self.cos_sim_index = dict()
        self.buffer = buffer
        self.director_response = False
        self.permission_2_talk = False
        self.relationship_flag = 0
        self.relationship_petition = []
        self.retrieve_petition = []
        self.current_retrieve_i = -1
        self.convo_context_flag = 0
        self.agent_stack = dict()
        self.initial_phrase = ""
        self.question_task = ""
        self.understand.set_state(self._choose_action)


    def set_relationship_with(self, orchestrator_id, t_agents_name):
        self.relationship_flag += 1
        self.relationship_petition.append(orchestrator_id)
        self.agent_stack[orchestrator_id] = t_agents_name
        self.ack[orchestrator_id] = False

    def get_relationship_ack(self, orchestrator_id):
        if self.ack[orchestrator_id]:
            self.ack[orchestrator_id] = False
            return True
        return False
    
    def set_retrieve_context(self, orchestrator_id):
        self.convo_context_flag += 1 
        self.retrieve_petition.append(orchestrator_id)

    def get_cos_sim(self, orchestrator_id):
        if self.ack[orchestrator_id]:
            self.ack[orchestrator_id] = False
            return True, self.cos_sim_index[orchestrator_id]
        return False, 0
    
    def set_director_response(self, permission, initial_phrase, question_task):
        self.director_response = True
        self.permission_2_talk = permission
        self.initial_phrase = initial_phrase
        self.question_task = question_task
    
    # S0
    def _choose_action(self):
        if self.relationship_flag > 0:
            if self.relationship_petition[0] in self.buffer.keys():
                self.relationship_flag -= 1
                print("Understand - S0: Setting Relationship -> S1")
                self.understand.set_state(self._look_agent_relationship)
        elif self.convo_context_flag > 0:
            for retrieve_id in self.retrieve_petition:
                if retrieve_id in self.buffer.keys():
                    if self.buffer[retrieve_id]["retrieve_flag"]:
                        self.current_retrieve = retrieve_id
                        self.convo_context_flag -= 1
                        self.buffer[retrieve_id]["retrieve_flag"] = False
                        print("Understand - S0: Retrieve Relevant Data for Convo -> S2")
                        self.understand.set_state(self._get_context) 
                        break    
        else: 
            print("Understand - S0: Waiting to be Required -> S0")

    # S1
    def _look_agent_relationship(self):
        for t_agent_name in self.agent_stack[self.relationship_petition[0]]:
            t_a_retrieve = self.agent.ag_retrieve(t_agent_name)
            self.buffer[self.relationship_petition[0]]["participants"][t_agent_name] = self.agent.generate_agent_relationship(t_agent_name, t_a_retrieve)
        self.agent_stack.pop(self.relationship_petition[0])
        self.ack[self.relationship_petition[0]] = True
        self.relationship_petition.remove(self.relationship_petition[0])
        print("Understand - S1: Relationship Set -> S0")
        self.understand.set_state(self._choose_action)

    # S2
    def _get_context(self):
        retrieval_keys = self.buffer[self.current_retrieve]["retrieval_keys"]
        if retrieval_keys["personal"] == "" or retrieval_keys["personal"] in self.agent.name:
            time_interval = retrieval_keys["time"] 
            first_date = retrieval_keys["initial_date"]
            last_date = retrieval_keys["final_date"]
            initial_hour = retrieval_keys["initial_hour"]
            last_hour = retrieval_keys["final_hour"] 

            if retrieval_keys["themed"]:
                embeded_convo = self.buffer[self.current_retrieve]["retrieval_keys"]["convo_embeddings"][-4:]
                self.buffer[self.current_retrieve]["prompt_context"]["retrieved_nodes"], self.cos_sim_index[self.current_retrieve] = self.agent.sim_retrieve(embeded_convo, time_interval, first_date, last_date, initial_hour, last_hour)
            else:
                self.buffer[self.current_retrieve]["prompt_context"]["retrieved_nodes"] = self.agent.priority_retrieve(time_interval, first_date, last_date, initial_hour, last_hour)
                self.cos_sim_index[self.current_retrieve] = 1
        else:
            self.cos_sim_index[self.current_retrieve] = 0

        self.ack[self.current_retrieve] = True
        print("Understand - S2: Getting Relevant Data From Memory -> S3")
        self.understand.set_state(self._wait_director_response)
         
    # S3
    def _wait_director_response(self):
        if self.director_response:
            self.director_response = False
            if self.permission_2_talk:
                self.buffer[self.current_retrieve]["speak_flag"] = True
                self.buffer[self.current_retrieve]["prompt_context"]["initial_phrase"] = self.initial_phrase
                self.buffer[self.current_retrieve]["prompt_context"]["question_task"] = self.question_task
                print("Understand - S3: Permission to Talk. Sending Flag to Talk Module -> S4")
                self.understand.set_state(self._ack_permission_talk)
            else:
                print("Understand - S3: No Permission to Talk -> S0") 
                self.buffer[self.current_retrieve]["active"] = False     
                self.retrieve_petition.remove(self.current_retrieve)
                self.understand.set_state(self._choose_action)
        else:
            print("Understand - S3: Waiting Director Response -> S3")
        
    # S4
    def _ack_permission_talk(self):
        if not self.buffer[self.current_retrieve]["speak_flag"]:
            self.retrieve_petition.remove(self.current_retrieve)
            print("Understand - S4: Awcknowledge Received -> S0")
            self.understand.set_state(self._choose_action)
        else:
            print("Understand - S4: Waiting for Talk Awcknowledge -> S4")

    def update(self):
        self.understand.update()