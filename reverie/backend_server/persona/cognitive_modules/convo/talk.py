"""
Author: José Loucel

File: understand.py
Description: This defines the "talk" fsm for the conversational agents. 
"""

import sys
sys.path.append('../../')

from persona.text_processing import *
from persona.cognitive_modules.convo.fsm import *

class Talk:
    def __init__(self, agent, buffer):
        self.talk = FSM()
        self.agent = agent
        self.buffer = buffer
        self.current_orch = ""
        self.new_message = ""
        self.end_convo = False
        self.talk.set_state(self._permission_to_talk)

    def get_last_message(self, orch_id):
        return self.buffer[orch_id]["last_message"]


    def _permission_to_talk(self):
        for orch_id in self.buffer.keys():
            if self.buffer[orch_id]["speak_flag"] and self.buffer[orch_id]["active"]:
                self.current_orch = orch_id
                self.talk.set_state(self._generate_sentence)
                print("Talk - S0: Permission to Talk -> S1")
                return None
                
        print("Talk - S0: Waiting for Permission to Talk -> S0")
        
    # metodo para obtener el último mensaje
    def _generate_sentence(self): 
        #convo, end = self.agent.generate_phrase(self.buffer[self.current_orch]["prompt_context"]["initial_phrase"], self.buffer[self.current_orch]["participants"], self.buffer[self.current_orch]["convo"], self.buffer[self.current_orch]["prompt_context"]["retrieved_nodes"], self.buffer[self.current_orch]["prompt_context"]["question_task"])
        convo, end = self.agent.generate_emotional_phrase(self.buffer[self.current_orch]["prompt_context"]["initial_phrase"], self.buffer[self.current_orch]["participants"], self.buffer[self.current_orch]["convo"], self.buffer[self.current_orch]["prompt_context"]["retrieved_nodes"], self.buffer[self.current_orch]["prompt_context"]["question_task"])
        self.buffer[self.current_orch]["last_message"] = convo
        self.buffer[self.current_orch]["convo"].append(convo)
        stimulus, intensity = self.agent.identify_emotion(convo)
        self.buffer[self.current_orch]["message_emotion"].append([stimulus, intensity])
        self.buffer[self.current_orch]["emotions"].append(self.agent.get_emotions())
        _, a_convo = convo.split(":", 1)
        self.buffer[self.current_orch]["retrieval_keys"]["convo_embeddings"].append(generate_bert_text_embedding(a_convo))

        self.buffer[self.current_orch]["speak_flag"] = False
        self.buffer[self.current_orch]["active"] = False
        print("Talk - S1: Generating Phrase -> S0")
        print(convo)
        self.talk.set_state(self._permission_to_talk)
        

    def update(self):
        self.talk.update()
