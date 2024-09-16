"""
Author: José Loucel

File: understand.py
Description: This defines the "hear" fsm for the conversational agents. 
"""
import datetime
import sys
sys.path.append('../../')

from persona.text_processing import *
from persona.cognitive_modules.convo.fsm import *

class Hear:
    def __init__(self, agent, buffer):
        self.hear = FSM()
        self.agent = agent
        self.buffer = buffer

        self.current_orch = ""
          
        self.hear.set_state(self._choose_action)
    
    def set_new_message(self, orchestrator_id, message):
        self.buffer[orchestrator_id]["message_flag"] = True
        self.buffer[orchestrator_id]["convo"].append(message)
        self.buffer[orchestrator_id]["last_message"] = message


    def set_convo(self, theme, orchestrator_id):
        self.buffer[orchestrator_id] = dict()
        self.buffer[orchestrator_id]["active"] = True
        self.buffer[orchestrator_id]["convo"] = []
        self.buffer[orchestrator_id]["message_emotion"] = []
        self.buffer[orchestrator_id]["emotions"] = []
        self.buffer[orchestrator_id]["last_message"] = ""
        self.buffer[orchestrator_id]["participants"] = dict()
        self.buffer[orchestrator_id]["message_flag"] = False
        self.buffer[orchestrator_id]["retrieve_flag"] = True
        self.buffer[orchestrator_id]["speak_flag"] = False

        retrieval_keys = dict()
        retrieval_keys["personal"] = "" 
        retrieval_keys["convo_embeddings"] = []

        if theme != "":
            retrieval_keys["themed"] = True
            retrieval_keys["time"] = False
            retrieval_keys["initial_date"] = None
            retrieval_keys["final_date"] = None
            retrieval_keys["initial_hour"] = None
            retrieval_keys["final_hour"] = None
            retrieval_keys["convo_embeddings"].append(generate_bert_text_embedding(theme)) 
        else:
            current_time = self.agent.scratch.curr_time
            retrieval_keys["themed"] = False
            retrieval_keys["time"] = True
            retrieval_keys["initial_date"] = current_time.date()
            retrieval_keys["final_date"] = current_time.date()
            retrieval_keys["initial_hour"] = datetime.time(0, 0, 0)
            retrieval_keys["final_hour"] = current_time.time()

        prompt_context = dict()
        prompt_context["initial_phrase"] = ""
        prompt_context["retrieved_nodes"] = []
        prompt_context["question_task"] = ""
            
        self.buffer[orchestrator_id]["retrieval_keys"] = retrieval_keys
        self.buffer[orchestrator_id]["prompt_context"] = prompt_context
    
    # S0
    def _choose_action(self):
        for orchestrator_id in self.buffer.keys():
            if (self.buffer[orchestrator_id]["message_flag"] and not self.buffer[orchestrator_id]["active"]):
                self.current_orch = orchestrator_id
                self.buffer[orchestrator_id]["active"] = True
                self.buffer[orchestrator_id]["message_flag"] = False
                print("Hear - S0: Processing New Phrase from ", orchestrator_id, "-> S1")
                self.hear.set_state(self._process_text)
                return None

        print("Hear - S0: Waiting to be Required -> S0")

    # S1
    def _process_text(self): 
        print(self.buffer[self.current_orch]["convo"])
        convo_phrase = self.buffer[self.current_orch]["convo"][-1]
        agent_name, convo = convo_phrase.split(":", 1)

        #Extract phrase features
        if identify_question(convo_phrase):
            retrieval_keys = classify_phrase(self.agent.scratch.curr_time, self.buffer[self.current_orch]["convo"])
            if agent_name != retrieval_keys["personal"]:
                self.buffer[self.current_orch]["retrieval_keys"]["personal"] = retrieval_keys["personal"]
            else:
                self.buffer[self.current_orch]["retrieval_keys"]["personal"] = ""

            self.buffer[self.current_orch]["retrieval_keys"]["themed"] = retrieval_keys["themed"]
            self.buffer[self.current_orch]["retrieval_keys"]["time"] = retrieval_keys["time"]
            
            if self.buffer[self.current_orch]["retrieval_keys"]["time"]:
                if retrieval_keys["final_date"] != None:
                    if retrieval_keys["final_date"] < self.agent.scratch.curr_time.date():
                        self.buffer[self.current_orch]["retrieval_keys"]["initial_date"] = retrieval_keys["initial_date"]
                        self.buffer[self.current_orch]["retrieval_keys"]["final_date"] = retrieval_keys["final_date"]
                        if retrieval_keys["final_hour"] != None:
                            self.buffer[self.current_orch]["retrieval_keys"]["initial_hour"] = retrieval_keys["initial_hour"]
                            self.buffer[self.current_orch]["retrieval_keys"]["final_hour"] = retrieval_keys["final_hour"]
                        else:
                            self.buffer[self.current_orch]["retrieval_keys"]["initial_hour"] = time(hour=0, minute=0, second=0)
                            self.buffer[self.current_orch]["retrieval_keys"]["final_hour"] = time(hour=23, minute=59, second=59)
                    elif retrieval_keys["final_date"] == self.agent.scratch.curr_time.date():
                        if retrieval_keys["final_hour"] != None:
                            if retrieval_keys["final_hour"] <= self.agent.scratch.curr_time.time():
                                self.buffer[self.current_orch]["retrieval_keys"]["initial_date"] = retrieval_keys["initial_date"]
                                self.buffer[self.current_orch]["retrieval_keys"]["final_date"] = retrieval_keys["final_date"]
                                self.buffer[self.current_orch]["retrieval_keys"]["initial_hour"] = retrieval_keys["initial_hour"]
                                self.buffer[self.current_orch]["retrieval_keys"]["final_hour"] = retrieval_keys["final_hour"]
                            else:
                                self.buffer[self.current_orch]["retrieval_keys"]["time"] = False
                                self.buffer[self.current_orch]["retrieval_keys"]["themed"] = True
                        else:
                            self.buffer[self.current_orch]["retrieval_keys"]["time"] = False
                            self.buffer[self.current_orch]["retrieval_keys"]["themed"] = True
                    else:
                        if retrieval_keys["final_hour"] != None:
                            self.buffer[self.current_orch]["retrieval_keys"]["initial_date"] = retrieval_keys["initial_date"] - datetime.timedelta(days=7)
                            self.buffer[self.current_orch]["retrieval_keys"]["final_date"] = retrieval_keys["final_date"] - datetime.timedelta(days=1)
                            self.buffer[self.current_orch]["retrieval_keys"]["initial_hour"] = retrieval_keys["initial_hour"]
                            self.buffer[self.current_orch]["retrieval_keys"]["final_hour"] = retrieval_keys["final_hour"]
                        else:
                            self.buffer[self.current_orch]["retrieval_keys"]["time"] = False
                            self.buffer[self.current_orch]["retrieval_keys"]["themed"] = True        
        else:
            self.buffer[self.current_orch]["retrieval_keys"]["themed"] = True
            self.buffer[self.current_orch]["retrieval_keys"]["time"] = False

        #Extract emotions
        stimulus, intensity = self.agent.identify_emotion(convo)
        self.buffer[self.current_orch]["message_emotion"].append([stimulus, intensity])
        self.agent.update_emotions(stimulus, intensity)
        self.buffer[self.current_orch]["emotions"].append(self.agent.get_emotions())

        self.buffer[self.current_orch]["retrieval_keys"]["convo_embeddings"].append(generate_bert_text_embedding(convo))

        print("Hear - S1: New Phrased Procesed. Sending Flag to Understand Module -> S2")
        self.buffer[self.current_orch]["retrieve_flag"] = True
        self.hear.set_state(self._wait_ack)
        
    # S2
    def _wait_ack(self):
        if not self.buffer[self.current_orch]["retrieve_flag"]:
            print("Hear - S2: Awcknowledge Received -> S0")
            self.hear.set_state(self._choose_action)
        else:
            print("Hear - S2: Waiting for Understand Awcknowledge -> S2")

    def update(self):
        self.hear.update()
