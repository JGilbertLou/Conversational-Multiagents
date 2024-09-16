"""
Author: José Loucel

File: director.py
Description: This defines the "director" fsm, responsible of the coordination between conversational agents. 
"""

import sys
sys.path.append('../../')

import uuid

from persona.text_processing import *
from persona.cognitive_modules.convo.fsm import *

class Director:
    def __init__(self, theme):
        self.director = FSM()
        self.director_id = uuid.uuid4()

        self.theme = theme
        self.convo = []
        self.last_message = ""

        self.n_agents = 0
        self.agents = dict()
        self.agents_to_join = []
        self.new_joiners = []
        self.agent_to_leave = ""
        self.agent_ack = []
        self.possible_speakers = []

        self.current_speaker = ""
        self.future_speaker = ""
        self.max_round_wo_talking = 1
        self.max_cos_sim_index = 0
                
        self.director.set_state(self._decide_process)

    def register_agent(self, agent):
        agent.join_conversation(self.theme, self.director_id)
        self.agents_to_join.append(agent)

    def _eliminate_agent(self, agent_name):
        for agent_name in self.possible_speakers:
            self.agents[agent_name]["reference"].delete_relationship(self.director_id, self.current_speaker)
        self.agents[agent_name].pop()
        self.n_agents -= 1

        # Puede que lo tenga que quitar
        '''self.agents[agent_name] = dict()
        self.agents[agent_name]["reference"] = agents[agent_name]
        self.agents[agent_name]["round_wo_talking"] = 1'''

    def _build_context_for_prompt(self):
        agents_name_str = ""
        new_joiners_str = ""

        for agent_name in self.possible_speakers:
            if agent_name in self.new_joiners:
                new_joiners_str += agent_name + ", "
            else:
                agents_name_str += agent_name + ", "

        if self.last_message == "":
            i_p_status = "starting a"
            f_p_status = "start the conversation"
        elif self.agents[self.current_speaker]["leaving"]:
            i_p_status = "leaving the"
            f_p_status = "farewell"
        else:
            i_p_status = "having a"
            f_p_status = "continue the conversation"

        initial_prompt = self.current_speaker + " is " +  i_p_status + " conversation with " + agents_name_str
        
        if self.theme != "":
            initial_prompt += " about " + self.theme

        initial_prompt += "."

        if new_joiners_str != "":
            initial_prompt += new_joiners_str + " are joining the conversation."
        
        final_prompt = "what should " + self.current_speaker + " say next to " + f_p_status + "?"

        return initial_prompt, final_prompt
    

    #S0
    def _decide_process(self):
        if len(self.agents_to_join) > 0:  
            print("Orchestrator - S0: Add New Agent to Conversation -> S1")
            self.director.set_state(self._request_to_look_for_relationship)
        elif self.agents[agent_name]["leaving"] == True:
            print("Orchestrator - S0: Add New Agent to Conversation -> S1")
            self.director.set_state(self._request_to_look_for_relationship)
        else:
            if self.n_agents > 1:
                if len(self.possible_speakers) == (self.new_joiners):
                    self.new_joiners = []
                print("Orchestrator - S0: Select Next Speaker -> S3")
                # Missing set agents permission to retrieve message
                # Manipular possible speakers en relationship, ya que si se unen más personas a la conversación, estaría bien que ellas hablaran primero
                self.agent_ack = self.possible_speakers.copy()
                for agent_name in self.possible_speakers:
                    self.agents[agent_name]["reference"].set_retrieve_relevant_context(self.director_id)
                self.director.set_state(self._agent_selection)

    #S1
    def _request_to_look_for_relationship(self):
        new_agent = self.agents_to_join.pop()
        self.new_joiners.append(new_agent.name)
        self.possible_speakers.append(new_agent.name)
        print("Orchestrator - S1: Request Agents to Find Their Relationship with ", new_agent.name, " -> S2")
        convo_agents_names = [agent_name for agent_name in self.agents.keys()]
        self.agents[new_agent.name] = dict()
        self.agents[new_agent.name]["reference"] = new_agent
        self.agents[new_agent.name]["round_wo_talking"] = 1
        self.agents[new_agent.name]["leaving"] = False

        for agent_name in convo_agents_names:
            self.agents[agent_name]["reference"].set_relationship_with([new_agent.name], self.director_id)
            self.agent_ack.append(agent_name)

        self.agents[new_agent.name]["reference"].set_relationship_with(convo_agents_names, self.director_id)
        self.agent_ack.append(new_agent.name)
        self.director.set_state(self._ack_relationship)

    #S2
    def _ack_relationship(self):
        for agent_name in self.agent_ack.copy():
            if self.agents[agent_name]["reference"].get_relationship_ack(self.director_id):
                self.agent_ack.remove(agent_name)

        if len(self.agent_ack) == 0:
            print("People in the conversation: ", self.agents.keys())
            print("\n")
            if len(self.agents_to_join) == 0:
                print("Orchestrator - S2: Decide Next Action -> S0")
                self.director.set_state(self._decide_process)
            else:
                print("Orchestrator - S2: Add New Agent to Conversation -> S1")
                self.director.set_state(self._request_to_look_for_relationship)
        else:
            print("Orchestrator - S2: Wait for Agents to Identify Their Relationship -> S2")

    #s3
    def _agent_selection(self):
        for agent_name in self.agent_ack.copy():
            ack, end, similarity_index = self.agents[agent_name]["reference"].get_context_sim(self.director_id)
            if ack:
                if self.agent_to_leave == "":
                    participation_index = self.agents[agent_name]["round_wo_talking"] / self.max_round_wo_talking
                    speak_permission_index = 0.3*participation_index + 0.7*similarity_index
                    if end:
                        self.future_speaker = agent_name
                        self.agents[agent_name]["leaving"] = True
                    elif self.max_cos_sim_index < speak_permission_index:
                        self.max_cos_sim_index = speak_permission_index
                        self.future_speaker = agent_name
        
        if len(self.agent_ack) == 0:
            self.possible_speakers.remove(self.future_speaker)
            rounds_wo_talk = 0
            for agent_name in self.possible_speakers:
                self.agents[agent_name]["reference"].set_permission_2_talk(False, "", "")
                self.agents[agent_name]["round_wo_talking"] += 1
                if self.agents[agent_name]["round_wo_talking"] > rounds_wo_talk:
                    rounds_wo_talk = self.agents[agent_name]["round_wo_talking"]

            if (self.current_speaker != ""):
                self.agents[self.current_speaker]["round_wo_talking"] = 1        
                self.possible_speakers.append(self.current_speaker)
             
            self.current_speaker = self.future_speaker
            self.max_round_wo_talking = rounds_wo_talk

            initial_prompt, question_task = self._build_context_for_prompt()
            self.agents[self.current_speaker]["reference"].set_permission_2_talk(True, initial_prompt, question_task)
            self.agents[self.current_speaker]["round_wo_talking"] = 0

            self.new_joiners = []

            print("Orchestrator - S3: Sending Agents Permission Resolution -> S4")
            self.director.set_state(self._new_message)
        else:
            print("Orchestrator - S3: Waiting Context From All Agents - > S3")

    #S4
    def _new_message(self): 
        last_message = self.agents[self.current_speaker]["reference"].get_last_message(self.director_id)
        if self.last_message != last_message:
            self.convo.append(last_message)
            self.last_message = last_message

            for agent_name in self.possible_speakers:
                self.agents[agent_name]["reference"].set_new_message(last_message, self.director_id)
            self.future_speaker = ""
            self.max_cos_sim_index = 0

            if self.agents[self.current_speaker]["leaving"]:
                self._eliminate_agent(self.current_speaker)
                self.current_speaker = ""
                
            print("Orchestrator - S4: New Message Received - Decide Next Action -> S0")
            self.director.set_state(self._decide_process)
        else:
            print("Orchestrator - S4: Waiting New Message -> S4")
         
    def execute_fsm(self):
        self.director.update()