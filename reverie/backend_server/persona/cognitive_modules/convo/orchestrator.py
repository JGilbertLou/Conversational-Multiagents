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
from persona.cognitive_modules.convo.stack_fsm import *

class Orchestrator:
    def __init__(self, theme, user_prompt = False):
        self.orchestrator = StackFSM()
        self.orchestrator_id = uuid.uuid4()

        self.theme = theme
        self.convo = []
        self.last_message = ""

        self.user_prompt = user_prompt
        self.users = []
        self.current_user = ""
        self.active_user = False

        self.n_participants = 0
        self.agents = dict()

        self.agents_to_join = []
        self.users_to_join = []
        self.new_joiners = []
        self.new_participant_name = ""
        self.participant_reference = None

        self.agent_to_leave = ""
        self.agent_ack = []
        self.possible_speakers = []

        self.current_speaker = ""
        self.future_speaker = ""
        self.is_leaving = False
        self.max_round_wo_talking = 1
        self.max_cos_sim_index = 0
                
        self.orchestrator.push_state(self._choose_action)


    def register_agent(self, agent):
        print(agent.name)
        agent.join_conversation(self.theme, self.orchestrator_id)
        self.agents_to_join.append(agent)

    def register_user(self, user):
        print(user)
        self.users_to_join.append(user)

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
    


    def _decide_process(self):
        if len(self.agents_to_join) > 0:  
            print("Orchestrator - S0: Add New Agent to Conversation -> S1")
            self.director.set_state(self._request_to_look_for_relationship)
        elif self.agents[agent_name]["leaving"] == True:
            print("Orchestrator - S0: Add New Agent to Conversation -> S1")
            self.director.set_state(self._request_to_look_for_relationship)
        else:
            if self.user_prompt:
                sim_command = input("Enter option: ")
                if sim_command.lower() == "register": 
                    new_user = input("Enter user name: ")
                    self.users.append(new_user)
                    # TODO: add to the agents
                elif sim_command.lower() == "delete":
                    delete_user = input("Enter user name: ")
                    self.users.remove(delete_user)
                elif sim_command.lower() == "talk":
                    if len(self.users) > 0:
                        if len(self.users) > 1:
                            for i in range(len(self.users)):
                                print(i, self.users[i])
                            user_index = input("Select user: ")
                            self.current_user = self.users[int(user_index)]
                        else:
                            self.current_speaker = self.users[0]
                        phrase = input("Enter phrase: ")
                        u_message = self.current_speaker + phrase
                        self.convo.append(u_message)
                        self.user_talk = True
                        self.agent_ack = self.possible_speakers.copy()

                        for agent_name in self.possible_speakers:
                            self.agents[agent_name]["reference"].set_retrieve_relevant_context(self.director_id)


    def _register_agents_flow(self):
        self.orchestrator.push_state(self._relationship_ack)
        self.orchestrator.push_state(self._build_participant_profile)
        self.orchestrator.push_state(self._set_relationship_with_participant)
    
    def _register_user_flow(self):
        self.orchestrator.push_state(self._relationship_ack)
        self.orchestrator.push_state(self._set_relationship_with_participant)

    def _agent_convo_participation_flow(self):
        self.orchestrator.push_state(self._distribute_new_message)
        self.orchestrator.push_state(self._wait_new_agent_message)
        self.orchestrator.push_state(self._inform_permission)   
        self.orchestrator.push_state(self._inform_no_permission)
        self.orchestrator.push_state(self._retrieval_ack_and_selection)

    def _user_convo_participation_flow(self):
        self.orchestrator.push_state(self._distribute_new_message)
        self.orchestrator.push_state(self._user_speak)
        self.orchestrator.push_state(self._inform_no_permission)
        self.orchestrator.push_state(self._retrieval_ack)
    
    #S0
    def _choose_action(self):
        if len(self.agents_to_join) > 0:  
            self.participant_reference = self.agents_to_join.pop(0)
            self.new_participant_name = self.participant_reference.name
            print("Orchestrator - S0: Add New Agent to the Conversation")
            self.n_participants += 1
            self._register_agents_flow()
        else:
            if self.user_prompt:
                if len(self.users_to_join) > 0:
                    self.new_participant_name = self.users_to_join.pop(0)
                    self.users.append(self.new_participant_name)
                    print("Orchestrator - S0: Add New User to the Conversation")
                    self.n_participants += 1
                    self._register_user_flow()
                else:
                    if self.n_participants > 1:
                        sim_command = input("Enter convo action: ")
                        if sim_command in ["la", "listen agent"]:
                            print("Orchestrator - S0: Select Next Speaker")

                            if len(self.possible_speakers) == (self.new_joiners):
                                self.new_joiners = []
                            
                            for agent_name in self.possible_speakers:
                                self.agents[agent_name]["reference"].set_retrieve_relevant_context(self.orchestrator_id)
                            self.agent_ack = self.possible_speakers.copy()

                            self._agent_convo_participation_flow()

                        elif sim_command in ["aa", "answer agent"]:
                            print("Orchestrator - S0: User Talk")

                            for agent_name in self.possible_speakers:
                                self.agents[agent_name]["reference"].set_retrieve_relevant_context(self.orchestrator_id)
                            self.agent_ack = self.possible_speakers.copy()
                            for index in range(len(self.users)):
                                print(index, self.users[index])
                            user_index = input("Enter the user index: ")
                            self.future_speaker = self.users[int(user_index)]

                            self._user_convo_participation_flow()
                    else:
                        print("Orchestrator - S0: Not Enough Participants")
            else:           
                if self.n_participants > 1:
                    print("Orchestrator - S0: Select Next Speaker")
                    if len(self.possible_speakers) == (self.new_joiners):
                        self.new_joiners = []
                    
                    for agent_name in self.possible_speakers:
                        self.agents[agent_name]["reference"].set_retrieve_relevant_context(self.orchestrator_id)
                    
                    self.agent_ack = self.possible_speakers.copy()
                    self._agent_convo_participation_flow()
                else:
                        print("Orchestrator - S0: Not Enough Participants")

    #S1
    def _set_relationship_with_participant(self):
        print("Orchestrator - S1: Request Agents to Find Their Relationship with ", self.new_participant_name)
        self.new_joiners.append(self.new_participant_name)
        for agent_name in self.agents.keys():
            self.agents[agent_name]["reference"].set_relationship_with(self.orchestrator_id, [self.new_participant_name])   
        self.agent_ack = list(self.agents.keys())

        self.orchestrator.pop_state()

    #S2 
    def _build_participant_profile(self):
        print("Orchestrator - S2: Request " + self.new_participant_name + " to Find The Relationship with the Other Agents")

        self.possible_speakers.append(self.new_participant_name)
        self.participant_reference.set_relationship_with(self.orchestrator_id, list(self.agents.keys()))
        
        self.agents[self.new_participant_name] = dict()
        self.agents[self.new_participant_name]["reference"] = self.participant_reference
        self.agents[self.new_participant_name]["round_wo_talking"] = 1
        self.agents[self.new_participant_name]["leaving"] = False

        print(self.agent_ack)
        self.agent_ack.append(self.new_participant_name)

        self.orchestrator.pop_state()
    
    #S3
    def _relationship_ack(self):
        for agent_name in self.agent_ack.copy():
            if self.agents[agent_name]["reference"].get_relationship_ack(self.orchestrator_id):
                self.agent_ack.remove(agent_name)

        if len(self.agent_ack) == 0:
            print("People in the conversation: ", list(self.agents.keys()) + self.users)
            print("Orchestrator - S3: All Agents Have Defined the Relathinship with the New Agent")
            self.orchestrator.pop_state()
        else:
            print("Orchestrator - S3: Waiting for All Agents to Define the Relationship with the New Agent")
        
    #S4
    def _retrieval_ack_and_selection(self):
        for agent_name in self.agent_ack.copy():
            ack, similarity_index = self.agents[agent_name]["reference"].get_context_sim(self.orchestrator_id)
            if ack:
                self.agent_ack.remove(agent_name)
                participation_index = self.agents[agent_name]["round_wo_talking"] / self.max_round_wo_talking
                speak_permission_index = 0.3*participation_index + 0.7*similarity_index
                if self.max_cos_sim_index < speak_permission_index:
                    self.max_cos_sim_index = speak_permission_index
                    self.future_speaker = agent_name

        if len(self.agent_ack) == 0:
            self.possible_speakers.remove(self.future_speaker)
            print("Orchestrator - S4: Retreival Context Confirmed From All Agents - New Speaker Selected")
            self.orchestrator.pop_state()
        else:
            print("Orchestrator - S4: Waiting Retreival Context Confirmation From All Agents")

    #S5
    def _retrieval_ack(self):
        for agent_name in self.agent_ack.copy():
            ack, _ = self.agents[agent_name]["reference"].get_context_sim(self.orchestrator_id)
            if ack:
                self.agent_ack.remove(agent_name)
        if len(self.agent_ack) == 0:
            print("Orchestrator - S5: Retreival Context Confirmed From All Agents")
            self.orchestrator.pop_state()
        else:
            print("Orchestrator - S5: Waiting Retreival Context Confirmation From All Agents")

    #S6
    def _inform_no_permission(self):
        print("Orchestrator - S6: Inform Agents with Permission Denied")
        max_round_wo_talk = 1

        for agent_name in self.possible_speakers:
            self.agents[agent_name]["reference"].set_permission_2_talk(False, "", "")
            self.agents[agent_name]["round_wo_talking"] += 1
            if self.agents[agent_name]["round_wo_talking"] > max_round_wo_talk:
                max_round_wo_talk = self.agents[agent_name]["round_wo_talking"]

        if self.current_speaker in self.agents.keys():
            self.agents[self.current_speaker]["round_wo_talking"] = 0     
            self.possible_speakers.append(self.current_speaker)

        self.current_speaker = self.future_speaker
        print(max_round_wo_talk)
        self.max_round_wo_talking = max_round_wo_talk
        self.orchestrator.pop_state()

    #S7
    def _inform_permission(self):
        print("Orchestrator - S7: Inform Agent with Permission Granted")

        initial_prompt, question_task = self._build_context_for_prompt()
        self.agents[self.current_speaker]["reference"].set_permission_2_talk(True, initial_prompt, question_task)
        self.agents[self.current_speaker]["round_wo_talking"] = 0
        self.new_joiners = []
        self.orchestrator.pop_state()
        print("permission given")

    #S8
    def _wait_new_agent_message(self):
        last_message = self.agents[self.current_speaker]["reference"].get_last_message(self.orchestrator_id)
        
        if self.last_message != last_message:
            self.convo.append(last_message)
            self.last_message = last_message
            print("Orchestrator - S8: Ask User for New Message")
            self.orchestrator.pop_state()
            
        else:
            print("Orchestrator - S8: Ask User for New Message")

    #S9
    def _user_speak(self):
        print("Orchestrator - S9: Ask User for New Message")
        user_message = input("Type the message: ")
        self.last_message = self.current_speaker + ":" + user_message
        print(self.last_message)
        self.convo.append(self.last_message)
        self.orchestrator.pop_state()

    #10
    def _distribute_new_message(self):
        for agent_name in self.possible_speakers:
            self.agents[agent_name]["reference"].set_new_message(self.orchestrator_id, self.last_message)
        self.future_speaker = ""
        self.max_cos_sim_index = 0
        print("Orchestrator - S10: New Message Received - Decide Next Action")
        self.orchestrator.pop_state()

    


    def execute_sfsm(self):
        #print(self.orchestrator.stack)
        self.orchestrator.update()