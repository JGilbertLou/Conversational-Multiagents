"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: conference.py
Description: This is the main program for running generative agent simulations
that defines the ReverieServer class. This class maintains and records all  
states related to the simulation. The primary mode of interaction for those  
running the simulation should be through the open_server function, which  
enables the simulator to input command-line prompts for running and saving  
the simulation, among other tasks.

Release note (June 14, 2023) -- Reverie implements the core simulation 
mechanism described in my paper entitled "Generative Agents: Interactive 
Simulacra of Human Behavior." If you are reading through these lines after 
having read the paper, you might notice that I use older terms to describe 
generative agents and their cognitive modules here. Most notably, I use the 
term "agents" to refer to generative agents, "associative memory" to refer 
to the memory stream, and "reverie" to refer to the overarching simulation 
framework.
"""
import json
import numpy
import datetime
import pickle
import time
import math
import os
import shutil
import traceback
import torch
import warnings

#from selenium import webdriver

from global_methods import *
from utils import *
from maze import *
from persona.agent import *
from persona.cognitive_modules.retrieve import *
from persona.cognitive_modules.convo.director import *
from persona.cognitive_modules.convo.orchestrator import *
from persona.prompt_template.gpt_structure import *
from persona.cognitive_modules.converse import *
from persona.memory_structures.associative_memory import *
from persona.text_processing import *

from transformers import BertTokenizer, BertModel, BartTokenizer, BartModel, RagTokenizer, RagRetriever, RagSequenceForGeneration, DPRQuestionEncoderTokenizer, DPRQuestionEncoder, RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from rake_nltk import Rake
#nltk.download('punkt')

##############################################################################
#                                CONFERENCE                                  #
##############################################################################

class ConferenceServer: 
  def __init__(self, 
               fork_sim_code,
               sim_code):
    # FORKING FROM A PRIOR SIMULATION:
    # <fork_sim_code> indicates the simulation we are forking from. 
    # Interestingly, all simulations must be forked from some initial 
    # simulation, where the first simulation is "hand-crafted".
    self.fork_sim_code = fork_sim_code
    fork_folder = f"{fs_storage}/{self.fork_sim_code}"
    # <sim_code> indicates our current simulation. The first step here is to 
    # copy everything that's in <fork_sim_code>, but edit its 
    # reverie/meta/json's fork variable. 
    self.sim_code = sim_code
    sim_folder = f"{fs_storage}/{self.sim_code}"
    #copyanything(fork_folder, sim_folder)

    with open(f"{sim_folder}/reverie/meta.json") as json_file:  
      reverie_meta = json.load(json_file)

    with open(f"{sim_folder}/reverie/meta.json", "w") as outfile: 
      reverie_meta["fork_sim_code"] = fork_sim_code
      outfile.write(json.dumps(reverie_meta, indent=2))


    # LOADING REVERIE'S GLOBAL VARIABLES
    # The start datetime of the Reverie: 
    # <start_datetime> is the datetime instance for the start datetime of 
    # the Reverie instance. Once it is set, this is not really meant to 
    # change. It takes a string date in the following example form: 
    # "June 25, 2022"
    # e.g., ...strptime(June 25, 2022, "%B %d, %Y")
    self.start_time = datetime.datetime.strptime(
                        f"{reverie_meta['start_date']}, 00:00:00",  
                        "%B %d, %Y, %H:%M:%S")
    # <curr_time> is the datetime instance that indicates the game's current
    # time. This gets incremented by <sec_per_step> amount everytime the world
    # progresses (that is, everytime curr_env_file is recieved). 
    self.curr_time = datetime.datetime.strptime(reverie_meta['curr_time'], 
                                                "%B %d, %Y, %H:%M:%S")
    # <sec_per_step> denotes the number of seconds in game time that each 
    # step moves foward. 
    self.sec_per_step = reverie_meta['sec_per_step']
    
    # <step> denotes the number of steps that our game has taken. A step here
    # literally translates to the number of moves our agents made in terms
    # of the number of tiles. 
    self.step = reverie_meta['step']

    # SETTING UP agents IN REVERIE
    # <agents> is a dictionary that takes the persona's full name as its 
    # keys, and the actual persona instance as its values.
    # This dictionary is meant to keep track of all agents who are part of
    # the Reverie instance. 
    # e.g., ["Isabella Rodriguez"] = Persona("Isabella Rodriguezs")
    self.agents = dict()
    self.agents_names = []

    # Loading in all agents. 
    init_env_file = f"{sim_folder}/environment/{str(self.step)}.json"
    init_env = json.load(open(init_env_file))

    for agent_name in reverie_meta['persona_names']: 
      persona_folder = f"{sim_folder}/personas/{agent_name}"
      p_x = init_env[agent_name]["x"]
      p_y = init_env[agent_name]["y"]
      curr_persona = Agent(agent_name, persona_folder)

      self.agents[agent_name] = curr_persona
      self.agents_names.append(agent_name)

    # REVERIE SETTINGS PARAMETERS:  
    # <server_sleep> denotes the amount of time that our while loop rests each
    # cycle; this is to not kill our machine. 
    self.server_sleep = 0.1

  def save(self): 
    """
    Save all Reverie progress -- this includes Reverie's global state as well
    as all the agents.  

    INPUT
      None
    OUTPUT 
      None
      * Saves all relevant data to the designated memory directory
    """
    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    # Save Reverie meta information.
    reverie_meta = dict() 
    reverie_meta["fork_sim_code"] = self.fork_sim_code
    reverie_meta["start_date"] = self.start_time.strftime("%B %d, %Y")
    reverie_meta["curr_time"] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
    reverie_meta["sec_per_step"] = self.sec_per_step
    reverie_meta["maze_name"] = self.maze.maze_name
    reverie_meta["persona_names"] = list(self.agents.keys())
    reverie_meta["step"] = self.step
    reverie_meta_f = f"{sim_folder}/reverie/meta.json"
    with open(reverie_meta_f, "w") as outfile: 
      outfile.write(json.dumps(reverie_meta, indent=2))

    # Save the agents.
    for persona_name, persona in self.agents.items(): 
      save_folder = f"{sim_folder}/personas/{persona_name}/bootstrap_memory"
      persona.save(save_folder)

  def start_path_tester_server(self): 
    """
    Starts the path tester server. This is for generating the spatial memory
    that we need for bootstrapping a persona's state. 

    To use this, you need to open server and enter the path tester mode, and
    open the front-end side of the browser. 

    INPUT 
      None
    OUTPUT 
      None
      * Saves the spatial memory of the test agent to the path_tester_env.json
        of the temp storage. 
    """
    def print_tree(tree): 
      def _print_tree(tree, depth):
        dash = " >" * depth

        if type(tree) == type(list()): 
          if tree:
            print (dash, tree)
          return 

        for key, val in tree.items(): 
          if key: 
            print (dash, key)
          _print_tree(val, depth+1)
      
      _print_tree(tree, 0)

    # <curr_vision> is the vision radius of the test agent. Recommend 8 as 
    # our default. 
    curr_vision = 8
    # <s_mem> is our test spatial memory. 
    s_mem = dict()

    # The main while loop for the test agent. 
    while (True): 
      try: 
        curr_dict = {}
        tester_file = fs_temp_storage + "/path_tester_env.json"
        if check_if_file_exists(tester_file): 
          with open(tester_file) as json_file: 
            curr_dict = json.load(json_file)
            os.remove(tester_file)
          
          # Current camera location
          curr_sts = self.maze.sq_tile_size
          curr_camera = (int(math.ceil(curr_dict["x"]/curr_sts)), 
                         int(math.ceil(curr_dict["y"]/curr_sts))+1)
          curr_tile_det = self.maze.access_tile(curr_camera)

          # Initiating the s_mem
          world = curr_tile_det["world"]
          if curr_tile_det["world"] not in s_mem: 
            s_mem[world] = dict()

          # Iterating throughn the nearby tiles.
          nearby_tiles = self.maze.get_nearby_tiles(curr_camera, curr_vision)
          for i in nearby_tiles: 
            i_det = self.maze.access_tile(i)
            if (curr_tile_det["sector"] == i_det["sector"] 
                and curr_tile_det["arena"] == i_det["arena"]): 
              if i_det["sector"] != "": 
                if i_det["sector"] not in s_mem[world]: 
                  s_mem[world][i_det["sector"]] = dict()
              if i_det["arena"] != "": 
                if i_det["arena"] not in s_mem[world][i_det["sector"]]: 
                  s_mem[world][i_det["sector"]][i_det["arena"]] = list()
              if i_det["game_object"] != "": 
                if (i_det["game_object"] 
                    not in s_mem[world][i_det["sector"]][i_det["arena"]]):
                  s_mem[world][i_det["sector"]][i_det["arena"]] += [
                                                         i_det["game_object"]]

        # Incrementally outputting the s_mem and saving the json file. 
        print ("= " * 15)
        out_file = fs_temp_storage + "/path_tester_out.json"
        with open(out_file, "w") as outfile: 
          outfile.write(json.dumps(s_mem, indent=2))
        print_tree(s_mem)

      except:
        pass

      time.sleep(self.server_sleep * 10)

  def start_server_conversation(self, int_counter):
    while (True): 
      # Done with this iteration if <int_counter> reaches 0. 
      if int_counter == 0: 
        break

      movements = {"persona": dict(), 
                       "meta": dict()}
      for persona_name, persona in self.agents.items(): 
        # <next_tile> is a x,y coordinate. e.g., (58, 9)
        # <pronunciatio> is an emoji. e.g., "\ud83d\udca4"
        # <description> is a string description of the movement. e.g., 
        #   writing her next novel (editing her novel) 
        #   @ double studio:double studio:common room:sofa
        next_tile, pronunciatio, description = persona.move(
          self.maze, self.agents, self.agents_tile[persona_name], 
          self.curr_time)
        movements["persona"][persona_name] = {}
        movements["persona"][persona_name]["pronunciatio"] = pronunciatio
        movements["persona"][persona_name]["description"] = description
        movements["persona"][persona_name]["chat"] = (persona
                                                      .scratch.chat)

      # Include the meta information about the current stage in the 
      # movements dictionary. 
      movements["meta"]["curr_time"] = (self.curr_time 
                                          .strftime("%B %d, %Y, %H:%M:%S"))

      # We then write the agents' movements to a file that will be sent 
      # to the frontend server. 
      # Example json output: 
      # {"persona": {"Maria Lopez": {"movement": [58, 9]}},
      #  "persona": {"Klaus Mueller": {"movement": [38, 12]}}, 
      #  "meta": {curr_time: <datetime>}}
      curr_move_path = "f{sim_folder}/movement"
      if not os.path.exists(curr_move_path):
        os.makedirs(curr_move_path)
        
      curr_move_file = f"{sim_folder}/movement/{self.step}.json"
      with open(curr_move_file, "w") as outfile: 
        outfile.write(json.dumps(movements, indent=2))

      # After this cycle, the world takes one step forward, and the 
      # current time moves by <sec_per_step> amount. 
      self.step += 1
      self.curr_time += datetime.timedelta(seconds=self.sec_per_step)

      int_counter -= 1
          
      # Sleep so we don't burn our machines. 
    time.sleep(self.server_sleep)

  def start_server(self, int_counter): 
    """
    The main backend server of Reverie. 
    This function retrieves the environment file from the frontend to 
    understand the state of the world, calls on each agents to make 
    decisions based on the world state, and saves their moves at certain step
    intervals. 
    INPUT
      int_counter: Integer value for the number of steps left for us to take
                   in this iteration. 
    OUTPUT 
      None
    """
    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    # When a persona arrives at a game object, we give a unique event
    # to that object. 
    # e.g., ('double studio[...]:bed', 'is', 'unmade', 'unmade')
    # Later on, before this cycle ends, we need to return that to its 
    # initial state, like this: 
    # e.g., ('double studio[...]:bed', None, None, None)
    # So we need to keep track of which event we added. 
    # <game_obj_cleanup> is used for that. 
    game_obj_cleanup = dict()

    # The main while loop of Reverie. 
    while (True): 
      # Done with this iteration if <int_counter> reaches 0. 
      if int_counter == 0: 
        break

      # <curr_env_file> file is the file that our frontend outputs. When the
      # frontend has done its job and moved the agents, then it will put a 
      # new environment file that matches our step count. That's when we run 
      # the content of this for loop. Otherwise, we just wait. 
      curr_env_file = f"{sim_folder}/environment/{self.step}.json"
      if check_if_file_exists(curr_env_file):
        # If we have an environment file, it means we have a new perception
        # input to our agents. So we first retrieve it.
        try: 
          # Try and save block for robustness of the while loop.
          with open(curr_env_file) as json_file:
            new_env = json.load(json_file)
            env_retrieved = True
        except: 
          pass
      
        if env_retrieved: 
          # This is where we go through <game_obj_cleanup> to clean up all 
          # object actions that were used in this cylce. 
          for key, val in game_obj_cleanup.items(): 
            # We turn all object actions to their blank form (with None). 
            self.maze.turn_event_from_tile_idle(key, val)
          # Then we initialize game_obj_cleanup for this cycle. 
          game_obj_cleanup = dict()

          # We first move our agents in the backend environment to match 
          # the frontend environment. 
          for persona_name, persona in self.agents.items(): 
            # <curr_tile> is the tile that the persona was at previously. 
            curr_tile = self.agents_tile[persona_name]
            # <new_tile> is the tile that the persona will move to right now,
            # during this cycle. 
            new_tile = (new_env[persona_name]["x"], 
                        new_env[persona_name]["y"])

            # We actually move the persona on the backend tile map here. 
            self.agents_tile[persona_name] = new_tile
            self.maze.remove_subject_events_from_tile(persona.name, curr_tile)
            self.maze.add_event_from_tile(persona.scratch
                                         .get_curr_event_and_desc(), new_tile)

            # Now, the persona will travel to get to their destination. *Once*
            # the persona gets there, we activate the object action.
            if not persona.scratch.planned_path: 
              # We add that new object action event to the backend tile map. 
              # At its creation, it is stored in the persona's backend. 
              game_obj_cleanup[persona.scratch
                               .get_curr_obj_event_and_desc()] = new_tile
              self.maze.add_event_from_tile(persona.scratch
                                     .get_curr_obj_event_and_desc(), new_tile)
              # We also need to remove the temporary blank action for the 
              # object that is currently taking the action. 
              blank = (persona.scratch.get_curr_obj_event_and_desc()[0], 
                       None, None, None)
              self.maze.remove_event_from_tile(blank, new_tile)

          # Then we need to actually have each of the agents perceive and
          # move. The movement for each of the agents comes in the form of
          # x y coordinates where the persona will move towards. e.g., (50, 34)
          # This is where the core brains of the agents are invoked. 
          movements = {"persona": dict(), 
                       "meta": dict()}
          for persona_name, persona in self.agents.items(): 
            # <next_tile> is a x,y coordinate. e.g., (58, 9)
            # <pronunciatio> is an emoji. e.g., "\ud83d\udca4"
            # <description> is a string description of the movement. e.g., 
            #   writing her next novel (editing her novel) 
            #   @ double studio:double studio:common room:sofa
            next_tile, pronunciatio, description = persona.move(
              self.maze, self.agents, self.agents_tile[persona_name], 
              self.curr_time)
            movements["persona"][persona_name] = {}
            movements["persona"][persona_name]["movement"] = next_tile
            movements["persona"][persona_name]["pronunciatio"] = pronunciatio
            movements["persona"][persona_name]["description"] = description
            movements["persona"][persona_name]["chat"] = (persona
                                                          .scratch.chat)

          # Include the meta information about the current stage in the 
          # movements dictionary. 
          movements["meta"]["curr_time"] = (self.curr_time 
                                             .strftime("%B %d, %Y, %H:%M:%S"))

          # We then write the agents' movements to a file that will be sent 
          # to the frontend server. 
          # Example json output: 
          # {"persona": {"Maria Lopez": {"movement": [58, 9]}},
          #  "persona": {"Klaus Mueller": {"movement": [38, 12]}}, 
          #  "meta": {curr_time: <datetime>}}
          curr_move_path = "f{sim_folder}/movement"
          if not os.path.exists(curr_move_path):
            os.makedirs(curr_move_path)
            
          curr_move_file = f"{sim_folder}/movement/{self.step}.json"
          with open(curr_move_file, "w") as outfile: 
            outfile.write(json.dumps(movements, indent=2))

          # After this cycle, the world takes one step forward, and the 
          # current time moves by <sec_per_step> amount. 
          self.step += 1
          self.curr_time += datetime.timedelta(seconds=self.sec_per_step)

          int_counter -= 1
          
      # Sleep so we don't burn our machines. 
      time.sleep(self.server_sleep)

  def open_server(self): 
    """
    Open up an interactive terminal prompt that lets you run the simulation 
    step by step and probe agent state. 

    INPUT 
      None
    OUTPUT
      None
    """
    print ("Note: The agents in this simulation package are computational")
    print ("constructs powered by generative agents architecture and LLM. We")
    print ("clarify that these agents lack human-like agency, consciousness,")
    print ("and independent decision-making.\n---")

    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    while True: 
      sim_command = input("Enter option: ")
      sim_command = sim_command.strip()
      ret_str = ""

      try: 
        if sim_command.lower() in ["f", "fin", "finish", "save and finish"]: 
          # Finishes the simulation environment and saves the progress. 
          # Example: fin
          self.save()
          break
        elif "1v1" in sim_command.lower():
          print(self.agents["Isabella Rodriguez"].scratch.name)
          print(self.agents["Klaus Mueller"].scratch.name)
          print(self.agents["Isabella Rodriguez"].one_v_one_session(self.agents["Klaus Mueller"]))

        elif sim_command.lower() == "start path tester mode": 
          # Starts the path tester and removes the currently forked sim files.
          # Note that once you start this mode, you need to exit out of the
          # session and restart in case you want to run something else. 
          shutil.rmtree(sim_folder) 
          self.start_path_tester_server()

        elif sim_command.lower() == "exit": 
          # Finishes the simulation environment but does not save the progress
          # and erases all saved data from current simulation. 
          # Example: exit 
          shutil.rmtree(sim_folder) 
          break 

        elif sim_command.lower() == "save": 
          # Saves the current simulation progress. 
          # Example: save
          self.save()

        elif sim_command[:3].lower() == "run": 
          # Runs the number of steps specified in the prompt.
          # Example: run 1000
          int_count = int(sim_command.split()[-1])
          rs.start_server(int_count)

        elif ("print persona schedule" 
              in sim_command[:22].lower()): 
          # Print the decomposed schedule of the persona specified in the 
          # prompt.
          # Example: print persona schedule Isabella Rodriguez
          ret_str += (self.agents[" ".join(sim_command.split()[-2:])]
                      .scratch.get_str_daily_schedule_summary())

        elif ("print all persona schedule" 
              in sim_command[:26].lower()): 
          # Print the decomposed schedule of all agents in the world. 
          # Example: print all persona schedule
          for persona_name, persona in self.agents.items(): 
            ret_str += f"{persona_name}\n"
            ret_str += f"{persona.scratch.get_str_daily_schedule_summary()}\n"
            ret_str += f"---\n"

        elif ("print hourly org persona schedule" 
              in sim_command.lower()): 
          # Print the hourly schedule of the persona specified in the prompt.
          # This one shows the original, non-decomposed version of the 
          # schedule.
          # Ex: print persona schedule Isabella Rodriguez
          ret_str += (self.agents[" ".join(sim_command.split()[-2:])]
                      .scratch.get_str_daily_schedule_hourly_org_summary())
      
        elif ("print persona current tile" 
              in sim_command[:26].lower()): 
          # Print the x y tile coordinate of the persona specified in the 
          # prompt. 
          # Ex: print persona current tile Isabella Rodriguez
          ret_str += str(self.agents[" ".join(sim_command.split()[-2:])]
                      .scratch.curr_tile)

        elif ("print persona chatting with buffer" 
              in sim_command.lower()): 
          # Print the chatting with buffer of the persona specified in the 
          # prompt.
          # Ex: print persona chatting with buffer Isabella Rodriguez
          curr_persona = self.agents[" ".join(sim_command.split()[-2:])]
          for p_n, count in curr_persona.scratch.chatting_with_buffer.items(): 
            ret_str += f"{p_n}: {count}"

        elif ("print persona associative memory (event)" 
              in sim_command.lower()):
          # Print the associative memory (event) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (event) Isabella Rodriguez
          ret_str += f'{self.agents[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.agents[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_events())

        elif ("print persona associative memory (thought)" 
              in sim_command.lower()): 
          # Print the associative memory (thought) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (thought) Isabella Rodriguez
          ret_str += f'{self.agents[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.agents[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_thoughts())

        elif ("print persona associative memory (chat)" 
              in sim_command.lower()): 
          # Print the associative memory (chat) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (chat) Isabella Rodriguez
          ret_str += f'{self.agents[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.agents[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_chats())

        elif ("print persona spatial memory" 
              in sim_command.lower()): 
          # Print the spatial memory of the persona specified in the prompt
          # Ex: print persona spatial memory Isabella Rodriguez
          self.agents[" ".join(sim_command.split()[-2:])].s_mem.print_tree()

        elif ("print current time" 
              in sim_command[:18].lower()): 
          # Print the current time of the world. 
          # Ex: print current time
          ret_str += f'{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}\n'
          ret_str += f'steps: {self.step}'

        elif ("print tile event" 
              in sim_command[:16].lower()): 
          # Print the tile events in the tile specified in the prompt 
          # Ex: print tile event 50, 30
          cooordinate = [int(i.strip()) for i in sim_command[16:].split(",")]
          for i in self.maze.access_tile(cooordinate)["events"]: 
            ret_str += f"{i}\n"

        elif ("print tile details" 
              in sim_command.lower()): 
          # Print the tile details of the tile specified in the prompt 
          # Ex: print tile event 50, 30
          cooordinate = [int(i.strip()) for i in sim_command[18:].split(",")]
          for key, val in self.maze.access_tile(cooordinate).items(): 
            ret_str += f"{key}: {val}\n"

        elif ("call -- analysis" 
              in sim_command.lower()): 
          # Starts a stateless chat session with the agent. It does not save 
          # anything to the agent's memory. 
          # Ex: call -- analysis Isabella Rodriguez
          persona_name = sim_command[len("call -- analysis"):].strip() 
          self.agents[persona_name].open_convo_session("analysis")

        elif ("call -- load history" 
              in sim_command.lower()): 
          curr_file = maze_assets_loc + "/" + sim_command[len("call -- load history"):].strip() 
          # call -- load history the_ville/agent_history_init_n3.csv

          rows = read_file_to_list(curr_file, header=True, strip_trail=True)[1]
          clean_whispers = []
          for row in rows: 
            agent_name = row[0].strip() 
            whispers = row[1].split(";")
            whispers = [whisper.strip() for whisper in whispers]
            for whisper in whispers: 
              clean_whispers += [[agent_name, whisper]]

          load_history_via_whisper(self.agents, clean_whispers)

        print (ret_str)

      except:
        traceback.print_exc()
        print ("Error.")
        pass
      

  def open_confenrence(self):
    sim_folder = f"{fs_storage}/{self.sim_code}"      
    # Function to get sentence embedding
    '''def calculate_embedding(sentence, tokenizer, model):
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Get the hidden states from the last layer
        last_hidden_states = outputs.last_hidden_state
        # Average the token embeddings to get a sentence embedding
        sentence_embedding = torch.mean(last_hidden_states, dim=1)
        return sentence_embedding
    
    def embed_sentence(sentence, tokenizer, model):
        inputs = tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = model.question_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return outputs[0].cpu().numpy()
    

    rake = Rake()

    # Extract keywords
    bert_large_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    bert_large_model = BertModel.from_pretrained('bert-large-uncased')

    roberta_large_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    roberta_large_model = RobertaModel.from_pretrained('roberta-large')

    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    bart_model = BartModel.from_pretrained('facebook/bart-large')  

    sbert_model = SentenceTransformer('stsb-roberta-large')

    
    for agent_name, agent in self.agents.items():
      bert_embedding = dict()
      bart_embedding = dict()
      sbert_embedding = dict()
      gpt_embedding_3 = dict()
      save_folder = f"{sim_folder}/personas/{agent_name}/bootstrap_memory"
      for key, node in agent.c_mem.id_to_node.items():
        embedding_key = node.embedding_key
        if embedding_key not in sbert_embedding.keys():
          #bert_embedding[embedding_key] = calculate_embedding(embedding_key, bert_large_tokenizer, bert_large_model).numpy().tolist()
          #bart_embedding[embedding_key] = calculate_embedding(embedding_key, bart_tokenizer, bart_model).numpy().tolist()
          sbert_embedding[embedding_key] = sbert_model.encode(embedding_key, convert_to_tensor=True).numpy().tolist()
          #gpt_embedding_3[embedding_key] = get_embedding(embedding_key, 'text-embedding-3-large')
      #agent.save_embedding(save_folder, bert_embedding, "bert_embedding.json")
      #agent.save_embedding(save_folder, bart_embedding, "bart_embedding.json")
      agent.save_embedding(save_folder, sbert_embedding, "sbert_embedding.json")
      #agent.save_embedding(save_folder, gpt_embedding_3, "gpt_3_embedding.json")'''
                                               
    '''for agent_name, agent in self.agents.items(): 
      agent.save(save_folder, hippo_nodes, keyword_embeddings)
    # Download the punkt tokenizer if it's not already installed'''
    
    # Step 2: Define your array of sentences and the theme
    '''sentence1 = "Isabella Rodriguez is preparing for the Valentine's Day party"
    sentence2 = "Valentine's Party"

    # Initialize RAKE by providing a list of stopwords
    rake = Rake()

    # Extract keywords
    rake.extract_keywords_from_text(sentence1)
    keywords = rake.get_ranked_phrases()

    print("Keywords:", keywords)
    print(self.agents[self.agents_names[0]].c_mem.id_to_node['node_910'].description)'''

    # Step 1: Load pre-trained model
    '''sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 3: Generate embeddings for the sentences and the theme
    sentence_embeddings = sentence_transformer_model.encode(sentence1)
    theme_embedding = sentence_transformer_model.encode([sentence2])[0]

    # Step 4: Calculate cosine similarities
    similarities = cosine_similarity([theme_embedding], [sentence_embeddings])[0]
    print(similarities)'''
    # Step 5: Define a similarity threshold
    '''threshold = 0.5  # This threshold can be adjusted

    # Step 6: Filter sentences based on the threshold
    filtered_sentences = [sentences[i] for i in range(len(sentences)) if similarities[i] >= threshold]

    print("Filtered Sentences:")
    for sentence in filtered_sentences:
        print(sentence)'''
    
    '''high_cos_sim_phrases = [
    ["She felt a sense of accomplishment after finishing the project.", 
     "Completing the task gave her a feeling of fulfillment."],
    ["The quiet of the morning was soothing.", 
     "She found peace in the early hours of the day."],
    ["He couldn't wait for the event to start.", 
     "The upcoming occasion filled him with eager anticipation."],
    ["She discovered new ideas through the book.", 
     "The book introduced her to unfamiliar concepts."],
    ["Traveling abroad opened his eyes to new cultures.", 
     "Experiencing different countries broadened his understanding of the world."],
    ["He relished the silence after a busy day.", 
     "The calm after the day's chaos was something he cherished."],
    ["Walking through the forest made her feel connected to the earth.", 
     "She felt a profound bond with nature while hiking among the trees."],
    ["It slowly dawned on her that she had been wrong.", 
     "The realization of her mistake came gradually."]]
    high_phrase_num = len(high_cos_sim_phrases)
    
    low_cos_sim_phrases = [
    ["She felt a sense of accomplishment after finishing the project.", 
     "The weather forecast predicts rain for tomorrow."],    
    ["The quiet of the morning was soothing.", 
     "He is planning to buy a new car next week."],    
    ["He couldn't wait for the event to start.", 
     "The cake recipe requires two cups of sugar."],    
    ["She discovered new ideas through the book.", 
     "The train was delayed due to technical issues."],    
    ["Traveling abroad opened his eyes to new cultures.", 
     "The stock market saw significant gains today."],    
    ["He relished the silence after a busy day.", 
     "She prefers her coffee with cream and sugar."],  
    ["Walking through the forest made her feel connected to the earth.", 
     "The company announced its quarterly earnings report."],
    ["It slowly dawned on her that she had been wrong.", 
     "The software update includes new security features."]]
    low_phrase_num = len(low_cos_sim_phrases)

 # Load pre-trained BERT model and tokenizer
    bert_large_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    bert_large_model = BertModel.from_pretrained('bert-large-uncased')

    roberta_large_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    roberta_large_model = RobertaModel.from_pretrained('roberta-large')

    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    bart_model = BartModel.from_pretrained('facebook/bart-large')  

    sbert_model = SentenceTransformer('stsb-roberta-large')

    # Example sentences
    sentence1 = "Completing the task gave her a feeling of fulfillment."
    sentence2 = "She felt a sense of accomplishment after finishing the project."


    # Get embeddings
    gen_agent_high_cos_sim = 0
    gen_agent_low_cos_sim = 0
    sgen_agent_high_cos_sim = 0
    sgen_agent_low_cos_sim = 0
    lgen_agent_high_cos_sim = 0
    lgen_agent_low_cos_sim = 0
    bert_high_cos_sim = 0
    bert_low_cos_sim = 0
    roberta_high_cos_sim = 0
    roberta_low_cos_sim = 0
    bart_high_cos_sim = 0
    bart_low_cos_sim = 0
    sbert_high_cos_sim = 0
    sbert_low_cos_sim = 0

    for phrases in high_cos_sim_phrases:
      gen_agent_high_cos_sim += cos_sim(get_embedding(phrases[0]), get_embedding(phrases[1]))

      sgen_agent_high_cos_sim += cos_sim(get_embedding(sentence1, 'text-embedding-3-small'), get_embedding(sentence2, 'text-embedding-3-small'))

      lgen_agent_high_cos_sim += cos_sim(get_embedding(sentence1, 'text-embedding-3-large'), get_embedding(sentence2, 'text-embedding-3-large'))

      bert_embedding1 = calculate_embedding(phrases[0], bert_large_tokenizer, bert_large_model)
      bert_embedding2 = calculate_embedding(phrases[1], bert_large_tokenizer, bert_large_model)
      bert_high_cos_sim += cosine_similarity(bert_embedding1.numpy(), bert_embedding2.numpy())[0][0]

      roberta_embedding1 = calculate_embedding(phrases[0], roberta_large_tokenizer, roberta_large_model)
      roberta_embedding2 = calculate_embedding(phrases[1], roberta_large_tokenizer, roberta_large_model)
      roberta_high_cos_sim += cosine_similarity(roberta_embedding1.numpy(), roberta_embedding2.numpy())[0][0]

      bart_embedding1 = calculate_embedding(phrases[0], bart_tokenizer, bart_model)
      bart_embedding2 = calculate_embedding(phrases[1], bart_tokenizer, bart_model)
      bart_high_cos_sim += cosine_similarity(bart_embedding1.numpy(), bart_embedding2.numpy())[0][0]

      sbert_embedding1 = sbert_model.encode(phrases[0], convert_to_tensor=True)
      sbert_embedding2 = sbert_model.encode(phrases[1], convert_to_tensor=True)
      sbert_high_cos_sim += cosine_similarity(sbert_embedding1.detach().numpy().reshape(1, -1), sbert_embedding2.detach().numpy().reshape(1, -1))[0][0]

    for phrases in low_cos_sim_phrases:
      gen_agent_low_cos_sim += cos_sim(get_embedding(phrases[0]), get_embedding(phrases[1]))

      sgen_agent_low_cos_sim += cos_sim(get_embedding(sentence1, 'text-embedding-3-small'), get_embedding(sentence2, 'text-embedding-3-small'))

      lgen_agent_low_cos_sim += cos_sim(get_embedding(sentence1, 'text-embedding-3-large'), get_embedding(sentence2, 'text-embedding-3-large'))

      bert_embedding1 = calculate_embedding(phrases[0], bert_large_tokenizer, bert_large_model)
      bert_embedding2 = calculate_embedding(phrases[1], bert_large_tokenizer, bert_large_model)
      bert_low_cos_sim += cosine_similarity(bert_embedding1.numpy(), bert_embedding2.numpy())[0][0]

      roberta_embedding1 = calculate_embedding(phrases[0], roberta_large_tokenizer, roberta_large_model)
      roberta_embedding2 = calculate_embedding(phrases[1], roberta_large_tokenizer, roberta_large_model)
      roberta_low_cos_sim += cosine_similarity(roberta_embedding1.numpy(), roberta_embedding2.numpy())[0][0]

      bart_embedding1 = calculate_embedding(phrases[0], bart_tokenizer, bart_model)
      bart_embedding2 = calculate_embedding(phrases[1], bart_tokenizer, bart_model)
      bart_low_cos_sim += cosine_similarity(bart_embedding1.numpy(), bart_embedding2.numpy())[0][0]

      sbert_embedding1 = sbert_model.encode(phrases[0], convert_to_tensor=True)
      sbert_embedding2 = sbert_model.encode(phrases[1], convert_to_tensor=True)
      sbert_low_cos_sim += cosine_similarity(sbert_embedding1.detach().numpy().reshape(1, -1), sbert_embedding2.detach().numpy().reshape(1, -1))[0][0]


    print("Gen-Agents Avg. High Cosine Similarity: ", gen_agent_high_cos_sim/high_phrase_num)
    print("Small Gen-Agents Avg. High Cosine Similarity: ", sgen_agent_high_cos_sim/high_phrase_num)
    print("Large Gen-Agents Avg. High Cosine Similarity: ", lgen_agent_high_cos_sim/high_phrase_num)
    print("BERT Avg. High Cosine Similarity:", bert_high_cos_sim/high_phrase_num)
    print("RoBERTa Avg. High Cosine Similarity:", roberta_high_cos_sim/high_phrase_num)
    print("BART Avg. High Cosine Similarity:", bart_high_cos_sim/high_phrase_num)
    print("Sentence-BERT Avg. High Cosine Similarity:", sbert_high_cos_sim/high_phrase_num)

    print("Gen-Agents Avg. Low Cosine Similarity: ", gen_agent_low_cos_sim/low_phrase_num)
    print("Small Gen-Agents Avg. Low Cosine Similarity: ", sgen_agent_low_cos_sim/low_phrase_num)
    print("Large Gen-Agents Avg. Low Cosine Similarity: ", lgen_agent_low_cos_sim/low_phrase_num)
    print("BERT Avg. Low Cosine Similarity:", bert_low_cos_sim/low_phrase_num)
    print("RoBERTa Avg. Low Cosine Similarity:", roberta_low_cos_sim/low_phrase_num)
    print("BART Avg. Low Cosine Similarity:", bart_low_cos_sim/low_phrase_num)
    print("Sentence-BERT Avg. Low Cosine Similarity:", sbert_low_cos_sim/low_phrase_num)'''
    #print(similarities)

    '''rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    rag_retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True, trust_remote_code=True)
    rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=rag_retriever)

    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

    # Explicitly load the dataset with trust_remote_code=True
    dataset = load_dataset('wiki_dpr', split='train', trust_remote_code=True)

    # Initialize the retriever with the dataset
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", dataset=dataset)



    main_embedding = embed_sentence(sentence2, tokenizer, model)
    sentence_embeddings = [embed_sentence(node.embedding_key, tokenizer, model) for count, node in enumerate(nodes)]

    similarities = cosine_similarity(main_embedding, np.vstack(sentence_embeddings)).flatten()


# Get the indices of the top N most similar sentences
    top_n = 10
    top_indices = similarities.argsort()[-top_n:][::-1]


    [i.last_accessed, i]
    # Retrieve the top N sentences
    top_sentences = [nodes[i][1].embedding_key for i in top_indices]
    top_similarities = [similarities[i] for i in top_indices]

    # Print the top sentences and their similarity scores
    for i, (sentence, score) in enumerate(zip(top_sentences, top_similarities)):
        print(f"Rank {i+1}: {sentence} (Similarity: {score})")'''

    '''import spacy

# Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

# Example sentence
    sentence = "Isabella Rodriguez is helping customers with their orders"

# Process the sentence with spaCy
    doc = nlp(sentence)

# Extract named entities
    keywords = [ent.text for ent in doc.ents]

    print("Keywords:", keywords)

    import nltk
    from rake_nltk import Rake

    # Download the punkt tokenizer if it's not already installed
    #nltk.download('punkt')

    # Initialize RAKE by providing a list of stopwords
    rake = Rake()

    # Extract keywords
    rake.extract_keywords_from_text(sentence)
    keywords = rake.get_ranked_phrases()

    print("Keywords:", keywords)

    from keybert import KeyBERT

    # Initialize KeyBERT
    kw_model = KeyBERT()


    # Extract keywords
    keywords = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 2), stop_words='english')

    print("Keywords:", keywords)




    class BARTEmbeddings:
        def __init__(self, model_name='facebook/bart-large'):
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartModel.from_pretrained(model_name)
        
        def __call__(self, text_list):
            # Tokenize and encode the text
            encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Use the last hidden state of the [CLS] token as the embedding
            embeddings = model_output.last_hidden_state[:, 0, :]
            return embeddings.numpy()

    # Initialize KeyBERT with BART embeddings
    bart_embeddings = BARTEmbeddings()
    kw_model = KeyBERT(model=bart_embeddings)

    # Extract keywords
    keywords = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 2), stop_words='english')

    print("Keywords:", keywords)'''

    '''gen_agents_mem_retreive = []
    bert_mem_retreive = []
    bart_mem_retreive = []
    main_theme = "party"

    nodes = [i for created, i in nodes]
    gen_agents_theme_embedding = get_embedding(main_theme)
    bert_theme_embedding = calculate_embedding(main_theme, bert_tokenizer, bert_model)
    bart_theme_embedding = calculate_embedding(main_theme, bart_tokenizer, bart_model)

    for count, node in enumerate(nodes): 
      if (cos_sim(gen_agents_theme_embedding, self.agents[self.agents_names[0]].a_mem.embeddings[node.embedding_key]) > 0.75):
        gen_agents_mem_retreive.append(node.embedding_key)

      bert_similarity = cosine_similarity(bert_theme_embedding, calculate_embedding(node.embedding_key, bert_tokenizer, bert_model))
      if (bert_similarity[0][0] > 0.75):
        bert_mem_retreive.append(node.embedding_key)

      bart_similarity = cosine_similarity(bart_theme_embedding, calculate_embedding(node.embedding_key, bart_tokenizer, bart_model))
      if (bart_similarity[0][0] > 0.75):
        bart_mem_retreive.append(node.embedding_key)

    print(len(gen_agents_mem_retreive))
    print(len(bert_mem_retreive))
    print(len(bart_mem_retreive))'''

    '''relevance_out = dict()
    for count, node in enumerate(nodes): 
        node_embedding = persona.a_mem.embeddings[node.embedding_key]
        relevance_out[node.node_id] = cos_sim(node_embedding, focal_embedding)'''

    '''for agent_name in self.agents_names:
      self.agents[agent_name].add_convo_channel(convo)

    for i in range(5):
      for agent_name in self.agents_names:
        print(agent_name)
        self.agents[agent_name].execute_fsm()
        print("\n")
      director.execute_fsm()
      print("\n")'''


  

    orchestrator = Orchestrator("", False)

    free_agents = list(self.agents.keys())
    convo_agents = []
    convo_users = []

    '''from transformers import pipeline



    neuroticism = 0.4
    extraversion = 0.3
    agreeableness = 0.5
    openness = 0.2


    interaction_matrix = {
                          "joy": {"joy": 0.9, "sadness": -0.4, "anger": -0.4, "fear": -0.4},
                          "sadness": {"joy": -0.5, "sadness": 0.9, "anger": 0.4, "fear": 0.4},
                          "anger": {"joy": -0.5, "sadness": 0.4, "anger": 0.9, "fear": 0.2},
                          "fear": {"joy": -0.4, "sadness": 0.3, "anger": 0.2, "fear": 0.9},
                          "love": {"joy": 0.8, "sadness": 0.4, "anger": -0.5, "fear": -0.4},
                          "surprise": {"joy": 0.7, "sadness": 0.2, "anger": 0.3, "fear": 0.6}
                        }
    
    personality_coef = dict()
    personality_coef["joy"] =  extraversion*0.4 + agreeableness*0.4 - neuroticism*0.2
    personality_coef["sadness"] = neuroticism*0.6 + agreeableness*0.4
    personality_coef["anger"] = neuroticism
    personality_coef["fear"] =  neuroticism*0.8 - openness*0.2

    stimulus_coef = dict()

    for stimulus, emotion_convo_coef in interaction_matrix.items():
      stimulus_coef[stimulus] = dict()
      for emotion, convo_coef in emotion_convo_coef.items():
        stimulus_coef[stimulus][emotion] = convo_coef * personality_coef[emotion]


    print(stimulus_coef)
    
    emotions = dict()
    emotions["joy"] =  0.5
    emotions["sadness"] = 0.5
    emotions["anger"] = 0.5
    emotions["fear"] = 0.5

    import numpy as np
    import matplotlib.pyplot as plt

    # Define the data

    labels = list(emotions.keys())
    values = list(emotions.values())
    def _radar_diagram(self, labels, values):

      # Since the radar chart needs a closed shape, we need to "close the loop"
      values += values[:1]
      num_vars = len(labels)

      # Compute angle of each axis
      angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
      angles += angles[:1]

      # Plot
      fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

      ax.fill(angles, values, color='blue', alpha=0.25)
      ax.plot(angles, values, color='blue', linewidth=2)

      ax.set_ylim(0, 1)

      # Add labels
      ax.set_yticklabels([])
      ax.set_xticks(angles[:-1])
      ax.set_xticklabels(labels)

      plt.show()

    labels = list(emotions.keys())
    values = list(emotions.values())
   # _radar_diagram(self, labels, values)

   

    for emotion, val in emotions.items():
      print(emotion, val)

    # Crear un pipeline para clasificación de emociones con un modelo preentrenado
    classifier = pipeline('sentiment-analysis', model="bhadresh-savani/distilbert-base-uncased-emotion")
    #bhadresh-savani/distilbert-base-uncased-emotion
    #nateraw/bert-base-uncased-emotion

    # Mensaje recibido
    message = "There's an inexplicable warmth that's bubbling up inside me, as if my whole being is radiating this quiet, profound contentment that words can barely capture."
    emotion_results = classifier(message)
    print(emotion_results)  
    
    message = "It's as though the colors of the world have dulled, and even the things that once brought joy now feel distant, unreachable."
    emotion_results = classifier(message)
    print(emotion_results)  

    message = "My blood feels like it's boiling, and this overwhelming sense of injustice is so sharp that I can barely focus on anything else but the sheer audacity of what's happened."
    emotion_results = classifier(message)
    print(emotion_results)  

    message = "My mind feels scrambled, as if everything I thought I understood has been thrown into chaos by this unexpected revelation."
    emotion_results = classifier(message)
    print(emotion_results)  

    message = "It's as if the ground beneath me has shifted, leaving me momentarily untethered, and my mind is racing to catch up with this completely unexpected turn of events."
    emotion_results = classifier(message)
    print(emotion_results)  

    message = "There's a quiet intensity in the way I feel about you, a depth of emotion that's hard to articulate, but it's like every part of me is drawn to you in ways that go beyond simple affection."
    emotion_results = classifier(message)
    print(emotion_results)  

    
    
    
    
    
    # Ejemplo de salida: 'fear' 0.85'''

    # Ajuste según perfil de personalidad (ejemplo simplificado)
    #perfil = {"anger": {"fear": 0.7, "alegría": 0.3}}
    #ajuste = perfil["optimista"]
    #ajustado = {result['label']: result['score'] * ajuste.get(result['label'], 1) for result in emotion_results}

    #print(ajustado)  # Ejemplo de salida ajustada según perfil

    #Joy (Happiness) Surprise

    #Sadness

    #Anger

    #Fear

    #Disgust

# Neutral

    phrase = "Did you see the football game? It was one of the best games of the season."

    emotions = ["joy", "sadness", "anger", "fear", "love", "surprise"]
    #print(self.agents["Isabella Rodriguez"].estimate_emotion("Maria, I really appreciate your enthusiasm, but I'm feeling a bit frustrated right now. I know we've all put in a lot of effort, but it just feels like things are out of our control. I don't want to disappoint Isabella. Maybe we can focus on the essentials and keep it simple?", emotions))


    while True:
      sim_command = input("Enter command: ")
      command = sim_command.split()

      if sim_command.lower() in ["f", "fin", "finish", "save and finish"]:
        break

      elif command[0].lower() in ["e", "execute"]:
        
        for i in range(int(command[1])):
          for agent_name in convo_agents:
            print(agent_name)
            self.agents[agent_name].execute_fsm()
            print("\n")
          orchestrator.execute_sfsm()
          print("\n")

      elif sim_command.lower() in ["ra", "register agent"]:
        
        for index in range(len(free_agents)):
          print(index, free_agents[index])

        agent_index = input("Type the index of the agent that should join the conversation: ")
        agent_index = int(agent_index)
        if agent_index > -1 and agent_index < len(free_agents):
          agent_to_join = free_agents.pop(agent_index)
          orchestrator.register_agent(self.agents[agent_to_join])
          convo_agents.append(agent_to_join)

      elif sim_command.lower() in ["ea", "eliminate agent"]:
        
        for index in range(len(convo_agents)):
          print(index, convo_agents[index])

        agent_index = input("Type the index of the agent that should be eliminated from the conversation: ")
        agent_index = int(agent_index)
        if agent_index > -1 and agent_index < len(convo_agents):
          agent_to_leave = convo_agents.pop(agent_index)
          orchestrator.eliminate_agent(self.agents[agent_to_leave])
          free_agents.append(agent_to_leave)

      elif sim_command.lower() in ["ru", "register user"]: 
        user_name = input("Type the user name: ")
        convo_users.append(user_name)
        orchestrator.register_user(user_name)

      elif sim_command.lower() in ["eu", "eliminate user"]:
        
        for index in range(len(convo_users)):
          print(index, convo_users[index])

        user_index = input("Type the index of the user that should be eliminated from the conversation: ")
        user_index = int(user_index)
        if user_index > -1 and user_index < len(convo_users):
          user_to_leave = convo_users.pop(user_index)
          orchestrator.eliminate_user(user_to_leave)

      elif sim_command.lower() in ["cc", "current convo"]:
        for phrase in orchestrator.convo:
          print(phrase)

      elif sim_command.lower() in ["aee", "agent emotion evolution"]:
        
        for i in range(len(convo_agents)):
          print(i, convo_agents[i])
        
        agent_index = input("Type the index of the agent that you want to see: ")

        convo = self.agents[convo_agents[int(agent_index)]].buffer[orchestrator.orchestrator_id]["convo"]
        convo_em = self.agents[convo_agents[int(agent_index)]].buffer[orchestrator.orchestrator_id]["message_emotion"]
        emotions = self.agents[convo_agents[int(agent_index)]].buffer[orchestrator.orchestrator_id]["emotions"]

        for i in range(len(emotions)):
          print(convo[i])
          print(convo_em[i])
          print(emotions[i])
      elif sim_command.lower() in ["ec", "emotional convo"]:
        convo = self.agents[convo_agents[0]].buffer[orchestrator.orchestrator_id]["convo"]
        convo_em = self.agents[convo_agents[0]].buffer[orchestrator.orchestrator_id]["message_emotion"]
        for i in range(len(convo_em)):
          print(convo_em[i])
      elif sim_command.lower() in ["sc", "save convo"]:
        convo = self.agents[convo_agents[0]].buffer[orchestrator.orchestrator_id]["convo"]
        convo_em = self.agents[convo_agents[0]].buffer[orchestrator.orchestrator_id]["message_emotion"]
        
        csv_filename = 'convo/convo_emotions.csv'

        with open(csv_filename, mode='a', newline='', encoding='utf-8') as csv_file:
          writer = csv.writer(csv_file)
          for i in range(len(convo_em)):
            em_convo = f'"{convo_em[i][0]}"'
            writer.writerow([convo[i], em_convo])
      
    '''for i in range(40):
      print("epoch: ", i, "\n")
      for agent_name in self.agents.keys():
        
        print(agent_name)
        self.agents[agent_name].execute_fsm()
        print("\n")
      director.execute_fsm()
      print("\n")
    
    print("Convo After", i, "epochs")
    for message in director.convo:
      print(message)'''
      
    '''for agent_name in self.agents.keys():
        print(agent_name)
        for buffer in self.agents[agent_name].buffer.keys():
          for agent, description in self.agents[agent_name].buffer[buffer]["participants"].items():
            print(agent, ":", description)'''
    # Start New Convo

    # Continue Convo

    # End Convo

    # Add a New Agent to the Convo

if __name__ == '__main__':
  
  warnings.filterwarnings("ignore", message=".*beta.*renamed.*bias.*")
  warnings.filterwarnings("ignore", message=".*gamma.*renamed.*weight.*")
  cs = ConferenceServer("July1_the_ville_isabella_maria_klaus-step-3-21", 
                     "s13")
  
  cs.open_confenrence()
  #rs = ReverieServer("base_the_ville_isabella_maria_klaus", 
  #                    "July1_the_ville_isabella_maria_klaus-step-3-1")
  # rs = ReverieServer("July1_the_ville_isabella_maria_klaus-step-3-20", 
  #                    "July1_the_ville_isabella_maria_klaus-step-3-21")
  # rs.open_server()


  # origin = input("Enter the name of the forked simulation: ").strip()
  # target = input("Enter the name of the new simulation: ").strip()

  # rs = ReverieServer(origin, target)

  
  #agent_chat_v4(rs.agents, 5)
  
  #rs.open_server()
