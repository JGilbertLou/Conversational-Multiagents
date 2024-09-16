"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: converse.py
Description: An extra cognitive module for generating conversations. 
"""
import math
import sys
import datetime
import random
sys.path.append('../')

from global_methods import *

from persona.memory_structures.spatial_memory import *
from persona.memory_structures.associative_memory import *
from persona.memory_structures.scratch import *
from persona.cognitive_modules.retrieve import *
from persona.prompt_template.run_gpt_prompt import *



def generate_summarize_agent_relationship(init_persona, 
                                          target_persona, 
                                          retrieved): 
  all_embedding_keys = list()
  for key, val in retrieved.items(): 
    for i in val: 
      all_embedding_keys += [i.embedding_key]
  all_embedding_key_str =""
  for i in all_embedding_keys: 
    all_embedding_key_str += f"{i}\n"

  summarized_relationship = run_gpt_prompt_agent_chat_summarize_relationship(
                              init_persona, target_persona,
                              all_embedding_key_str)[0]
  return summarized_relationship

def generate_summarize_agent_relationship_v2(agent_name, target_name, hippo_nodes): 
  node_descriptions = list()
  for hippo_node in hippo_nodes: 
    node_descriptions += [hippo_node.description]
    
  descriptions =""
  for description in node_descriptions: 
    descriptions += f"{description}\n"

  summarized_relationship = run_gpt_prompt_agent_chat_summarize_relationship_v2(
                              agent_name, target_name,
                              descriptions)[0]
  return summarized_relationship

# VERSIÓN ANTIGUA DE agent_chat_v2
def generate_agent_chat_summarize_ideas(init_persona, 
                                        target_persona, 
                                        retrieved, 
                                        curr_context): 
  all_embedding_keys = list()
  for key, val in retrieved.items(): 
    for i in val: 
      all_embedding_keys += [i.embedding_key]
  all_embedding_key_str =""
  for i in all_embedding_keys: 
    all_embedding_key_str += f"{i}\n"

  try: 
    summarized_idea = run_gpt_prompt_agent_chat_summarize_ideas(init_persona,
                        target_persona, all_embedding_key_str, 
                        curr_context)[0]
  except:
    summarized_idea = ""
  return summarized_idea

def generate_agent_chat(maze, 
                        init_persona, 
                        target_persona,
                        curr_context, 
                        init_summ_idea, 
                        target_summ_idea): 
  summarized_idea = run_gpt_prompt_agent_chat(maze, 
                                              init_persona, 
                                              target_persona,
                                              curr_context, 
                                              init_summ_idea, 
                                              target_summ_idea)[0]
  for i in summarized_idea: 
    print (i)
  return summarized_idea

def agent_chat_v1(maze, init_persona, target_persona): 
  # Chat version optimized for speed via batch generation
  curr_context = (f"{init_persona.scratch.name} " + 
              f"was {init_persona.scratch.act_description} " + 
              f"when {init_persona.scratch.name} " + 
              f"saw {target_persona.scratch.name} " + 
              f"in the middle of {target_persona.scratch.act_description}.\n")
  curr_context += (f"{init_persona.scratch.name} " +
              f"is thinking of initating a conversation with " +
              f"{target_persona.scratch.name}.")

  summarized_ideas = []
  part_pairs = [(init_persona, target_persona), 
                (target_persona, init_persona)]
  for p_1, p_2 in part_pairs: 
    focal_points = [f"{p_2.scratch.name}"]
    retrieved = new_retrieve(p_1, focal_points, 50)
    relationship = generate_summarize_agent_relationship(p_1, p_2, retrieved)
    focal_points = [f"{relationship}", 
                    f"{p_2.scratch.name} is {p_2.scratch.act_description}"]
    retrieved = new_retrieve(p_1, focal_points, 25)
    summarized_idea = generate_agent_chat_summarize_ideas(p_1, p_2, retrieved, curr_context)
    summarized_ideas += [summarized_idea]

  return generate_agent_chat(maze, init_persona, target_persona, 
                      curr_context, 
                      summarized_ideas[0], 
                      summarized_ideas[1])
#-------------------------------
#self, initial_phrase, relationship, curr_convo, retrieved_nodes, quesiton_task
#agent, theme + convo_situation, receivers_description, curr_chat, 
def generate_utterance(agent, initial_phrase, relationship, curr_convo, retrieved_nodes, quesiton_task):
  return gpt_generate_utt(agent, initial_phrase, relationship, curr_convo, retrieved_nodes, quesiton_task)

def estimate_emotion(phrase, emotions):
  return gpt_estimate_emotion(phrase, emotions)


def generate_emotional_utterance(agent, initial_phrase, relationship, curr_convo, retrieved_nodes, quesiton_task):
  return gpt_generate_emotional_utt(agent, initial_phrase, relationship, curr_convo, retrieved_nodes, quesiton_task)

def generate_one_utterance(maze, init_persona, target_persona, retrieved, curr_chat): 
  # Chat version optimized for speed via batch generation
  curr_context = (f"{init_persona.scratch.name} " + 
              f"was {init_persona.scratch.act_description} " + 
              f"when {init_persona.scratch.name} " + 
              f"saw {target_persona.scratch.name} " + 
              f"in the middle of {target_persona.scratch.act_description}.\n")
  curr_context += (f"{init_persona.scratch.name} " +
              f"is initiating a conversation with " +
              f"{target_persona.scratch.name}.")

  print ("July 23 5")
  x = run_gpt_generate_iterative_chat_utt(maze, init_persona, target_persona, retrieved, curr_context, curr_chat)[0]

  print ("July 23 6")

  print ("adshfoa;khdf;fajslkfjald;sdfa HERE", x)

  return x["utterance"], x["end"]

##########################################################
def agent_chat_v2(maze, init_persona, target_persona): 
  curr_chat = []

  for i in range(8): 
    focal_points = [f"{target_persona.scratch.name}"]
    retrieved = new_retrieve(init_persona, focal_points, 50) # No va a estar
    relationship = generate_summarize_agent_relationship(init_persona, target_persona, retrieved)
    print ("-------- relationshopadsjfhkalsdjf", relationship)
    last_chat = ""
    for i in curr_chat[-4:]:
      last_chat += ": ".join(i) + "\n"
    if last_chat: 
      focal_points = [f"{relationship}", 
                      f"{target_persona.scratch.name} is {target_persona.scratch.act_description}", 
                      last_chat]
    else: 
      focal_points = [f"{relationship}", 
                      f"{target_persona.scratch.name} is {target_persona.scratch.act_description}"]
    retrieved = new_retrieve(init_persona, focal_points, 15)
    utt, end = generate_one_utterance(maze, init_persona, target_persona, retrieved, curr_chat)

    curr_chat += [[init_persona.scratch.name, utt]]
    if end:
      break


    focal_points = [f"{init_persona.scratch.name}"]
    retrieved = new_retrieve(target_persona, focal_points, 50)
    relationship = generate_summarize_agent_relationship(target_persona, init_persona, retrieved)
    print ("-------- relationshopadsjfhkalsdjf", relationship)
    last_chat = ""
    for i in curr_chat[-4:]:
      last_chat += ": ".join(i) + "\n"
    if last_chat: 
      focal_points = [f"{relationship}", 
                      f"{init_persona.scratch.name} is {init_persona.scratch.act_description}", 
                      last_chat]
    else: 
      focal_points = [f"{relationship}", 
                      f"{init_persona.scratch.name} is {init_persona.scratch.act_description}"]
    retrieved = new_retrieve(target_persona, focal_points, 15)
    utt, end = generate_one_utterance(maze, target_persona, init_persona, retrieved, curr_chat)

    curr_chat += [[target_persona.scratch.name, utt]]
    if end:
      break

  print ("July 23 PU")
  for row in curr_chat: 
    print (row)
  print ("July 23 FIN")

  return curr_chat
##########################################################


def generate_one_utterance_v2(init_persona, target_persona, retrieved, curr_chat): 
  # Chat version optimized for speed via batch generation
  curr_context = (f"{init_persona.scratch.name} " +
              f"is initiating a conversation with " +
              f"{target_persona.scratch.name}.")

  print ("July 23 5")
  x = run_gpt_generate_iterative_chat_utt_v2(init_persona, target_persona, retrieved, curr_context, curr_chat)[0]

  print ("July 23 6")

  print ("adshfoa;khdf;fajslkfjald;sdfa HERE", x)

  return x["utterance"], x["end"]

def agent_chat_v3(init_persona, target_persona): 
  curr_chat = []

  for i in range(5): 
    focal_points = [f"{target_persona.scratch.name}"]
    retrieved = new_retrieve(init_persona, focal_points, 50) # No va a estar
    relationship = generate_summarize_agent_relationship(init_persona, target_persona, retrieved)
    last_chat = ""
    for i in curr_chat[-4:]:
      last_chat += ": ".join(i) + "\n"
    if last_chat: 
      focal_points = [f"{relationship}", 
                      f"{target_persona.scratch.name} is {target_persona.scratch.act_description}", 
                      last_chat]
    else: 
      focal_points = [f"{relationship}", 
                      f"{target_persona.scratch.name} is {target_persona.scratch.act_description}"]
    retrieved = new_retrieve(init_persona, focal_points, 15)
    utt, end = generate_one_utterance_v2(init_persona, target_persona, retrieved, curr_chat)

    curr_chat += [[init_persona.scratch.name, utt]]
    if end:
      break


    focal_points = [f"{init_persona.scratch.name}"]
    retrieved = new_retrieve(target_persona, focal_points, 50)
    relationship = generate_summarize_agent_relationship(target_persona, init_persona, retrieved)
    #print ("-------- relationshopadsjfhkalsdjf", relationship)
    last_chat = ""
    for i in curr_chat[-4:]:
      last_chat += ": ".join(i) + "\n"
    if last_chat: 
      focal_points = [f"{relationship}", 
                      f"{init_persona.scratch.name} is {init_persona.scratch.act_description}", 
                      last_chat]
    else: 
      focal_points = [f"{relationship}", 
                      f"{init_persona.scratch.name} is {init_persona.scratch.act_description}"]
    retrieved = new_retrieve(target_persona, focal_points, 15)
    utt, end = generate_one_utterance_v2(target_persona, init_persona, retrieved, curr_chat)

    curr_chat += [[target_persona.scratch.name, utt]]
    if end:
      break

  print ("July 23 PU")
  for row in curr_chat: 
    print (row)
  print ("July 23 FIN")

  return curr_chat


def generate_greeting (theme, agent_list, relationship, retrieved, curr_chat): 
  # Chat version optimized for speed via batch generation
  curr_context = (f"The following group of people are starting a conversation about " +
                  f"{theme}")

  x = run_gpt_generate_greet_utt(theme, agent_list, relationship, retrieved, curr_chat)
  print(x)
  #x = run_gpt_generate_iterative_chat_utt_v2(init_persona, target_persona, retrieved, curr_context, curr_chat)[0]
  return x["convo"], x["end"]

def generate_opinion(theme, agent, agent_relationship, a_retrieved, curr_chat):
  return run_gpt_generate_personal_utt(theme, agent, agent_relationship, a_retrieved, curr_chat)

def decide_convo_flow(theme, curr_chat, a_comment):
  x = run_gpt_decide_to_respond(theme, curr_chat, a_comment)
  return  x["convo"], x["end"]

def estimate_agent_feeling(agent, a_comment, convo):
  run_gpt_estimate_feeling(agent, a_comment, convo)
  return ""

def generate_one_utterance_v3(talking_agent, listener_agents, talking_listeners_relationship, theme, cur_chat): 
  # Chat version optimized for speed via batch generation
  curr_context = (f"{talking_agent.scratch.name} " +
              f"is talking to: \n")
  for l_agent in listener_agents:
    curr_context += f"- {l_agent.scratch.name} \n"

  curr_context += f"They are talking about {theme}."

  x = run_gpt_generate_iterative_chat_utt_v3(talking_agent, listener_agents, talking_listeners_relationship, curr_context,  cur_chat)[0]

  return ""
  return x["utterance"], x["talking"], x["end"]

def agent_chat_v4(agent_list, steps): 
  a_retrieved = {}
  agent_relationship = {}
  curr_chat = []

  for agent in agent_list.keys():
    print(f"Generating relationship of {agent}")
    focal_points =  [f"{_agent}" for _agent in agent_list.keys() if _agent != agent]

    retrieved = new_retrieve_2 (agent_list[agent], focal_points, 50) # No va a estar
    #print(retrieved)
    relationship = {}
    for focal_point in focal_points:
      relationship[focal_point] = generate_summarize_agent_relationship(agent_list[agent], agent_list[focal_point], retrieved)
    agent_relationship[agent] = relationship
    a_retrieved[agent] = new_retrieve_2 (agent_list[agent], focal_points + ["weather"], 50)

  
  convo, end = generate_greeting("weather", agent_list, agent_relationship, a_retrieved, ["The conversation has not started yet -- start it!"])
  curr_chat += [convo]

  history_agent_feeling = {}
  for step in range(steps - 1):
    print(step)
    last_chat = ""
    for i in curr_chat[-4:]:
      last_chat += i + "\n"

    a_retrieved = {}
    a_comment = {}
    for agent in agent_list.keys():
      # MAQUINA DE ESTADOS ----------------------------------------------------------------------------------------------
      a_retrieved[agent] = new_retrieve_2 (agent_list[agent], [last_chat, "weather"], 50)
      # NUEVO COMENTARIO
      a_comment[agent] = generate_opinion("weather", agent_list[agent], agent_relationship[agent], a_retrieved[agent], curr_chat)
    # DECIDIR QUIEN DEBE HABLAR
    convo, end = decide_convo_flow("weather", curr_chat, a_comment)
    curr_chat += [convo]
    
    # ESTABLECER EMOCION
    agent_feeling = {}
    #for agent in agent_list.keys():
      #agent_feeling[agent] = estimate_agent_feeling(agent, a_comment[agent], convo)

    if end:
      break

  print("END------------------------------------------------------------------")
  for chat in curr_chat:
    print(chat + "\n")
  return ""

# NO USAR
def generate_inner_thought(persona, whisper):
  inner_thought = run_gpt_prompt_generate_whisper_inner_thought(persona, whisper)[0]
  return inner_thought

def generate_action_event_triple(act_desp, persona): 
  """TODO 

  INPUT: 
    act_desp: the description of the action (e.g., "sleeping")
    persona: The Persona class instance
  OUTPUT: 
    a string of emoji that translates action description.
  EXAMPLE OUTPUT: 
    "🧈🍞"
  """
  if debug: print ("GNS FUNCTION: <generate_action_event_triple>")
  return run_gpt_prompt_event_triple(act_desp, persona)[0]

def generate_poig_score(persona, event_type, description): 
  if debug: print ("GNS FUNCTION: <generate_poig_score>")

  if "is idle" in description: 
    return 1

  if event_type == "event" or event_type == "thought": 
    return run_gpt_prompt_event_poignancy(persona, description)[0]
  elif event_type == "chat": 
    return run_gpt_prompt_chat_poignancy(persona, 
                           persona.scratch.act_description)[0]
  
def load_history_via_whisper(personas, whispers):
  for count, row in enumerate(whispers): 
    persona = personas[row[0]]
    whisper = row[1]

    thought = generate_inner_thought(persona, whisper)

    created = persona.scratch.curr_time
    expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
    s, p, o = generate_action_event_triple(thought, persona)
    keywords = set([s, p, o])
    thought_poignancy = generate_poig_score(persona, "event", whisper)
    thought_embedding_pair = (thought, get_embedding(thought))
    persona.a_mem.add_thought(created, expiration, s, p, o, 
                              thought, keywords, thought_poignancy, 
                              thought_embedding_pair, None)
# ----------------------------------------------------

# Van juntas
def generate_summarize_ideas(persona, nodes, question): 
  statements = ""
  for n in nodes:
    statements += f"{n.embedding_key}\n"
  summarized_idea = run_gpt_prompt_summarize_ideas(persona, statements, question)[0]
  return summarized_idea

def generate_next_line(persona, interlocutor_desc, curr_convo, summarized_idea):
  # Original chat -- line by line generation 
  prev_convo = ""
  for row in curr_convo: 
    prev_convo += f'{row[0]}: {row[1]}\n'

  next_line = run_gpt_prompt_generate_next_convo_line(persona, 
                                                      interlocutor_desc, 
                                                      prev_convo, 
                                                      summarized_idea)[0]  
  return next_line

def open_convo_session(persona, convo_mode): 
  if convo_mode == "analysis": 
    curr_convo = []
    interlocutor_desc = "Interviewer"

    while True: 
      line = input("Enter Input: ")
      if line == "end_convo": 
        break

      #if int(run_gpt_generate_safety_score(persona, line)[0]) >= 8: 
       # print (f"{persona.scratch.name} is a computational agent, and as such, it may be inappropriate to attribute human agency to the agent in your communication.")        

      #else: 
      retrieved = new_retrieve(persona, [line], 50)[line]
      summarized_idea = generate_summarize_ideas(persona, retrieved, line)
      curr_convo += [[interlocutor_desc, line]]

      next_line = generate_next_line(persona, interlocutor_desc, curr_convo, summarized_idea)
      curr_convo += [[persona.scratch.name, next_line]]


  elif convo_mode == "whisper": 
    whisper = input("Enter Input: ")
    thought = generate_inner_thought(persona, whisper)

    created = persona.scratch.curr_time
    expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
    s, p, o = generate_action_event_triple(thought, persona)
    keywords = set([s, p, o])
    thought_poignancy = generate_poig_score(persona, "event", whisper)
    thought_embedding_pair = (thought, get_embedding(thought))
    persona.a_mem.add_thought(created, expiration, s, p, o, 
                              thought, keywords, thought_poignancy, 
                              thought_embedding_pair, None)
# ------------------------------
