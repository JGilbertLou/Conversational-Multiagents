"""
Author: José Loucel

File: fsm.py
Description: This defines the fsm structure for the conversational modules. 
"""

class FSM:
    def __init__(self):
        self.active_state = None
        
    def set_state(self, state):
        self.active_state = state

    def update(self):
        if self.active_state is not None:
            self.active_state()