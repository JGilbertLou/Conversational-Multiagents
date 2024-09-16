"""
Author: José Loucel

File: stack_fsm.py
Description: This defines the fsm structure for the conversational modules. 
"""

class StackFSM:
    def __init__(self):
        self.stack = []

    def update(self):
        current_state_function = self.get_current_state()
        if current_state_function is not None:
            current_state_function()

    def pop_state(self):
        if len(self.stack) > 0:
            return self.stack.pop()
        return None

    def push_state(self, state):
        if self.get_current_state() != state:
            self.stack.append(state)

    def get_current_state(self):
        return self.stack[-1] if len(self.stack) > 0 else None


    