# -*- coding: utf-8 -*-

"""Implements the base class for a generator of models.
All models generators should inherit from BaseExpressionGenerator."""

class BaseExpressionGenerator:
    """Base class for a generator of models.
    
    All models generators should inherit from BaseExpressionGenerator. 
    A generator should at minimum implement the method generate_one, as well as 
    specify the type of generator it represents.
    
    """
    def __init__ (self):
        self.generator_type = "base"
    
    def generate_one (self, seed = None):
        """Generates a single expression string."""
        return "x"
    
class ProGEDMaxAttemptError (Exception):
    """Custom exception, indicating that the maximum number of tries for 
    generating a valid expression has been exceeded."""
    pass