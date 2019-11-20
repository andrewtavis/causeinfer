  
"""
- The module contains causeinfer's algorithms

Inlcuded algorithmic methods:
1. Two Model Approach
2. Interaction Term Approach - Lo 2002
3. Response Transformation Approach - Lai 2006; Kane, Lo and Zheng 2014
4. Generalized Random Forest - Athey, Tibshirani, and Wager 2018
"""

from .two_model import TwoModel
from .interaction_term import InteractionTerm
from .response_transformation import ResponseTransformation
from .generalized_random_forest import GRF

from .evaluation import Evaluation