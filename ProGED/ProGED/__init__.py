# -*- coding: utf-8 -*-

from ProGED.ProGED.model import Model
from ProGED.ProGED.model_box import ModelBox, symbolic_difference
from ProGED.ProGED.generators.grammar import GeneratorGrammar
from ProGED.ProGED.generators.grammar_construction import grammar_from_template
from ProGED.ProGED.parameter_estimation import fit_models
from ProGED.ProGED.task import EDTask
from ProGED.ProGED.equation_discoverer import EqDisco

__version__ = 0.8