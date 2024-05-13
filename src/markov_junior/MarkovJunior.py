"""
Author: Mehmet Kayra Oguz
Date: July 10, 2023
"""

from . import PatternMatch as pm
import numpy as np
import random
import time


class Context():
    """
    A context has an environment where a defined grammar can be applied
    """

    def __init__(self, environment: str = "", grammar: list=None, allowMatchShuffle:bool=True, seed:int=0):
        self.seed = seed
        self.allowMatchShuffle = allowMatchShuffle

        # Environment is a 2d String
        self.environment = environment

        # Grammar includes sequences or markov rule sets. It has left to right hiearchy.
        if grammar == None:
            self.grammar = []
        else:
            self.grammar = grammar
            
        # Result
        self.result = None

    def applyGrammar(self) -> str:
        """
        Apply the grammar on the environment, returns the result as string
        """
    
        environment = pm.strToNpArray(self.environment)
    
        for index, item in enumerate(self.grammar):

            if isinstance(item, MultiRule) or isinstance(item, Rule):
                environment, applyable, self.seed = item.applyRule(environment, self.seed+1, self.allowMatchShuffle)
                item.applyable = applyable
                
            elif isinstance(item, Markov) or isinstance(item, Sequence):
                environment, applyable, self.seed = item.applyRuleSet(environment, self.seed+1, self.allowMatchShuffle)
                item.applyable = applyable

        if isinstance(environment, np.ndarray):
            self.result = pm.npArrayToString(environment)
        elif isinstance(environment, str):
            self.result = environment
        else:
            raise Exception("Produced result is neither an instance of str nor np.ndarray")
        
        return self.result

    def addToGrammar(self, item):
        self.grammar.append(item)


class Rule():
    """
    Fundamental rule class, a rule consists of an input(pattern) and an output(replacement).
    A rule is applied on an environment (background) on the possible areas in the environment
    """

    def __init__(self, pattern: str, replacement: str, rotation=0):
        if rotation == None:
            self.rotation = 0
        else:
            self.rotation = int(rotation)

        if isinstance(pattern, str):
            self.pattern = np.rot90(pm.strToNpArray(pattern), self.rotation)
        else:
            self.pattern = np.rot90((pattern), self.rotation)

        if isinstance(replacement, str):
            self.replacement = np.rot90(
                pm.strToNpArray(replacement), self.rotation)
        else:
            self.replacement = np.rot90((replacement), self.rotation)
            
        self.applyable = False


class MultiRule():
    """
    Secondary rule class, with multiple input(pattern) and output(replacement) tuples (rules).
    A rule is applied on an environment (background) on the possible areas in the environment.
    MultiRole has the advantage of delvering a rule from its rule list to be applied
    """

    def __init__(self, rules=None, rulesRandom=0, terminating: bool=False):
        """
        If rules random is true only one random rule from the rules
        will be applied, else all the rules will be applied if possible
        """
        if rules == None:
            self.rules = []
        else:
            self.rules = rules

        if rulesRandom == None:
            self.rulesRandom = False
        else:
            self.rulesRandom = int(rulesRandom)

        self.terminating = terminating
        
        self.ruleWeights = []

    def addRule(self, rule: Rule, ruleWeight=1):
        self.rules.append(rule)
        self.ruleWeights.append(ruleWeight)

    def applyRule(self, environment: np.ndarray, currSeed: int, matchShuffle: bool):
        print("Applying rule")


class OneMulti(MultiRule):
    """
    First pattern occuerence is replaced
    """

    def applyRule(self, environment: np.ndarray, currSeed: int, matchShuffle: bool):
        """
        Apply rule to the first occuerence/s
        """

        newEnv = None
        success = False
        currSeed +=1
        
        if self.rulesRandom:
            random.Random(currSeed).shuffle(self.rules)

        for rule in self.rules:
            newEnv, success = pm.replacePattern(currSeed, rule, background=environment, count=1, allowMatchShuffle=matchShuffle)
            if success:
                rule.applyable = True
            if success and self.rulesRandom:
                return (newEnv, success, currSeed)
            
        return (newEnv, success, currSeed)


class AllMulti(MultiRule):
    """
    All pattern occuerences are replaced without overlapping
    """

    def applyRule(self, environment: np.ndarray, currSeed: int, matchShuffle: bool):
        """
        Apply rule to all occuerences without overlapping
        """
      
        background = None
        success = False
        currSeed +=1
        
        if self.rulesRandom:
            random.Random(currSeed).shuffle(self.rules)

        for rule in self.rules:
            background, success = pm.replacePattern(currSeed, rule, background=environment, allowMatchShuffle=matchShuffle, overlap=False)
            if success:
                rule.applyable = True
            if success and self.rulesRandom:
                return (background, success, currSeed)
        
        return (background, success, currSeed)


class PrlMulti(MultiRule):
    """
    All pattern occuerences are replaced by not caring about overlaps
    """

    def applyRule(self, environment: np.ndarray, currSeed: int, matchShuffle: bool):
        """
        Apply rule to all occuerences with overlapping
        """

        background = None
        success = False
        currSeed += 1
        
        if self.rulesRandom:
            random.Random(currSeed).shuffle(self.rules)

        for rule in self.rules:
            background, success = pm.replacePattern(currSeed, rule, background=environment, overlap=True, allowMatchShuffle=matchShuffle)
            if success:
                rule.applyable = True
            if success and self.rulesRandom:
                return (background, success, currSeed)
     
        return (background, success, currSeed)


class RuleSet():
    """
    A rule set may include other rulesets and/or single rules
    """

    def __init__(self, items=None):
        if items == None:
            self.items = []
        else:
            self.items = items

        self.applyable = False
    
    def applyRuleSet(self, environment: np.ndarray, currSeed: int, matchShuffle: bool):
        print("Appyling ruleset")


class Sequence(RuleSet):
    """
    Sequence applies rules or markov rule sets one time with hiearichal order
    left to right
    """

    def __init__(self, items=None, loop=1):
        if items == None:
            self.items = []
        else:
            self.items = items

        if loop == None:
            self.loop = 1
        else:
            self.loop = int(loop)
        
        self.applyable = True

    def applyRuleSet(self, environment: np.ndarray, currSeed: int, matchShuffle: bool):
        self.applyable = True 
        success = False
        #currSeed +=1
        for i in range(self.loop):
            for item in self.items:
                if isinstance(item, MultiRule) or isinstance(item, Rule):
                    environment, success, currSeed = item.applyRule(environment, currSeed+1, matchShuffle)
                elif isinstance(item, RuleSet):
                    environment, success, currSeed = item.applyRuleSet(environment, currSeed+1, matchShuffle)
                if not success:
                    self.applyable = False
                    
        return environment, success, currSeed
    
    def addItem(self, item):
        self.items.append(item)


class Markov(RuleSet):
    """
    Markov applies rules or rulesets using markov algorithm
    """

    def applyRuleSet(self, environment: np.ndarray, currSeed: int, matchShuffle: bool):
        #currSeed +=1
        environmentHistory = []
        
        initEnvironment = environment.copy()
        prevEnvironment = environment.copy()  # Store the previous environment state
        
        environmentHistory.append((initEnvironment.copy(), 1))
        
        iterations = 0
        self.applyable = True
        
        while True:
            iterations += 1
            terminate = True
            success = False
            
            prevEnvironment = environment.copy()
            
            for item in self.items:
                if isinstance(item, MultiRule):
                    environment, success, currSeed = item.applyRule(environment.copy(), currSeed+1, matchShuffle)
                    if item.terminating:
                        terminate = True
                        break
                elif isinstance(item, RuleSet):
                    environment, success, currSeed = item.applyRuleSet(environment.copy(), currSeed+1, matchShuffle)
                    #print(pm.npArrayToString(environment))
                if success: # If at least one successful attempt do not terminate markov but break 
                    terminate = False
                    break
        
            included = False
            # If the same environment comes up more than 2 times than markov is invalid
            for index in range(len(environmentHistory)):
                env, occurence = environmentHistory[index] 
                
                # If markov is invalid then immediatelly get out and mark non applyable
                if occurence > 3:
                    self.applyable = False     
                    return environment, self.applyable, currSeed
                
                # Update environment history
                if np.array_equal(environment, env):
                    environmentHistory[index] = (env, occurence+1)
                    included = True
                
            if not included:    
                environmentHistory.append((environment.copy(), 1))
                       
            # If no rule could be applied or too much iterations are executed stop or
            if (terminate or iterations >= 500):
                break

        # If the markov failed in the first iteration or too much iterations are executed 
        # or if we reached the initial environemnt then the markov is invalid
        if iterations == 1 or iterations >= 500 or np.array_equal(environment, initEnvironment):
            self.applyable = False

        return environment, self.applyable, currSeed

    def addItem(self, item):
        self.items.append(item)
