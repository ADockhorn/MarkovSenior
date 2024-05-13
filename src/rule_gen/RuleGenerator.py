"""
Author: Mehmet Kayra Oguz 
Date: July 10, 2023
"""

from markov_junior import Interpretor as intp
from markov_junior.MarkovJunior import *
from rule_gen import PatternParse as pp
from functools import lru_cache
import Util as dp
import random

@lru_cache(maxsize=None)
def generate_random_grammars_from_env_sample(environment: str, sample: str):
    """
    Generates random grammars
    ->Grammars mainly consist of Markovs and individual Ones (rare Alls)
    -->Markovs mainly consist of individual Ones (rare Alls and Sequences)
    --->Sequences mainly consist of individual Ones (rare Alls)
    """

    start_time = time.time()

    pairs = generate_in_out_pairs_from_env_sample(environment, sample)
    rules = generate_rules(pairs)
    ruleSetsAndRules = generate_rulesets_and_rules(rules)

    grammars = []

    while ruleSetsAndRules:
        sublistLength = random.randint(2, 10)
        sublist = ruleSetsAndRules[:sublistLength]
        grammars.append(sublist)
        ruleSetsAndRules = ruleSetsAndRules[sublistLength:]

    end_time = time.time()
    print("Random grammars generation took:",
          (end_time - start_time), "seconds")
    return grammars

@lru_cache(maxsize=None)
def generate_relation_based_random_grammars(sample, environment, windowHeight: int = 3, windowWidth: int = 3, population:int = 100, pairs=None):
    """
    Generates grammars with consecutive pattern and relation based rules (Puzzel Model)
    """
    print("Generating relation based grammars...")
    start_time = time.time()
    #pairs = generate_all_possible_pairs(sampleAlphabet, environmentAlphabet, windowHeight, windowWidth)
    rules = generate_rules(pairs)
    ruleSetsAndRules = generate_rulesets_and_rules(rules)

    grammars = []

    for i in range(population):
        if ruleSetsAndRules:
            if len(ruleSetsAndRules) == 1:
                sublist = ruleSetsAndRules.extend(ruleSetsAndRules)
                if sublist is not None:
                    grammars.append(sublist)
                break

            # Limit sublist length to the remaining rules
            sublistLength = random.randint(2, min(10, len(ruleSetsAndRules)))
            sublist = ruleSetsAndRules[:sublistLength]

            if sublist is not None:
                grammars.append(sublist)
                ruleSetsAndRules = ruleSetsAndRules[sublistLength:]

    for g in grammars:
        if g is None:
            print("EMPTY GRAMMAR PRODUCED")
            exit(1)

    end_time = time.time()
    print("Random relation based grammars generation took:",
          (end_time - start_time), "seconds")
    return grammars

@lru_cache(maxsize=None)
def generate_relation_based_in_out_pairs(sample, environment, windowHeight: int = 3, windowWidth: int = 3):
    """
    Generates relation based input and output pairs
    """

    pairs = []
    
    for windowSize in range(windowWidth, 0, -1):
        windowWidth = windowSize
        windowHeight = windowSize
        sampleAlphabet = pp.get_alphabet_from_sample(sample, windowHeight, windowWidth)
        environmentAlphabet = pp.get_alphabet_from_sample(environment, windowHeight, windowWidth)
        patternRelations = pp.extract_relations(sample, sampleAlphabet, windowSize,onlyNeighbours=True)

        air = pm.strToNpArray(pp.wildcard_window(windowHeight, windowWidth, "-"))
        
        for key, relation in patternRelations.items():
            distRel, posRel, ratio = relation
            originKey, targetKey = key

            # From origin to target 
            if not np.all(sampleAlphabet[originKey] == ord("-")) and not pp.alphabet_includes(sampleAlphabet[originKey], environmentAlphabet):
                originSamplePattern = pm.strToNpArray(pp.convert_air_to_wilcard(pm.npArrayToString(sampleAlphabet[originKey])))
            else:
                originSamplePattern = sampleAlphabet[originKey]
            
            if not np.all(sampleAlphabet[targetKey] == ord("-")) and not pp.alphabet_includes(sampleAlphabet[targetKey], environmentAlphabet):
                targetSamplePattern = pm.strToNpArray(pp.convert_air_to_wilcard(pm.npArrayToString(sampleAlphabet[targetKey])))
            else:
                targetSamplePattern = sampleAlphabet[targetKey]
            

            for position, probabilty in posRel.items():
                if not np.array_equal(targetSamplePattern, originSamplePattern):
                    # Initializers: anchors the initial patterns
                    if pp.pattern_includes(originSamplePattern, "X") or pp.pattern_includes(originSamplePattern, "|") :
                        #pairs.append(generate_directional_pair(originSamplePattern, originSamplePattern, originSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(originSamplePattern, originSamplePattern, targetSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(originSamplePattern, targetSamplePattern, originSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(originSamplePattern, targetSamplePattern, air, position, windowHeight, windowWidth))

                        for envPattern in environmentAlphabet:
                            if not np.array_equal(envPattern, originSamplePattern) and not np.array_equal(envPattern, targetSamplePattern):
                                #pairs.append(generate_directional_pair(originSamplePattern, targetSamplePattern, envPattern, position, windowHeight, windowWidth))
                                #pairs.append(generate_directional_pair(envPattern, targetSamplePattern, envPattern, position, windowHeight, windowWidth))
                                pairs.append(generate_directional_pair(originSamplePattern, targetSamplePattern, envPattern, position, windowHeight, windowWidth))
                                #pairs.append(generate_directional_pair(envPattern, targetSamplePattern, envPattern, position, windowHeight, windowWidth))
                                
                                #pairs.append(generate_directional_pair(envPattern, originSamplePattern, envPattern, position, windowHeight, windowWidth))
                                #pairs.append(generate_directional_pair(envPattern, originSamplePattern, originSamplePattern, position, windowHeight, windowWidth))
                                #pairs.append(generate_directional_pair(envPattern, originSamplePattern, targetSamplePattern, position, windowHeight, windowWidth))

                    # Successors: builds on anchors
                    if not pp.pattern_includes(targetSamplePattern, "|") and not pp.pattern_includes(targetSamplePattern, "X") :
                        #pairs.append(generate_directional_pair(targetSamplePattern, originSamplePattern, targetSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(targetSamplePattern, targetSamplePattern, originSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(targetSamplePattern, targetSamplePattern, targetSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(targetSamplePattern, originSamplePattern, originSamplePattern, position, windowHeight, windowWidth))
                    
                        pairs.append(generate_directional_pair(originSamplePattern, targetSamplePattern, air, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(originSamplePattern, targetSamplePattern, originSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(originSamplePattern, originSamplePattern, originSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(originSamplePattern, originSamplePattern, targetSamplePattern, position, windowHeight, windowWidth))
                        
                        #pairs.append(generate_directional_pair(air, originSamplePattern, targetSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(air, targetSamplePattern, originSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(air, targetSamplePattern, targetSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(air, originSamplePattern, originSamplePattern, position, windowHeight, windowWidth))
                    
                        #pairs.append(generate_directional_pair(air, targetSamplePattern, targetSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(air, targetSamplePattern, originSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(air, originSamplePattern, originSamplePattern, position, windowHeight, windowWidth))
                        #pairs.append(generate_directional_pair(air, originSamplePattern, targetSamplePattern, position, windowHeight, windowWidth))

    random.shuffle(pairs)
    return pairs

            
def generate_directional_pair(origin, target, envPattern, position, windowHeight: int = 3, windowWidth: int = 3):
    """
    Generates a single position related input and output pair
    """

    window = pm.strToNpArray(pp.wildcard_window(windowHeight, windowWidth))

    x, y = position
    pattern = None
    replacement = None
    if x < 0:   # West
        pattern = np.concatenate((envPattern, origin), axis=1)
        replacement = np.concatenate((target, window), axis=1)
    elif x > 0:  # East
        pattern = np.concatenate((origin, envPattern), axis=1)
        replacement = np.concatenate((window, target), axis=1)
    elif y < 0:  # South
        pattern = np.concatenate((envPattern, origin), axis=0)
        replacement = np.concatenate((target, window), axis=0)
    elif y > 0:  # North
        pattern = np.concatenate((origin, envPattern), axis=0)
        replacement = np.concatenate((window, target), axis=0)
    return (pattern, replacement)


def generate_directional_pairs(origin, target, envPattern, windowHeight: int = 3, windowWidth: int = 3):
    """
    Generates position related input and output pairs for each possible relative position
    """

    window = pm.strToNpArray(pp.wildcard_window(windowHeight, windowWidth))
    pairs = []
    pattern = None
    replacement = None
    # West
    pattern = np.concatenate((envPattern, origin), axis=1)
    replacement = np.concatenate((target, window), axis=1)
    pairs.append((pattern, replacement))
    # East
    pattern = np.concatenate((origin, envPattern), axis=1)
    replacement = np.concatenate((window, target), axis=1)
    pairs.append((pattern, replacement))
    # South
    pattern = np.concatenate((envPattern, origin), axis=0)
    replacement = np.concatenate((target, window), axis=0)
    pairs.append((pattern, replacement))
    # North
    pattern = np.concatenate((origin, envPattern), axis=0)
    replacement = np.concatenate((window, target), axis=0)
    pairs.append((pattern, replacement))

    return pairs


def generate_rules(pairs: list):
    """
    Generates rules from input output pairs with chances 10% All 90% One
    """

    rules = []
    for pair in pairs:
        if random.uniform(0, 1) < 0.1:
            rules.append(AllMulti([Rule(pair[0], pair[1])]))
        else:
            terminal = False
            rules.append(OneMulti([Rule(pair[0], pair[1])],terminating=terminal))
    return rules


def generate_rulesets_and_rules(rules: list):
    """
    Generates ruleSets from rules, returns a list of rules and ruleset mixtures
    """

    ruleSets = []

    while rules:
        # Get 5 rules make random mixture of Rulsesets and Rules
        sublistLength = random.randint(1, 5)
        sublist = rules[:sublistLength]
        rules = rules[sublistLength:]
        random.shuffle(sublist)

        # Create Rulesets with 95% chance
        if random.uniform(0, 1) < 0.95 and len(sublist) >= 2:
            # Nest random sequences in markov with 95% chance
            if random.uniform(0, 1) < 0.95 :
                ruleSets.append(Markov(sublist))
            else:
                markov = Markov()
                for sl in split_list(sublist, random.randint(1, len(sublist)-1)):
                    sequence = Sequence()
                    if isinstance(sl, list):
                        for item in sl:
                            sequence.addItem(item)
                    else:
                        sequence.addItem(sl)
                    markov.addItem(sequence)
                ruleSets.append(markov)
        else:
            for rule in sublist:
                ruleSets.append(rule)

    random.shuffle(ruleSets)
    return ruleSets

@lru_cache(maxsize=None)
def generate_in_out_pairs_from_env_sample(environment: str, sample: str, windowHeight: int = 3, windowWidth: int = 3):
    """
    For initial generation
    Generates every possible pairs from 3x3 windows from the sample and environment
    By using windows from environment for inputs and windows from sample for output  
    """
    samplePatterns = list(pp.get_patterns(sample, windowHeight, windowWidth))
    environmentPatterns = list(pp.get_patterns(
        environment, windowHeight, windowWidth))
    random.shuffle(samplePatterns)
    random.shuffle(environmentPatterns)
    pairs = []
    for envPattern in environmentPatterns:
        for samplePattern in samplePatterns:
            pairs.append((envPattern, samplePattern))
    print("Pairs generated from sample and environment")
    return pairs


def generate_in_out_pairs_patterns(patterns: list, replacements: list):
    """
    Generates every possible pairs from a list of patterns and replacements
    """
    random.shuffle(patterns)
    random.shuffle(replacements)
    pairs = []
    for replacement in replacements:
        for pattern in patterns:
            pairs.append((pattern, replacement))
    return pairs


def split_list(list_s, numOfSplits):
    """
    Splits a list to equal sized sub lists
    """
    
    length = len(list_s)
    if length == 1 and numOfSplits == 2:
        return [list_s[0], list_s[0]]
    
    sublist_length = length // numOfSplits
    if numOfSplits <= 1 or length == 0 or sublist_length == 0:
        return list_s
    sublists = []

    for i in range(0, length, sublist_length):
        sublists.append((list_s[i:i+sublist_length]))
    return sublists

def isPosRelHorizontal(position: tuple):
    x, y = tuple
    if x < 0:   # West
        return True
    elif x > 0:  # East
        return True
    elif y < 0:  # South
        return False
    elif y > 0:  # North
        return False
