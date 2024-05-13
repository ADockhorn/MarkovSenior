"""
Author: Mehmet Kayra Oguz 
Date: July 10, 2023
"""

import sys
sys.path.append('..')

import markov_junior.Interpretor as intp
import markov_junior.PatternMatch as pm
import xml.etree.ElementTree as ET
import concurrent.futures
from numba import jit
from functools import lru_cache
import numpy as np
import math
import time
from src.markov_junior.PatternMatch import strToNpArray


@lru_cache(maxsize=None)
def kl_divergence(sample: str, generated:str, windowHeight:int=3, windowWidth:int=3, diversityFactor:int=0, withEnv=True)-> float:
    """
    Finds a KL-Divergence ratioScore between sample and generated contents
    sample: P, generated: Q
    """
    divergence = 0
    
    # Get unique windows from the sample and generated contents
    windowsSample = get_occurance_alphabet_from_sample(sample, windowHeight, windowWidth, withEnv)
    windowsGenerated = get_occurance_alphabet_from_sample(generated, windowHeight, windowWidth, withEnv)
    
    samplePatternProbs = process_alphabet(windowsSample, diversityFactor)
    generatedPatternProbs = process_alphabet(windowsGenerated, diversityFactor)

    for pattern, occProbSample in samplePatternProbs.items():
        if pattern in generatedPatternProbs:
            divergence += occProbSample * math.log(occProbSample / generatedPatternProbs[pattern])
        else: 
            divergence += occProbSample * math.log(occProbSample / sys.float_info.epsilon)
            
    return divergence



def process_alphabet(occuranceAlphabet:dict, diversityFactor:int=0):
    """
    Returns a dictionary of normalized pattern occurrence probabilities 'P(x)' values per pattern
    from occurrence alphabet
    """
    patternProbabilities = dict()
    totalOccurance = 0
    for pattern, occurance in occuranceAlphabet.items():
        totalOccurance += occurance
        
    # compute adjusted probabilities
    for pattern, occurance in occuranceAlphabet.items():
        patternProbabilities[pattern] = occ_prob_single_pattern(occurance, totalOccurance)

    # normalize the probabilities so they sum up to 1
    sum_of_probabilities = sum(patternProbabilities.values())
    for pattern in patternProbabilities:
        patternProbabilities[pattern] /= sum_of_probabilities
    
    return patternProbabilities
    
@jit(nopython=True)
def occ_prob_single_pattern(patternOccurance: int, totalOccurance:int, epsilon:float=sys.float_info.epsilon)-> float:
    """
    This calculates estimated probability of a single pattern occurrence
    P(x) or Q(x) from the KL-Divergence
    """
    return (patternOccurance + epsilon) / ((totalOccurance + epsilon) * (1 + epsilon))
    

def increase_diversity(patternProbabilities: dict, ratio: float, envAlphabet:dict={}) -> dict:
    """
    Adjusts the probabilities in the dict based on the given ratio between 0 and 1,
    The closer the ratio is to 1, the more probabilities close to 0 are boosted,
    while probabilities close to 1 are not affected.
    """
    if ratio > 0:
        for pattern, prob in patternProbabilities.items():
            # Avoid changing probabilities of the patterns that also are a part of the environment
            if not alphabet_includes(pm.strToNpArray(pattern), envAlphabet):
                adjustedProb = prob * math.exp(ratio * (1 - prob))
                patternProbabilities[pattern] = adjustedProb

    return patternProbabilities

def mirror_dataset(patternProbabilities: dict, ratio: float, envAlphabet:dict={}) -> dict:
    """
    Mirrors the probabilites in the dict based on the given ratio between 0 and 1,
    0 makes no differnce, 0.5 makes all probabilities almost equal, 1 flips all probabilities
    """
    if ratio > 0:
        for pattern, prob in patternProbabilities.items():
            # Avoid changing probabilities of the patterns that also are a part of the environment
            if not np.isin(pattern, envAlphabet).any():
                mirroredProb = (1 - prob) * ratio + prob * (1 - ratio)
                if mirroredProb <= 0:
                    print(f"ZERO OR SMALLAR: {prob},{mirroredProb}")
                    patternProbabilities[pattern] = sys.float_info.epsilon
                else:
                    patternProbabilities[pattern] = (mirroredProb)
            
    return patternProbabilities


def grid_frequency_score(occuranceAlphabetSample:dict, occuranceAlphabetOutput:dict, div_factor: float=0.0):
    """
    Returns a score between 0 and 1 for how diverse the output is relative to diversity score
    """
    score = 0
    pattern_counter = 0
    succesfull_placements = 0
    coherency_score =0
    
    # Find the pattern with highest occurrence
    max_occurance_sample = max(occuranceAlphabetSample.values())
                                                
    for pattern, occurance in occuranceAlphabetSample.items():
        if pattern not in ['X', '|', '-']:
            pattern_counter +=  1
            
            #normalized_occurance_sample = occurance / max_occurance_sample

            if pattern in occuranceAlphabetOutput:
                div_ratio = occuranceAlphabetOutput[pattern] / (occurance + occurance * div_factor)
                max_ratio = occuranceAlphabetOutput[pattern] / max_occurance_sample
                normal_ratio = occuranceAlphabetOutput[pattern] / occurance

                coherent = isinstance(occuranceAlphabetOutput[pattern] / occurance, int) or isinstance(occurance / occuranceAlphabetOutput[pattern], int)
                oddNumRelease = abs(occuranceAlphabetOutput[pattern] - occurance) <= 1 and occurance % 2 != 0
                
                # Check if coherent and allow some defects
                if  coherent or oddNumRelease  or  occurance  == 1:
                    score += 0.1
                    if div_ratio == 1: # High diversity
                        score += div_ratio
                        coherency_score += 1
                    elif max_ratio == 1: # At least as much as most occuring grid from the sample
                        score += max_ratio
                        coherency_score += 1
                    elif normal_ratio >= 0.5 and normal_ratio <= 1.5 :  # If not close to it's own original occurence
                        score += 0.1
                        coherency_score += 1
                    else:
                        score -= 1
                else:
                    score -= 0.5
    if sum(occuranceAlphabetSample.values()) < sum(occuranceAlphabetOutput.values()):
        score += 1
    
    score += coherency_score / pattern_counter
    
    # Normalize score, handling potential negative score
    score = min(score / pattern_counter, 1)
    if score < 0:
        return 0
    return score

def extract_relations(environment: str, alphabet: list, windowSize:int=4, onlyNeighbours: bool=False)-> dict:
    """
    Finds contextual and positional relations between patterns
    """
    environment = pm.strToNpArray(environment)

    results = []

    # For each pattern combinations find distance and positional relations
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for indexOrigin, origin in enumerate(alphabet):
            for indexTarget, target in enumerate(alphabet):
                results.append(executor.submit(get_relation, indexOrigin, origin, indexTarget, target, environment, windowSize, onlyNeighbours))

    relations = dict()
    # Gather the results from the thread pool
    for result in concurrent.futures.as_completed(results):
        indexOrigin, indexTarget, distRel, posRel, ratio = result.result()
        if distRel and posRel:
            relations[(indexOrigin, indexTarget)] = (distRel, posRel, ratio)

    return relations


def get_relation(indexOrigin, origin, indexTarget, target, environment, windowSize, onlyNeighbours: bool):
    """
    Thread function wrapper
    """
    distRel, posRel, ratio = extract_dist_pos_relations(origin, target, environment, windowSize, onlyNeighbours)
    return (indexOrigin, indexTarget, distRel, posRel, ratio)


def extract_dist_pos_relations(originPattern, targetPattern, environment, windowSize:int=4, onlyNeighbours: bool= False):
    """
    Extract for an origin pattern how far away the closest target pattern may be
    and extract for an origin pattern where the closest target pattern may be

    Returns a tuple of dictionaries of vectors and distances between the two 
    closest origin and target patterns and their frequencies 
    """
    
    def is_neighbour(origin, target):
        return euclidean_dist(origin, target) == windowSize
               
    # Get the match coordinates of both pattern on the environment
    origins = pm.matchPattern(originPattern, environment)
    targets = pm.matchPattern(targetPattern, environment)

    vectors = []
    distances = []
    
    if len(targets) == 0 or len(origins) == 0:
        return (None, None, 0)
    
    ratio = len(targets) / len(origins)
    
    for origin in origins:
        target = min(targets, key=lambda coord: euclidean_dist(origin, coord))
        closestTargets = [coord for coord in targets if euclidean_dist(origin, target) == euclidean_dist(origin, coord)]
        for target in closestTargets:
            if onlyNeighbours:
                if is_neighbour(origin, target):
                    vectors.append(vectoral_diff(origin, target))
                    distances.append(euclidean_dist(origin, target))
            else:
                vectors.append(vectoral_diff(origin, target))
                distances.append(euclidean_dist(origin, target))

    return (get_percentage_distribution(distances), get_percentage_distribution(vectors), ratio)


def get_alphabet_from_xml(path: str):
    """
    Get the alphabet as patterns and canvases as char number type np.arrays from 
    """
    context = ET.parse(path)
    alphabet = []
    cavnas = []
    for item in list(context.find('alphabet')):
        if item.tag == 'symbol':
            alphabet.append(pm.strToNpArray(
                intp.format_param(item.get("pattern"))))
        elif item.tag == "canvas":
            cavnas.append(pm.strToNpArray(
                intp.format_param(item.get("pattern"))))
    return alphabet

@lru_cache(maxsize=None)
def get_alphabet_from_sample(sample: str, windowHeight:int=1, windowWidth:int=1):
    """
    Get the alphabet as single elements from the sample
    """
    print("Extracting alphabet...")
    alphabet_set = set()
    for symbol in get_patterns(sample, windowHeight, windowWidth):
        alphabet_set.add(symbol)

    alphabet = []
    for symbol in sorted(alphabet_set):
        alphabet.append(pm.strToNpArray(symbol))
   
    print("Alphabet extracted from sample with size:", len(alphabet))
    return alphabet

@lru_cache(maxsize=None)
def get_occurance_alphabet_from_sample(sample: str, windowHeight:int=1, windowWidth:int=1, include_env:bool=False) -> dict:
    """
    Get the alphabet as single elements from the sample
    """
    
    alphabet= dict()
    for symbol in get_patterns(sample, windowHeight, windowWidth):
       
        if (not pattern_includes(symbol,"X") and not pattern_includes(symbol, "-")) or include_env: 
            if symbol in alphabet:
                alphabet[symbol] += 1
            else:
                alphabet[symbol] = 1

    #print("Alphabet with symbol  extracted from sample with size:", len(alphabet))
    return alphabet

def euclidean_dist(origin: tuple, target: tuple):
    """
    Returns the Euclidean distance between two coordinate tuples or single indices
    """
    if origin == target:
        return math.inf

    if len(origin) != len(target):
        raise ValueError("Coordinates must have the same number of dimensions")

    squared_distance = sum((a - b) ** 2 for a, b in zip(origin, target))
    return math.sqrt(squared_distance)


def vectoral_diff(origin, target):
    """
    Returns the relative position of the target from origin
    """
    return (target[0]-origin[0], target[1]-origin[1])
 
def get_percentage_distribution(items):
    """
    Returns the percentage distribution of each item in an iterable
    """
    total_count = len(items)
    distribution = dict()

    for item in items:
        if item in distribution:
            distribution[item] += 1
        else:
            distribution[item] = 1

    percentage_distribution = dict()

    for item, count in distribution.items():
        percentage = round((count / total_count), 2)
        percentage_distribution[item] = percentage

    return percentage_distribution

@lru_cache(maxsize=None)
def get_patterns(environment: str, windowHeight=3, windowWidth=3) -> list:
    """
    Returns windowHeight times windowWidth sized patterns from the image as set of strings
    """
    
    window = pm.strToNpArray(wildcard_window(windowHeight, windowWidth))
    environment = pm.strToNpArray(environment)
    
    matches = pm.matchPattern(window, environment)
    patterns = []
    for match in matches:
        x, y = match
        patterns.append(pm.npArrayToString(environment[y:y+windowHeight, x:x+windowWidth]))
    return patterns

@lru_cache(maxsize=None)
def wildcard_window(windowHeight=3, windowWidth=3, wildcard="*"):
    """
    Creates a window out of wildcards
    """
    
    window = ""
    for height in range(windowHeight):
        for width in range(windowWidth):
            window += wildcard
        window += "\n"
    return window

@lru_cache(maxsize=None)
def convert_to_env(pattern:str):
    result = ""
    for row in pattern.strip().split('\n'):
        for char in row.strip():
            if char == 'X' or char == '-':
                result += ((char))
            else:
                result += (('-'))
        result += "\n"
    return result

@lru_cache(maxsize=None)
def convert_air_to_wilcard(pattern:str, wildcard:str="*"):
    result = ""
    for row in pattern.strip().split('\n'):
        for char in row.strip():
            if char == '-':
                result += wildcard
            else:
                result += char
        result += "\n"
    return result


def alphabet_includes(pattern:np.ndarray, alphabet:list):
    for alph_pattern in alphabet:
        if np.array_equal(pattern, alph_pattern):
            return True
    return False

def pattern_includes(pattern, symbol:str="X"):
    if isinstance(pattern, str):
        pattern = pm.strToNpArray(pattern)
    return np.isin(ord(symbol), pattern)

def extract_rectangular_region(output, x, y, height, width):
    lines = output.split('\n')
    region_lines = lines[y : y + height]
    region_string = '\n'.join([line[x : x + width] for line in region_lines])
    return region_string