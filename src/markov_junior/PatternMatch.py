"""
Author: Mehmet Kayra Oguz and ChatGPT (minimal)
Date: July 10, 2023
"""

from functools import lru_cache
from numba import jit
import numpy as np
import math
import random
import cv2

def matchPattern(pattern, background, wildcard='*'):
    """
    Finds matching patterns on a background, returns the upper left coordinates in form
    (x,y) of each found pattern

    :param np.array pattern:        to be searched pattern
    :param np.array background:     searching environment
    :param char wildcard:           the wildcard ascii character

    :return match coordinates in (x,y) tuples
    """

    # Create a mask array to ignore wildcard values
    mask = np.where(pattern != ord(wildcard), True, False).astype(np.uint8)

    # Apply matchTemplate to the encoded arrays
    result = cv2.matchTemplate(background, pattern, cv2.TM_SQDIFF, mask=mask)

    # Get only the near perfect matches
    matches = np.where(result <= 0.0001)

    # Convert perfect machtes to coordinates
    matches = zip(matches[1], matches[0])

    # Get only the ones inbounds of the background
    inBounds = []
    for x, y in matches:
        if not isRectangleOutOfBounds(x, y, len(pattern[0]), len(pattern), background):
            inBounds.insert(0,(x, y))
    matches = inBounds
    return matches

def replacePattern(seed, rule, background, overlap=True, count=math.inf, wildcard='*', allowMatchShuffle=False):
    """
    Replaces matched patterns and returns the new background

    :param np.array pattern:        to be searched pattern
    :param np.array replacement:    to be replaced pattern
    :param np.array background:     searching environment
    :param bool overlap:            allow overlaps when replacing
    :param int count:               how many occurrences must be replaced
    :param char wildcard:           the wildcard ascii character

    :returns updated background as np.array and true if a replacement happened
    """

    pattern = rule.pattern
    replacement = rule.replacement

    matches = matchPattern(pattern, background)

    if len(matches) > 0:
        if allowMatchShuffle:
            random.Random(seed).shuffle(matches)
    else:
        return (background, False)

    if count > len(matches):
        count = len(matches)

    replaced = False
    for i in range(count):
        x, y = matches[i]
        window = background[y:y+len(pattern), x:x+len(pattern[0])]

        # If pattern overlap is not allowed, current window must be a valid match
        if overlap or comparePatterns(pattern, window):
            # Replace the matched pattern except the places with the wild cards
            mask = np.where(replacement != ord(wildcard), True, False)
            background[y:y+len(pattern), x:x+len(pattern[0])][mask] = replacement[mask]
            replaced = True

    return (background, replaced)


@jit(nopython=True)
def strToNpArray(pattern: str):
    """
    Converts 2d string to 2d char number np.array, helper method
    """
    result = []
    for row in pattern.strip().split('\n'):
        temp = []
        for element in row.strip():
            temp.append(ord(element))
        result.append(temp)
    return np.array(result, dtype='uint8')


@jit(nopython=True)
def npArrayToString(pattern):
    """
    Converts 2d np.array to 2d string, helper method 
    """
    result = ""
    for index, row in enumerate(pattern):
        for col in row:
            result+= "".join(chr(col))
        if index < len(pattern)-1:
            result += "\n"
    return result


@jit(nopython=True)
def comparePatterns(template, window, wildcard='*'):
    """
    Checks if two patterns are equivalent 
    """
    # Replace wildcard values if any in template with corresponding values in background
    template = np.where(template == ord(wildcard), window, template)
    return np.array_equal(template, window)


@jit(nopython=True)
def isRectangleOutOfBounds(x, y, width, height, background):
    """
    Checks if given rectangle is outside of background
    """
    if x < 0 or y < 0 or x + width > len(background[0]) or y + height > len(background):
        return True
    return False
