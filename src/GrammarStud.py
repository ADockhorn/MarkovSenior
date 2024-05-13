"""
Author: Mehmet Kayra Oguz 
Date: July 10, 2023
"""

import os
from markov_junior import MarkovJunior
from markov_junior import Interpretor as intp
from libraries.mario_level_visualizer import mlv
import Util as dp
from PIL import Image


def collectGrammars(grammarPaths:list):
    contexts = dict()
    for grammarPath in grammarPaths:
        contexts[grammarPath] = intp.parse_xml(grammarPath)
    return contexts
        
def applyGrammars(root_path, destination_path, grammarPaths:list):
    """
    Applies predefined grammars and outputs the result in the destination path
    """
    contexts =collectGrammars(grammarPaths)
    print(contexts)
    outputs = []
    if contexts:
        for path, context in contexts.items():
            output = context.applyGrammar()
            print(output)
            outputs.append(output)
            dp.save_file(output, "output"+os.path.basename(path)[-1:]+".txt")

    content = dp.combine_squares_into_level(outputs)
    mlv.world_string_to_png(content, root_path, destination_path)
    mlv.world_string_to_png(content, root_path, os.path.join(root_path, "user_generated.png"))
    return Image.open(destination_path)
    
