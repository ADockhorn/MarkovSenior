"""
Author: Mehmet Kayra Oguz
Date: July 10, 2023
"""

import xml.etree.ElementTree as ET
import xml.dom.minidom
from . import MarkovJunior
from . import PatternMatch

def parse_grammar(context: MarkovJunior.Context, envPath="env.txt"):
    """
    Parses the context object grammar into the xml file 
    """
    context_xml = ET.Element("context")
    context_xml.set("seed", "0")
    context_xml.set("shuffle_matches", str(int(context.allowMatchShuffle)))

    environment_xml = ET.SubElement(context_xml, "environment")
    environment_xml.set("path", envPath)

    grammar_xml = ET.SubElement(context_xml, "grammar")

    for item in context.grammar:
        if isinstance(item, MarkovJunior.Sequence):
            grammar_xml.append(parse_sequence(item))
        elif isinstance(item, MarkovJunior.Markov):
            grammar_xml.append(parse_markov(item))
        elif isinstance(item, MarkovJunior.OneMulti):
            grammar_xml.append(parse_one(item))
        elif isinstance(item, MarkovJunior.AllMulti):
            grammar_xml.append(parse_all(item))
        elif isinstance(item, MarkovJunior.PrlMulti):
            grammar_xml.append(parse_prl(item))

    context_xml_str = xml.dom.minidom.parseString(ET.tostring(context_xml)).toprettyxml(indent="    ")
    #print(context_xml_str)
    #with open(file_path, 'w') as file:
    #    file.write(context_xml_str)
    return context_xml_str


def parse_one(one:MarkovJunior.OneMulti):
    one_xml = ET.Element("one")
    for rule in one.rules:
       rule_xml = ET.SubElement(one_xml, "rule")
       rule_xml.set("in", format_param(rule.pattern))
       rule_xml.set("out", format_param(rule.replacement))
    return one_xml


def parse_all(all:MarkovJunior.AllMulti):
    all_xml = ET.Element("all")
    for rule in all.rules:
       rule_xml = ET.SubElement(all_xml, "rule")
       rule_xml.set("in", format_param(rule.pattern))
       rule_xml.set("out", format_param(rule.replacement))
    return all_xml


def parse_prl(prl:MarkovJunior.PrlMulti):
    prl_xml = ET.Element("prl")
    for rule in prl.rules:
       rule_xml = ET.SubElement(prl_xml, "rule")
       rule_xml.set("in", format_param(rule.pattern))
       rule_xml.set("out", format_param(rule.replacement))
    return prl_xml


def parse_markov(markov:MarkovJunior.Markov):
    markov_xml = ET.Element("markov")
    for item in markov.items:
        if isinstance(item, MarkovJunior.Sequence):
            markov_xml.append(parse_sequence(item))
        elif isinstance(item, MarkovJunior.Markov):
            markov_xml.append(parse_markov(item))
        elif isinstance(item, MarkovJunior.OneMulti):
            markov_xml.append(parse_one(item))
        elif isinstance(item, MarkovJunior.AllMulti):
            markov_xml.append(parse_all(item))
        elif isinstance(item, MarkovJunior.PrlMulti):
            markov_xml.append(parse_prl(item)) 
    
    return markov_xml


def parse_sequence(sequence:MarkovJunior.Sequence):
    sequence_xml = ET.Element("sequence")
    for item in sequence.items:
        if isinstance(item, MarkovJunior.Sequence):
            sequence_xml.append(parse_sequence(item))
        elif isinstance(item, MarkovJunior.Markov):
            sequence_xml.append(parse_markov(item))
        elif isinstance(item, MarkovJunior.OneMulti):
            sequence_xml.append(parse_one(item))
        elif isinstance(item, MarkovJunior.AllMulti):
            sequence_xml.append(parse_all(item))
        elif isinstance(item, MarkovJunior.PrlMulti):
            sequence_xml.append(parse_prl(item)) 
            
    return sequence_xml


def format_param(param) -> str:
    param = PatternMatch.npArrayToString(param)
    return param.replace('\n', '/')