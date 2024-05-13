"""
Author: Mehmet Kayra Oguz and ChatGPT
Date: July 10, 2023
Description: This is an interface for grammar training
"""

import argparse
import os
import time
import uuid
from rule_gen.GeneticRuleRefiner import *
from markov_junior import Interpretor as intp
from markov_junior import Decompiler as decom
from markov_junior.MarkovJunior import *
import Util as dp
import concurrent.futures


# GPT
def create_and_sort_refiners(samples, environment, novelty_factor, coherency_factor, population_size, mutation_rate, win_size, mid_win_size, mic_win_size, max_gram_len, diversity_factor, mac_win_size, tolerance_factor, shuffle_matches, levelload_conn):
    # Create a pool of processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        refiners = [None] * len(samples)  # Create a list with placeholders for each refiner
        gen = 0
        futures = {}
        for index, sample in enumerate(samples):
            future = executor.submit(GeneticRuleRefiner, sample, environment, novelty_factor, coherency_factor, population_size,
                                mutation_rate, win_size, win_size, mid_win_size, mid_win_size, mic_win_size, mic_win_size, max_gram_len, diversity_factor, mac_win_size, mac_win_size, index, tolerance_factor,shuffle_matches)
            futures[future] = index

            def callback(future):
                nonlocal gen
                gen += 1
                refiner = future.result()
                original_index = futures[future]
                refiners[original_index] = refiner  # Place each refiner in its original position

                if levelload_conn:
                    prog = round(((gen) / len(samples)) * 100)
                    data = {
                        "progress": prog, 
                        "currFinalOut": None, 
                        "end": False
                    }
                    levelload_conn.send(data)
                    
            future.add_done_callback(callback)
        
    return refiners

def gym_divcon(name, novelty_factor, coherency_factor, population_size, mutation_rate, win_size, mid_win_size, mic_win_size, max_gram_len, diversity_factor, mac_win_size, tolerance_factor, shuffle_matches=0, sample_path="../resources/lvl-1.txt", env_path="../resources/env.txt"):
    
    # Create sub directory for  divcon attempt
    main_path = create_dir("../outputs", "_divcon")
    
    print("\n######## Refinment by divide and conquer ############\n")
    arguments = arguments_str(name, novelty_factor, coherency_factor, population_size, mutation_rate, win_size, mid_win_size, mic_win_size, max_gram_len, diversity_factor, mac_win_size)
    print(arguments) 
    dp.save_file(arguments, os.path.join(main_path, "arguments.txt"))
    

    samples = dp.split_level_into_rectangles(sample_path, 16, 16)
    environment = intp.parse_environment(env_path)
    refiners = create_and_sort_refiners(samples, environment, novelty_factor, coherency_factor, population_size, mutation_rate, win_size, mid_win_size, mic_win_size, max_gram_len, diversity_factor, mac_win_size,tolerance_factor, shuffle_matches, None)
    results = []

    print(f"#############################################################")
    print(f"################# Refining now {len(refiners)} grammars ##################")
    print(f"#############################################################\n")
    
    for index, refiner in enumerate(refiners):
        # Create sub directory for refiner
        refiner_path = create_dir(main_path, "_index_"+str(index), unique_name=False)

        print(f"################ Refining {index} ###################")
        
        if refiner.sample == environment:
            results.append(refiner.sample)
            dp.save_file(refiner.sample, refiner_path +"/output_gen_0_fit_100"+".txt")
        else:
            xmlGrammar, output = refiner.refineGrammars(refiner_path)
            results.append(output)

        dp.save_file(dp.combine_squares_into_level(results),os.path.join(main_path, name+".txt"))
        
        print(f"########### Refining for {index} done ##############\n")
        
    print(f"########## Refinment done for {len(refiners)} grammars #############")
    print("Output is saved")

# GPT
def gym_divcon_gui(name, novelty_factor, coherency_factor, population_size, mutation_rate, win_size, mid_win_size, mic_win_size, max_gram_len, diversity_factor, mac_win_size, tolerance_factor, shuffle_matches, generation_conn=None, createpop_conn=None, levelload_conn=None, stash_path=None, sample_path="../resources/sample.txt", env_path="../resources/environment_small.txt"):
  
    print("\n######## Refinment by divide and conquer ############\n")
    arguments = arguments_str(name, novelty_factor, coherency_factor, population_size, mutation_rate, win_size, mid_win_size, mic_win_size, max_gram_len, diversity_factor, mac_win_size)
    print(arguments) 

    samples = dp.split_level_into_rectangles(sample_path, 16, 16)
    environment = intp.parse_environment(env_path)
    results = []
    
    if levelload_conn:
        data = {
            "progress": random.randint(1,4),
            "currFinalOut": None,
            "end": False
        }
        levelload_conn.send(data)

    refiners = create_and_sort_refiners(samples, environment, novelty_factor, coherency_factor, population_size, mutation_rate, win_size, mid_win_size, mic_win_size, max_gram_len, diversity_factor, mac_win_size, tolerance_factor, shuffle_matches, levelload_conn)

    for index, refiner in enumerate(refiners):
        # Create sub directory for refiner
     
        print(f"################ Refining {index} ###################")
        if refiner.sample == environment:
            results.append(refiner.sample)
        else:
            xmlGrammar, output = refiner.refineGrammars(stash_path,generation_conn, createpop_conn)
            results.append(output)
        
        if levelload_conn:
            data = {
                "progress": "100",
                "currFinalOut": dp.combine_squares_into_level(results),
                "end": False
            }
            levelload_conn.send(data)

        print(f"########### Refining for {index} done ##############\n")
    
    if levelload_conn:
            data = {
                "progress": "Completed",
                "currFinalOut": None,
                "end": True
            }
            levelload_conn.send(data)
    
    print(f"########## Refinment done for {len(refiners)} grammars #############")
    print("Output is saved")

# OBSOLETE
def gym_whole(name, novelty_factor, coherency_factor, population_size, mutation_rate, win_size, mid_win_size, mic_win_size, max_gram_len, diversity_factor, mac_win_size, tolerance_factor, shuffle_matches=0, sample_path="../resources/lvl-1.txt", env_path="../resources/env.txt"):
    
    # Create sub directory for  divcon attempt
    main_path = create_dir("../outputs", "_whole")
    
    print("\n######## Refinment by whole content ############\n")
    arguments = arguments_str(name, novelty_factor, coherency_factor, population_size, mutation_rate, win_size, mid_win_size, mic_win_size, max_gram_len, diversity_factor,mac_win_size)
    print(arguments) 
    dp.save_file(arguments, os.path.join(main_path, "arguments.txt"))
   

    sample = intp.parse_environment(sample_path)
    environment = ""

    geneticRuleRefiner = GeneticRuleRefiner(sample, environment, novelty_factor, coherency_factor, population_size, mutation_rate, win_size, win_size, mid_win_size, mid_win_size, mic_win_size, mic_win_size, max_gram_len, diversity_factor, mac_win_size, mac_win_size, 0, tolerance_factor,shuffle_matches)

    # Create sub directory for refiner
    refiner_path = create_dir(main_path, "_index_"+str(0))

    print(f"\nRefining now grammars for whole level...")

    print(f"################ Refining ###################")
    xmlGrammar, output = geneticRuleRefiner.refineGrammars(refiner_path)
    print(f"\n########### Refinment done for grammars ##############")

    dp.save_file(output, os.path.join(main_path, name+".txt"))
    print("Output is saved")

# GPT
def get_unique_name():
    timestamp = time.strftime("%d-%m-%Y_%H:%M:%S")

    # Extract the last part of the UUID
    unique_id = str(uuid.uuid4()).split("-")[-1]
    name = f"{timestamp}_{unique_id}"
    return name

def arguments_str(name, novelty_factor, coherency_factor, population_size, mutation_rate, win_size, mid_win_size, mic_win_size, max_gram_len, diversity_factor, mac_win_size):
    output_string = "===================== Arguments =====================\n"
    output_string += f"name of the output file:        {name}.txt\n"
    output_string += f"novelty factor:                 {novelty_factor}\n"
    output_string += f"coherency factor:               {coherency_factor}\n"
    output_string += f"population size:                {population_size}\n"
    output_string += f"mutation rate:                  {mutation_rate}\n"
    output_string += f"window size (for rule making):  {win_size}\n"
    output_string += f"macro evaluation window size:  {mac_win_size}\n"
    output_string += f"middle evaluation window size:  {mid_win_size}\n"
    output_string += f"micro evaluation window size:   {mic_win_size}\n"
    output_string += f"maximum grammar length:         {max_gram_len}\n"
    output_string += f"diversity factor:               {diversity_factor}\n"
    output_string += "=====================================================\n"
    return output_string


# GPT
def create_dir(path: str = ".", extension: str = "", unique_name:bool=True):
    directory_full_path = ""
    
    if unique_name:
        directory_full_path = os.path.join(path, get_unique_name()+extension)
    else:
        directory_full_path = os.path.join(path, time.strftime("%d-%m-%Y_%H:%M:%S")+"_"+extension)
    
    try:
        os.makedirs(directory_full_path)
        # print(f"Directory '{directory_name}' created successfully at '{directory_full_path}'.")
    except OSError:
        print(f"Creation of directory at '{directory_full_path}' failed.")

    return directory_full_path

# GPT
def main():
    
    if not os.path.exists("../outputs"):
        os.makedirs("../outputs")
        print(f" Output directory created.")
    else:
        print(f"Output directory already exists.")
 
    
    parser = argparse.ArgumentParser(
        description="Split level into rectangles and refine grammars.")
    parser.add_argument("--name", type=str, default="output",
                        help="Name the output (default: output)")
    parser.add_argument("--mode", choices=["divcon", "whole"],
                        default="divcon", help="Refinement mode: divcon (default) or whole")
    parser.add_argument("--novelty_factor", type=float, default=0.0,
                        help="Novelty factor, determines the degree of existence of unseen patterns (default: 0.0)")
    parser.add_argument("--coherency_factor", type=float, default=1,
                        help="Coherency factor determines the degree of coherency (default: 1)")
    parser.add_argument("--population_size", type=int, default=100,
                        help="Population size for GeneticRuleRefiner (default: 100)")
    parser.add_argument("--mutation_rate", type=float, default=0.25,
                        help="Mutation rate for GeneticRuleRefiner (default: 0.25)")
    parser.add_argument("--win_size", type=int, default=5,
                        help="Rectangle window size for rule making (default: 4)")
    parser.add_argument("--mid_win_size", type=int, default=4,
                        help="Middle sized rectangle window size for evaluation (default: 4)")
    parser.add_argument("--mic_win_size", type=int, default=2,
                        help="Micro sized rectangle window size for rule generation (default: 2)")
    parser.add_argument("--max_gram_len", type=int, default=10,
                        help="Maximum allowed grammar length (default: 10)")
    parser.add_argument("--diversity_factor", type=float, default=0,
                        help="Diversity factor, mirros the occurence probability of sample patterns (default: 0)")
    parser.add_argument("--mac_win_size", type=int, default=6,
                        help="Macro sized rectangle window size for rule generation (default: 6)")
    parser.add_argument("--tolerance_factor", type=float, default=0.05,
                        help="Tolerance factor for relatively non fit offsprings (default: 0.5)")
    args = parser.parse_args()

    name = args.name
    mode = args.mode
    novelty_factor = args.novelty_factor
    coherency_factor = args.coherency_factor
    population_size = args.population_size
    mutation_rate = args.mutation_rate
    win_size = args.win_size
    mid_win_size = args.mid_win_size
    mic_win_size = args.mic_win_size
    max_gram_len = args.max_gram_len
    diversity_factor = args.diversity_factor
    mac_win_size = args.mac_win_size
    tolerance_factor = args.tolerance_factor
    
    if mode == "divcon":
        gym_divcon(name, novelty_factor, coherency_factor, population_size, mutation_rate, win_size, mid_win_size, mic_win_size, max_gram_len, diversity_factor,mac_win_size, tolerance_factor)
    elif mode == "whole":
        gym_whole(name, novelty_factor, coherency_factor, population_size, mutation_rate, win_size, mid_win_size, mic_win_size, max_gram_len, diversity_factor,mac_win_size, tolerance_factor)


if __name__ == "__main__":
    main()
    
    
"""
python3 GrammarGym.py --name anan --mode whole --novelty_factor 1.0 --coherency_factor 1.0 --population_size 100 --mutation_rate 0.25 --win_size 6 --mid_win_size 16 --mic_win_size 8 --max_gram_len 100
"""