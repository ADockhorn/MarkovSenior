"""
Author: Mehmet Kayra Oguz 
Date: July 10, 2023
"""
import math
import sys
import Util as dp
import concurrent.futures
from rule_gen import PatternParse as pp
from rule_gen import RuleGenerator as rg
from markov_junior.MarkovJunior import *
from markov_junior import Interpretor as intp
from markov_junior import Decompiler as decom
from markov_junior import PatternMatch as pm
from functools import lru_cache
import statistics
from numba import jit
from src.markov_junior.PatternMatch import replacePattern

def calculate_fitness(grammar, grf_instance):
    return grammar, grf_instance.grammarFitness(grammar)


class GeneticRuleRefiner:

    def __init__(self, sample: str, environment: str="", noveltyFactor : float=0.0, coherencyFactor : float=1, populationSize: int = 100, mutationRate: float = 0.25, windowHeight:int= 4, windowWidth: int =4, evalWinMidHeight:int=4, evalWinMidWidth:int=4,  evalWinMicHeight:int=2, evalWinMicWidth:int=2, maxGrammarLength:int=10, diversityFactor:float=0.0, evalWinMacHeight:int=6, evalWinMacWidth:int=6, index:int=0, toleranceFactor: float=0.2, shuffleMatches: int=0 ):
        """
        Extracts patterns from sample and initializes a list of random grammar 
        """
        
        print("############ Initializing new refiner ###############")
        
        # The populations size and mutation rate of the genetic refiner
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        
        # Maximum grammar length
        self.maxGrammarLength = maxGrammarLength

        # How much pattern novelty should the output have relative to sample, 0 min novelty, 1 max novelty
        self.noveltyFactor = noveltyFactor 

        # How coherent the output should be relative to sample, 0 min coherency, 1 max coherency
        self.coherencyFactor = coherencyFactor

        # This adapts the occuerence probability of patterns
        # 0     original probabilites  
        # 0.5   almost all probabilities are equal, 
        # 1     the probabilities that are originally low have higher probabilities and vice versa
        self.diversityFactor = diversityFactor
        
        # Fitness toleration, higher values will make less fit offsprings to pass
        self.toleranceFactor = toleranceFactor
        
        # Allow shufföinh of matches
        self.shuffleMatches = shuffleMatches
        
        # Window sizes used for grammar making and evaluation
        self.windowHeight = windowHeight
        self.windowWidth = windowWidth

        # Sample is the source content where the rules are generated: str
        self.sample = sample

        # Environment is where a grammar is applied, this is also the origin environment of sample: str
        self.environment = pp.convert_to_env(sample)
        self.envAsNpArray = pm.strToNpArray(self.environment)
           
        # Fitness evaluation windows for middle and micro sized 
        self.evalWinMidHeight = evalWinMidHeight
        self.evalWinMidWidth = evalWinMidWidth
        
        self.evalWinMicHeight = evalWinMicHeight
        self.evalWinMicWidth = evalWinMicWidth
        
        self.evalWinMacHeight = evalWinMacHeight
        self.evalWinMacWidth = evalWinMacWidth

        # Pattern alphabet is the unique patterns from the sample
        self.sampleAlphabet = pp.get_alphabet_from_sample(self.sample, self.windowHeight, self.windowWidth)
        self.environmentAlphabet = pp.get_alphabet_from_sample(self.environment, self.windowHeight, self.windowWidth)

        # This alphabet includes occurances of each single grid used for diversity measures
        self.sampleAtomicOccuAlphabet = pp.get_occurance_alphabet_from_sample(self.sample, 1, 1, include_env=False)
        # Used to keep environment coherent
        self.environmentAtomicAlphabet = pp.get_alphabet_from_sample(self.environment, 1, 1)
        
        # Relations are contextual and positional relations between the alphabet elements
        self.atomicEnvRelations = pp.extract_relations(self.sample, self.environmentAlphabet, onlyNeighbours=False)
        self.atomicEnvRelationMaxFitness = self.relationFitness(self.sample)
        self.patternRelations = pp.extract_relations(self.sample, self.sampleAlphabet, onlyNeighbours=True)

        # Worst KL-Divergence scores
        antiSampleWindowMacro = pp.wildcard_window(len(self.environment.strip().split('\n')), len(self.environment.strip().split('\n')[0]))        
        self.worstKLDivClearedSample = self.contentFitness(antiSampleWindowMacro, self.environment, 8, 8, 0) 
        
        self.worstKLDivFitnessMacro = self.contentFitness(antiSampleWindowMacro, self.sample, self.evalWinMacHeight, self.evalWinMacWidth)
        self.worstKLDivFitnessMiddle = self.contentFitness(antiSampleWindowMacro, self.sample, self.evalWinMidHeight, self.evalWinMidWidth)
        self.worstKLDivFitnessMicro = self.contentFitness(antiSampleWindowMacro, self.sample, self.evalWinMicHeight, self.evalWinMicWidth)
        
        # Pair database, this list has all combinations of relation based
        self.pairs = rg.generate_relation_based_in_out_pairs(self.sample, self.environment, self.windowHeight, self.windowWidth)

        # Population consist of tuples (grammar, grammar's fitness percentage)
        grammars = rg.generate_relation_based_random_grammars(self.sample, self.environment, self.windowHeight, self.windowWidth, self.populationSize, self.pairs)

        print("Generating inital population...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self.population = list(executor.map(calculate_fitness, grammars, [self]*len(grammars)))

        self.fitnessList = [fitness for grammar, fitness in self.population]

        print("Generating sampling data...")
        #self.pairs = rg.generate_all_possible_pairs(self.sampleAlphabet, self.environmentAlphabet, self.windowHeight, self.windowWidth)
        
        print("################ Refiner initialized ################\n")
        
        # For information sending
        self.currBestOut = ""
        self.currSample = ""
        self.currGeneration = 0
         
        self.generatedOffsprings = 0
        self.highestFitness = 0
        self.avgFitness = 0
        self.index = index 
 
    def refineGrammars(self, stash_path:str=None, generation_conn=None, createpop_conn=None):
        """
        Refines the generated rules and grammars by genetic evolution
        """
        print("Population size:",self.populationSize,"|","Mutation rate:", self.mutationRate)
        print("\nRefining now rules")
        
        self.currGeneration = 0
        start_time = time.time()
        tempMutationRate = self.mutationRate
        prevHighestFitness = 0
        
        while True:
            if generation_conn:
                data = {
                    "currBestOut": None,
                    "currSample": self.sample,
                }
                generation_conn.send(data)
            
            self.currGeneration += 1
            offspring, avgFitness = self.createPop(createpop_conn,generation_conn)
            grammar, highestFitness = offspring

           # Dynamic mutation to avoid diversity collapse in population
            if highestFitness - avgFitness < 2:
                #self.mutationRate = 0.5
                print("Diversity collapse!")
            else:
                self.mutationRate = tempMutationRate

            context = Context(self.environment, grammar, self.shuffleMatches)
            output = context.applyGrammar()
            
            
            if stash_path and generation_conn and createpop_conn:  # For gui  
                # Save output image as txt
                dp.save_file(output, (stash_path+"/output"+str(self.index)+".txt"))
                # Save environment as txt
                dp.save_file(self.environment, (stash_path+"/env"+str(self.index)+".txt"))
                # Save grammar as xml
                grammarXml = decom.parse_grammar(context, stash_path+"/env"+str(self.index)+".txt")
                dp.save_file(grammarXml, stash_path+"/grammar"+str(self.index)+".xml") 
            elif stash_path:
                # Save output image as txt
                dp.save_file(output, (stash_path+"/output_gen_"+str(self.currGeneration)+"_fit_"+str(round(highestFitness))+".txt"))
                # Save grammar as xml
                grammarXml = decom.parse_grammar(context)
                dp.save_file(grammarXml, stash_path+"/grammar_gen_"+str(self.currGeneration)+"_fit_"+str(round(highestFitness))+".xml")
                
            self.status(time.time()-start_time, self.currGeneration,highestFitness, avgFitness, output, context, prevHighestFitness)
            prevHighestFitness = highestFitness
            
            if (highestFitness) >= 100 - (self.toleranceFactor*100) or self.currGeneration > 1000:
                
                if generation_conn:
                    data = {
                        "currBestOut": None,
                        "currSample": self.sample,
                    }
                        
                    generation_conn.send(data)
                    
                return (grammarXml, output)


    def createPop(self, createpop_conn=None, generation_conn=None) -> tuple:
        """
        Updates current population with new population that is created by reproduction
        of mostly higher fitness offspringNum
        Returns the best grammar with its fitness score from the updated population 
        """
        print("==========================================================================\n")
        print("Generating offsprings...")
        
        newPopulation = []
        newFitnessList = []
        
        start_time = time.time()
  
        # Create new offspringNum for new population
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.createOffspring, i) for i in range(self.populationSize)]
            offspringNum = 0
            highestFitness = 0
            
            for future in concurrent.futures.as_completed(futures):
                
                highestFitnessChanged = False
                
                offspring, fitness = future.result()
                newPopulation.append((offspring, fitness))
                newFitnessList.append(fitness)
                offspringNum += 1
                
                if fitness >= highestFitness:
                    highestFitness = fitness
                    highestFitnessChanged = True
                
                self.clear()
                print("Number of offsprings generated:", offspringNum)
                print("Highest fitness until now:     ",round(highestFitness, 2), "%")
                
                # Send information
                if createpop_conn:
                    # Set to be sent information
                    self.generatedOffsprings = offspringNum
                    self.highestFitness = round(highestFitness, 2)
                    self.avgFitness = round(statistics.mean(newFitnessList),2)
                    
                    data = {
                        "generatedOffsprings": str(self.generatedOffsprings),
                        "popSize": str(self.populationSize),
                        "highestFitness": str(self.highestFitness),
                        "avgFitness": str(self.avgFitness),
                        "currGeneration": str(self.currGeneration),
                        "index": str(self.index)
                    }
                    createpop_conn.send(data)         
                
                if generation_conn:
                    if highestFitnessChanged:
                        context = Context(self.environment, offspring, self.shuffleMatches)
                        output = context.applyGrammar()
                        
                        # Send Data
                        if generation_conn:
                            data = {
                                "currBestOut": output,
                                "currSample": self.sample,
                            }
                            generation_conn.send(data)
            #self.clear()
            
            # Send information
            if createpop_conn:
                data = {
                    "generatedOffsprings":"0",
                    "popSize": str(self.populationSize),
                    "highestFitness": "- ",
                    "avgFitness": "- ",
                    "currGeneration": str(self.currGeneration+1) ,
                    "index": str(self.index)
                }
                createpop_conn.send(data)        
            
        # Send Data
        if generation_conn:
            data = {
                "currBestOut": None,
                "currSample": self.sample,
            }
            generation_conn.send(data) 
            
                # Send information
        if createpop_conn:
            data = {
                "generatedOffsprings":"0",
                "popSize": str(self.populationSize),
                "highestFitness": "- ",
                "avgFitness": "- ",
                "currGeneration": str(self.currGeneration+1) ,
                "index": str(self.index)
            }
            createpop_conn.send(data)        
        
        # Send Data
        if generation_conn:
            data = {
                "currBestOut": None,
                "currSample": self.sample,
            }
            generation_conn.send(data) 
            
        print("Population generation took:",round(time.time() - start_time,2), "seconds")
        # Sort grammars and fitnesses based on their fitness scores and return the best grammar
        self.population = sorted(newPopulation, key=lambda item: item[1], reverse=True)
        self.fitnessList = sorted(newFitnessList, reverse=True)

        return (self.population[0], statistics.mean(self.fitnessList))

    def createOffspring(self, i) -> tuple:
        """
        Create offspring from 2 random parent offsprings weightes on their fitness
        If an exceptionally high fit offspring is generated we use it and a random 
        offspring, this allows to improve already good ofsprings with diverse
        offsprings and avoids diversity collapse
        """
        
        if self.fitnessList[0] >= 80:
            parents = (self.population[0], random.choice(self.population))
        else:
            # Get two random parents, weighted with fitness
            parents = random.choices(self.population, weights=self.fitnessList, k=2)
        # Evolve
        offspringGrammar = self.mutate(self.crossover(parents[0], parents[1]))
        # Evaluate
        offspringFitness = self.grammarFitness(offspringGrammar)
        
        return (offspringGrammar, offspringFitness)

    def grammarFitness(self, grammar: list) -> float:
        """
        Calculate grammar fitness by KL-Divergence on macro and micro levels
        """
        context = Context(self.environment, grammar, self.shuffleMatches)
        output = context.applyGrammar()
        
        mac = self.contentFitness(output, self.sample, self.evalWinMacHeight, self.evalWinMacWidth)
        mid = self.contentFitness(output, self.sample, self.evalWinMidHeight, self.evalWinMidWidth)
        mic = self.contentFitness(output, self.sample, self.evalWinMicHeight, self.evalWinMicWidth)

        # Evaluate environment defects   
        #cle = pp.kl_divergence(pp.convert_to_env(output), self.environment, 6, 6, True)
        #clef = self.proximityScoreFloor(cle, self.worstKLDivClearedSample, 0)

        # Evaluate the grammar using 3 differnt windows
        macrof = self.proximityScoreFloor(mac, self.worstKLDivFitnessMacro, self.worstKLDivFitnessMacro - self.worstKLDivFitnessMacro * self.coherencyFactor)
        middf = self.proximityScoreFloor(mid, self.worstKLDivFitnessMiddle, self.worstKLDivFitnessMiddle - self.worstKLDivFitnessMiddle * self.coherencyFactor)
        microf = self.proximityScoreFloor(mic, self.worstKLDivFitnessMicro, self.worstKLDivFitnessMicro - self.worstKLDivFitnessMicro * self.coherencyFactor)

        divFitness = pp.grid_frequency_score(self.sampleAtomicOccuAlphabet, pp.get_occurance_alphabet_from_sample(output), self.diversityFactor)
        klfitness = (np.average(np.array([ middf, microf, macrof])))
        if self.diversityFactor > 0:
            fitness = np.average(np.array([ divFitness, klfitness])) * 100
        else:
            fitness = klfitness * 100
            
        # Punish environment defects 
        if self.environmentFitness(output) < 1 or output == self.environment: 
            return fitness * 0.25

        return fitness

    def contentFitness(self, generatedContent: str, sampleContent: str, windowHeight: int = 3, windowWidth: int = 3, diversityFactor: int = 0) -> float:
        """
        Fitness estimation by KL-Divergance, the formula from the paper is used but modified
        """
                
        klDivergencePQ = pp.kl_divergence(sampleContent, generatedContent, windowHeight, windowWidth, diversityFactor)
        klDivergenceQP = pp.kl_divergence(generatedContent, sampleContent, windowHeight, windowWidth, diversityFactor)

        return ((1-self.noveltyFactor) * klDivergencePQ + (self.noveltyFactor) * klDivergenceQP)

    def environmentFitness(self, output:str):
        klScore = self.contentFitness(pp.convert_to_env(output), self.environment, 8, 8, 0) 
        return self.proximityScoreFloor(klScore, self.worstKLDivClearedSample, 0)
        
    def relationFitness(self, output:str, noveltyFactor: float = 0.0) -> float:
        """
        Calculates the fitness percentage of a relation with the sample relation
        Fitness is calculated by mixing positional and contextual coherence distribution similarities
        Returns percentage of relation similarity between sample relation and input relation
        in terms of probailistic distribution
        """
        relations = pp.extract_relations(output, self.environmentAtomicAlphabet, False)

        fitness = 0
        
        # For each relation from target relation collection find how much matching there is
        for targetKey, targetRelation in self.atomicEnvRelations.items():
            targetDistRel, targetPosRel, targetRatio = targetRelation
            if targetKey in relations:

                distRel, posRel, ratio = relations[targetKey]

                # Check how close are relations to each other based on distribution propability
                # Higher probability distributions have higher positive impact on fitness
                for eucDist, probabilty in targetDistRel.items():
                    if eucDist in distRel:
                        fitness += self.proximityScore(
                            distRel[eucDist], probabilty) * probabilty
                    else:
                        fitness += noveltyFactor * probabilty * 1  # More contextual coherence

                for vector, probabilty in targetPosRel.items():
                    if vector in posRel.keys():
                        fitness += (self.proximityScore(
                            posRel[vector], probabilty) * probabilty)
                    else:
                        fitness += noveltyFactor * probabilty

                if ratio > 0:
                    fitness += self.proximityScore(ratio, targetRatio)
                else:
                    fitness += noveltyFactor
            else:
                fitness += noveltyFactor * (len(targetDistRel.items()) + len(targetPosRel.items())+1)

        return fitness
    

    def crossover(self, parent1: list, parent2: list) -> list:
        """
        Crossover grammars by mixing both grammars
        Each crossover produces two new grammars
        """
        g1, f1 = parent1
        g2, f2 = parent2
        
        g1 = rg.split_list(g1, 2)
        g2 = rg.split_list(g2, 2)
        offspringGrammar1 = g1[0] + g2[1]
        offspringGrammar2 = g2[0] + g1[1]

        if self.grammarFitness(offspringGrammar1) > self.grammarFitness(offspringGrammar2):
            return offspringGrammar1

        return offspringGrammar2

    def mutate(self, grammar: list) -> list:
        """
        Mutation takes part in two places, first part mutates or expands the grammar by a single item,
        second part randomly inserts items in to the grammar, after that invalid parts are purged  
        """
        pairNum = random.randint(1, 30)
        rules = rg.generate_rules(random.sample(self.pairs, pairNum))
        ruleSetsAndRules = rg.generate_rulesets_and_rules(rules)
        
        # Mutate items of the grammar
        for itemIndex, item in enumerate(grammar):      
            if random.uniform(0, 1) < self.mutationRate or not item.applyable:  
                if ruleSetsAndRules:
                    grammar[itemIndex] = random.choice(ruleSetsAndRules)      
                        
        # Insert random rules into grammar    
        for i in range(len(grammar)):      
            if random.uniform(0, 1) < self.mutationRate and len(grammar) < self.maxGrammarLength:
                grammar.insert(random.randint(0, len(grammar)), random.choice(ruleSetsAndRules))   
        
        # Repairing mechanisms
        context = Context(self.environment, grammar, self.shuffleMatches)
        output = context.applyGrammar()

        # Purge non applyable items of the grammar
        context.grammar = self.purgeGrammar(context.grammar)
        
        # Repair core rules of surviving items of grammr
        for itemIndex, item in enumerate(context.grammar):
            if random.uniform(0, 1) < self.mutationRate:  
                self.mutateItemCore(item)      
                         
        return context.grammar    

    def purgeGrammar(self, grammar:list):
        """
        This is so bad but works
        """
        if len(grammar) > 1:
            for itemIndex, item in enumerate(grammar):      
                if not item.applyable and len(grammar) > 2:
                    grammar.pop(itemIndex)   
        return grammar
    
    def mutateItemCore(self, item):
        """
        Mutates core input output pairs if rule is not applyable
        """
        if isinstance(item, MultiRule):
            for rule in item.rules:
                if not rule.applyable or np.array_equal(rule.pattern, rule.replacement) or np.all(np.isin(rule.replacement,  [ord("*"), ord("-")])):
                    pattern, replacement = random.choice(self.pairs)
                    rule.pattern = pattern
                    rule.replacement = replacement
            return
        elif isinstance(item, Markov):
            for thing in item.items:
                self.mutateItemCore(thing)
        elif isinstance(item, Sequence):
            for thing in item.items:
                self.mutateItemCore(thing)
        
    def proximityScore(self, elem1: float, elem2: float) -> float:
        """
        Returns a score between 0-1 for how close two numbers are, returns 1 for identical numbers
        """
        if elem1 == elem2:
            return 1
        distance = abs(elem1 - elem2)
        maxDistance = max(abs(elem1), abs(elem2))

        if maxDistance > 0:
            return 1 - (distance / maxDistance)
        return 0

    def proximityScoreFloor(self, number: float, highest: float = 36.043653389117146, lowest: float = 0.0, tolerance: float = 0.0) -> float:
        """
        Returns a score between 0-1 for how close a number is to lowest relative to the highest number
        """
        if number <= lowest or tolerance == 1:
            return 1.0
        elif number >= highest:
            return 0.0
        else:
            return (1.0 - ((number - lowest) / (highest - lowest))) / (1.0)


    def proximityScoreCeiling(self, number: float, highest: float = 36.043653389117146, lowest: float = 0.0) -> float:
        """
        Returns a score between 0-1 for how close a number is to the highest relative to the lowest
        """
        if number >= highest:
            return 1.0
        elif number <= lowest:
            return 0.0
        else:
            return (number - lowest) / (highest - lowest)

    def status(self, elapsedTime, generation, maxFitness, avgFitness, output, context, prevHighestFitness) -> None:
        """
        Prints the current status of the evolutionary process 
        """
        print("--------------------------------------------------------------------------")
        print("Time elapsed since start: ", round(elapsedTime,2), "seconds")
        print("--------------------------------------------------------------------------")
        print("Novelty factor:",self.noveltyFactor,"/","Coherency factor:", self.coherencyFactor,"/","Window height x width:",self.windowWidth,"x",self.windowHeight)
        print("--------------------------------------------------------------------------")
        print("Diversity factor:", self.diversityFactor, "/", "Highest fitness:", round(maxFitness, 2), "%", "/", "Average fitness:", round(avgFitness, 2), "%")
        print("--------------------------------------------------------------------------\n")
        
        if maxFitness - prevHighestFitness > 0:
            print("\033[42mGeneration:    {}\033[0m\n".format(generation))  # Green background for improvement
        elif maxFitness - prevHighestFitness == 0:
            print("\033[47mGeneration:    {}\033[0m\n".format(generation))  # Grey background for no change
        else:
            print("\033[41mGeneration:    {}\033[0m\n".format(generation))  # Red background for negative change
        
        print(output, end="\n\n")
        decom.parse_grammar(context=context)

       

    def clear(self) -> None:
        print("\033[A\033[A\n\033[A\033[A")