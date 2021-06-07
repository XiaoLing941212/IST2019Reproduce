from __future__ import division
import math
import random
import pdb
import itertools

### implementation of SWAY algorithm
def sway(pop, split, better, terminate):
    def cluster(items, t):
        if len(items) < t:
            return items
        
        west, east, westItems, eastItems = split(items)

        if better(east, west):
            selected = eastItems
        if better(west, east):
            selected = westItems
        if not better(east, west) and not better(west, east):
            selected = random.sample(westItems+eastItems, len(items)//2)
        
        return cluster(selected, t)
    
    return cluster(pop, terminate)
       

### function to excute SWAY, and define the support function in SWAY
def sway_runner(table):
    def _calculateDis(r1, r2):
        return sum([(item1 - item2) ** 2 for item1, item2 in zip(r1, r2)]) ** 0.5
    
    def _split(pop):
        # randomly select one row
        rand = random.choice(pop)

        # calculate the distance of all the rows to the random row, and generate the east item
        ds = [_calculateDis(i[1], rand[1]) for i in pop]
        east = pop[ds.index(max(ds))]
        
        # calculate the distance of all the rows to the east row, and generate the west item
        ds = [_calculateDis(i[1], east[1]) for i in pop]
        west = pop[ds.index(max(ds))]

        # calculate the distance of a point to east and west
        mappings = []
        c = _calculateDis(east[1], west[1])

        for item in pop:
            a = _calculateDis(item[1], west[1])
            b = _calculateDis(item[1], east[1])
            d = (a**2 + c**2 - b**2) / (2*c)    # cosine rule
            mappings.append((item, d))
        
        mappings = sorted(mappings, key=lambda x: x[1])
        mappings = [i[0] for i in mappings]

        # assign each point to east or west
        n = len(mappings)
        eastItems = mappings[:int(n*0.2)] + mappings[int(n*0.5):int(n*0.8)]
        westItems = mappings[int(n*0.2):int(n*0.5)] + mappings[int(n*0.8):]
        # eastItems = mappings[:int(n*0.5)]
        # westItems = mappings[int(n*0.5):]

        return west, east, westItems, eastItems
    
    def bin_domination(ind1, ind2):
        for i, j in zip(ind1, ind2):
            if i > j:
                return False
            
            if ind1 == ind2:
                return False
            
            return True

    def _better(part1, part2):
        return bin_domination(part1, part2)

    # todo: con_domination
    
    # the execution of SWAY
    # terminate = math.sqrt(len(table))
    terminate = 100
    res = sway(table, _split, _better, terminate)

    return res
    

