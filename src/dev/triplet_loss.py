import torch
import torch.nn as nn

'''
Compute the difference between
- the distance between the query and the relevant document 
and 
- the distance between the query and the irrelevant document, plus the margin. 

The output loss is this difference, if it is positive, and zero otherwise.
'''

def triplet_loss_function((query, relevant_document, irrelevant_document), distance_function, margin):
    relevant_distance = distance_function(query, relevant_document)
    irrelevant_distance = distance_function(query, irrelevant_document)
    triplet_loss = max(0, relevant_distance - irrelevant_distance + margin)
    return triplet_loss