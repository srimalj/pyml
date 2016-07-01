"""
Common functionality
"""
import sys

def largestK(a, k):
    """
    Return the indices of the largest k elements in a
    """
    out = []
    for iter in xrange(0, min(k, len(a))):

        largestIndex = None
        largestValue = -sys.maxint
        
        for idx in xrange(0, len(a)):
            if not(idx in out) and (a[idx] > largestValue):
                largestValue = a[idx]
                largestIndex = idx
        out.append(largestIndex)
        
    return out

