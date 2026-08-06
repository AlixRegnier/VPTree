# Fast nearest-neighbor

## When using this library

* 1NN query (only one candidate)
* Nearest-Neighbor (NN) with ``epsilon = 0.0``
* Approximate Nearest-Neighbor (ANN) with ``epsilon > 0.0``
* Nearest-neighbor peeling **without replacement**  
(perfect for  solving nearest-neighbor heuristic)

## Usage

Easy to use. We provide a python wrapper --> ``pip3 install vptree``

## Requirements

This library is a standalone header-only that requires ``C++17``.

## Implementation

* Uses a VPTree [1] as underlying structure.
* Median is found in $\mathcal{O}(n)$ by C++ STL introselect (``nth_element``).
* Remaining elements are maintened in $\mathcal{O}(1)$ time complexity with a swap-and-pop algorithm.


## Not implemented yet 

**KNN**.

## References

[1] ref vptree  
[2] ref relaxed vptree