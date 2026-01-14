
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from canonical_toolkit.morphology.similarity.sim_matrix.matrix import SimilarityMatrix
from canonical_toolkit.morphology.similarity.sim_matrix.series import SimilaritySeries
from canonical_toolkit.morphology.similarity.options import VectorSpace, MatrixDomain
import scipy.sparse as sp
import numpy as np

def test_cumulative_type():
    print("Creating SimilarityMatrix...")
    mat = sp.csr_matrix([[1, 0], [0, 1]])
    sim_mat = SimilarityMatrix(
        matrix=mat,
        space=VectorSpace.FRONT,
        radius=0,
        domain=MatrixDomain.FEATURES
    )
    
    print(f"Original type: {type(sim_mat)}")
    
    # Test replace directly
    print("Testing replace()...")
    replaced = sim_mat.replace(matrix=mat)
    print(f"Replaced type: {type(replaced)}")
    
    if not isinstance(replaced, SimilarityMatrix):
        print("FAIL: replace() returned wrong type!")
    else:
        print("PASS: replace() returned correct type.")

    # Test cumulative in series
    print("\nTesting SimilaritySeries.to_cumulative()...")
    instances = []
    for r in range(2):
        inst = SimilarityMatrix(
            matrix=mat,
            space=VectorSpace.FRONT,
            radius=r,
            domain=MatrixDomain.FEATURES
        )
        instances.append(inst)
        
    series = SimilaritySeries(instances)
    cumulative_series = series.to_cumulative(inplace=False)
    
    first = cumulative_series[0]
    second = cumulative_series[1]
    
    print(f"Radius 0 type: {type(first)}")
    print(f"Radius 1 type: {type(second)}")
    
    if not isinstance(second, SimilarityMatrix):
         print("FAIL: Cumulative series contains wrong type at index > 0")
    else:
         print("PASS: Cumulative series contains correct types")

if __name__ == "__main__":
    test_cumulative_type()
