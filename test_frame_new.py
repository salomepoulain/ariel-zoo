import sys
import shutil
from pathlib import Path
import scipy.sparse as sp

# Ensure we can import the module
sys.path.append("src")

from canonical_toolkit.src.base.matrix.frame import MatrixFrame
from canonical_toolkit.src.base.matrix.series import MatrixSeries
from canonical_toolkit.src.base.matrix.matrix import MatrixInstance

def test_description_and_save():
    print("Testing description property...")
    
    # Create series A
    series_a_instances = []
    for idx in range(3):
        mat = sp.random(5, 5, density=0.3, format="csr")
        inst = MatrixInstance(
            matrix=mat,
            label="series_A",
            index=idx,
            tags={"type": "features"}
        )
        series_a_instances.append(inst)
    series_a = MatrixSeries(instances_list=series_a_instances)

    frame = MatrixFrame(series=[series_a])
    
    desc = frame.description
    print(f"Description: {desc}")
    
    # Note: labels are sorted, series_A is the only one.
    # Format: MatrixFrame_{num_series}series_{num_indices}indices_{shape_str}_{labels_str}
    expected_start = "MatrixFrame_1series_3indices_5x5_series_A"
    assert expected_start in desc, f"Expected {expected_start} to be part of {desc}"
    
    print("Testing save method with auto-naming...")
    # Clean up previous runs
    data_dir = Path.cwd() / "__data__"
    folder_name = desc
    save_path = data_dir / folder_name
    
    # Ensure clean state
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path_2 = data_dir / f"{folder_name}_2"
    if save_path_2.exists():
        shutil.rmtree(save_path_2)
    
    # Test 1: First save
    frame.save()
    assert save_path.exists(), f"Folder {save_path} should be created"
    print(f"First save successful: {save_path}")
    
    # Test 2: Second save (should create _2)
    frame.save()
    assert save_path_2.exists(), f"Duplicate folder {save_path_2} should be created"
    print(f"Second save (duplicate) successful: {save_path_2}")
    
    # Clean up
    if save_path.exists():
        shutil.rmtree(save_path)
    if save_path_2.exists():
        shutil.rmtree(save_path_2)

    print("✅ Description and Save tests passed!")

if __name__ == "__main__":
    test_description_and_save()
