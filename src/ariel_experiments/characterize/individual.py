from __future__ import annotations

# Standard library
import json
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

# Third-party libraries
import numpy as np
from rich.console import Console

# Local libraries
from ariel.body_phenotypes.robogen_lite.modules.brick import BRICK_MASS
from ariel.body_phenotypes.robogen_lite.modules.core import CORE_MASS
from ariel.body_phenotypes.robogen_lite.modules.hinge import (
    ROTOR_MASS,
    STATOR_MASS,
)

if TYPE_CHECKING:
    from networkx import DiGraph

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
SEED = 42

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)


# Type Aliases and TypeVars
T = TypeVar("T")

# Concrete value aliases (don't use TypeVar for concrete aliases)
NumericProperty = int | float
NonNumProperty = str | bool | list | tuple | dict | set
GraphProperty = NumericProperty | NonNumProperty

# Index of a graph in the population
type GraphIndex = int
type IndexMappings = list[GraphIndex]

# Derived property can be a value or list of indexes
DerivedProperty = GraphProperty | IndexMappings

# name aliases
type GraphPropertyName = str
type DerivedPropertyName = str

# Generic mapping when values are homogeneous (use NamedGraphPropertiesT[T])
NamedGraphPropertiesT = dict[GraphPropertyName, T]
# Backwards-compatible mixed container
type NamedGraphProperties = dict[GraphPropertyName, GraphProperty]


# Generic analyzer Protocol: callable that returns dict[str, T]
class PropertyAnalyzer(Protocol[T]):
    def __call__(self, individual: DiGraph) -> NamedGraphPropertiesT[T]: ...


def analyze_module_counts(individual: DiGraph) -> NamedGraphPropertiesT[int]:
    """
    Count different module types and edges in a directed graph individual.

    Parameters
    ----------
    individual : DiGraph
        A directed graph where nodes have a 'type' attribute indicating
        module type.

    Returns
    -------
    dict[str, int]
        Dictionary with counts for 'core', 'brick', 'hinge', 'none',
        'edges', and 'not-none' modules.

    Raises
    ------
    AssertionError
        If the number of core modules is not exactly 1.

    Notes
    -----
    - The 'not-none' count is the sum of core, brick, and hinge modules
    - Function assumes all nodes have a 'type' attribute in their data
    """
    result: dict[str, int] = {
        "core": sum(
            data["type"] == "CORE" for _, data in individual.nodes(data=True)
        ),
        "brick": sum(
            data["type"] == "BRICK" for _, data in individual.nodes(data=True)
        ),
        "hinge": sum(
            data["type"] == "HINGE" for _, data in individual.nodes(data=True)
        ),
        "none": sum(
            data["type"] == "NONE" for _, data in individual.nodes(data=True)
        ),
        "edges": len(individual.edges()),
    }
    result["not-none"] = result["core"] + result["brick"] + result["hinge"]
    return result


def analyze_mass(individual: DiGraph) -> NamedGraphPropertiesT[float]:
    """
    Calculate the total mass of a modular robot individual.

    Parameters
    ----------
    individual : DiGraph
        A directed graph representing the modular robot structure.

    Returns
    -------
    dict[str, float]
        Dictionary containing the total mass with key "mass".

    Notes
    -----
    - Mass calculation uses predefined constants: CORE_MASS, BRICK_MASS,
      ROTOR_MASS, and STATOR_MASS
    - Hinge mass is computed as sum of rotor and stator masses
    - Depends on analyze_module_counts() to get component counts
    """
    counts = analyze_module_counts(individual)
    core_mass = counts["core"] * CORE_MASS
    brick_tot_mass = counts["brick"] * BRICK_MASS
    hinge_tot_mass = counts["hinge"] * (ROTOR_MASS + STATOR_MASS)
    total_mass = core_mass + brick_tot_mass + hinge_tot_mass
    return {"mass": total_mass}


def analyze_json_hash(individual: DiGraph) -> NamedGraphPropertiesT[str]:
    """
    Compute a canonical hash for a directed graph based on its structure.

    Parameters
    ----------
    individual : DiGraph
        A NetworkX directed graph to analyze and hash.

    Returns
    -------
    dict[str, str]
        Dictionary containing the computed hash with key 'hash' and the
        hexadecimal hash string as value.

    Notes
    -----
    - Creates a canonical representation by sorting nodes and edges to
      ensure consistent hashing regardless of insertion order
    - Uses SHA-256 algorithm for cryptographic hash generation
    - Node and edge attributes are included in the hash computation
    """
    nodes = sorted([(n, dict(individual.nodes[n])) for n in individual.nodes()])
    edges = sorted([
        (u, v, dict(individual.edges[u, v])) for u, v in individual.edges()
    ])
    canonical = {"nodes": nodes, "edges": edges}
    hash_string = sha256(
        json.dumps(canonical, sort_keys=True).encode("utf-8"),
    ).hexdigest()
    return {"hash": hash_string}


def analyze_json_hash_no_id(individual: DiGraph) -> NamedGraphPropertiesT[str]:
    """
    Generate a canonical hash for a directed graph excluding node identifiers.

    Parameters
    ----------
    individual : DiGraph
        A NetworkX directed graph with node and edge data attributes.

    Returns
    -------
    dict[str, str]
        Dictionary containing the computed hash under the 'hash' key.

    Notes
    -----
    - Sorts nodes and edges by their JSON representation to ensure
      deterministic hashing regardless of graph traversal order
    - Uses SHA-256 algorithm for hash computation
    - Node IDs are excluded from the canonical representation, only node
      and edge data attributes are considered
    """
    nodes = sorted(
        [dict(data) for _, data in individual.nodes(data=True)],
        key=lambda d: json.dumps(d, sort_keys=True),
    )
    edges = sorted(
        [dict(data) for _, _, data in individual.edges(data=True)],
        key=lambda d: json.dumps(d, sort_keys=True),
    )
    canonical = {"nodes": nodes, "edges": edges}
    hash_string = sha256(
        json.dumps(canonical, sort_keys=True).encode("utf-8"),
    ).hexdigest()
    return {"hash_no_id": hash_string}


# -----------------------------------------------------------------
# TODO: @savio @sara


def analyze_branching(individual: DiGraph) -> NamedGraphPropertiesT[float]:
    """
    Measures the relative level of branching in the robot's morphology.

    Calculation Method: the number of modules that have 6 faces occupied/connected
    devided by the number of possible modules with 6 faces connected
    based of the robot size.

    DO NOT GIVE A GRAPH WITH NONE TYPES!!!
    """
    b = 0
    m = 0

    for node in individual.nodes():
        m += 1

        # 5 means all faces are connected since their back should always be connected
        if len(list(individual.successors(node))) == 5 and individual.nodes(data=True)[node]["type"] == "BRICK":
            b +=1
        if len(list(individual.successors(node))) == 6 and individual.nodes(data=True)[node]["type"] == "CORE":
            b +=1
    # a robot with less than 7 modules can never have a moduale with all their faces connected
    if m <7:
        return {"branching": 0.0}
    # max possible modules that could have 6 connected faces
    bmax = int((m-2)/5)
     
    return {"branching": b/bmax}


def analyze_number_of_limbs(
    individual: DiGraph,
) -> NamedGraphPropertiesT[float]:
    """
    measures the number of limbs relative to its size

    counts the number of limbs devided by the total possible limbs of a robot with m moduls
    """
    m = 0
    l = 0

    for node in individual.nodes():
        m += 1

        # counting the modules with no out going edges(aka the leafs)
        if len(list(individual.successors(node))) == 0 and individual.nodes(data=True)[node]["type"] != "CORE":
            l +=1

    # thoroetical maximum number of limbs considering robot size
    lmax = 4*int((m-8)/5) + (m-8) % 5 + 6 if m>7 else m-1
    if lmax<=0:
        return {"number_of_limbs": 0.0}
    return {"number_of_limbs": l/lmax}


def analyze_length_of_limbs(
    individual: DiGraph,
) -> NamedGraphPropertiesT[float]:
    """
    Measures the relative length of the limbs.

    Calculation Method: The number of components (excluding the core) that are only attached to two
    other components devided by the total amount of componets-2 (the thoretical maximum amount of compentents attached to 2 others based of size)
    """

    e = 0
    m = 0

    for node in individual.nodes():
        m += 1

        # 1 means 2 modules are connected to it since they are always connected in the back
        if len(list(individual.successors(node))) == 1 and individual.nodes(data=True)[node]["type"] != "CORE":
            e +=1
        
    if m <3:
        return {"length_of_limbs": 0.0}
    
    # thoretical maximum length of a limb of robot size m
    emax = m-2
    
    
    return {"length_of_limbs": e/emax}


def analyze_coverage(individual: DiGraph) -> NamedGraphPropertiesT[float]:
    """
    Measures how much of the morphology space is covered (M4).

    Calculation Method: The specific formula for 'Coverage' (M4) is not detailed in this source material,
    but it is noted that this descriptor ranges in value from 0 to 1 [1].
    Morphologies similar to snakes (like those predominant under S1 fitness) tended to have high coverage,
    covering the whole body area [5].
    """
    return {"coverage": 0.0}


def analyze_joints(individual: DiGraph) -> NamedGraphPropertiesT[float]:
    """
    Measures the number of effective joints in the morphology (M5).

    Calculation Method: This descriptor was reformulated for the study.
    The concept of an effective joint is defined by a joint module that is attached to any other module type [1].
    (Previously, attachment was required to be specifically to the core-component or a structural brick [1, 6].)
    This reformulation was done because robots were often observed developing limbs purely formed by a sequence of joints [1].
    The descriptor ranges in value from 0 to 1 [1].
    """
    j = 0
    for node, data in individual.nodes(data=True):
        if data.get("type") == "HINGE":
            # we check all neighbors
            neighbors = list(individual.predecessors(node)) + list(individual.successors(node))
            for n in neighbors:
                n_type = individual.nodes[n].get("type")
                if n_type != "HINGE":
                    j += 1
                    break

    jmax = sum(data.get("type") == "HINGE" for _, data in individual.nodes(data=True))
    joints_ratio = j / jmax if jmax > 0 else 0.0
    
    return {"joints": joints_ratio}

def give_dim(graph):
    """
    Giving the length width and height of the robotmeasured in nr of componets

    will find the length, width and heigt by looking how far its limbs stretches in opposit directions and adding them togther

    
    """
    w = [0]
    l = [0]
    h = [0]

    # finding the width, length and height
    w.append(find_length(graph, "FRONT", 0))
    w.append(find_length(graph, "BACK", 0))

    l.append(find_length(graph, "RIGHT", 0))
    l.append(find_length(graph, "LEFT", 0))

    h.append(find_length(graph, "TOP", 0))
    h.append(find_length(graph, "BOTTOM", 0))

    

    # connecting the 2 largest values from each list, subtracting 1 since the core got counted double
    cw = w[0] + w[1] - 1
    cl = l[0] + l[1] - 1
    ch = h[0] + h[1] - 1

    return (cw,cl,ch)


def find_length(graph, direction, node) -> int:
    """
    finds the length in any direction

    graph: the Digraph of nodes of the robot
    direction: direction is the face that it should be heading to, this is the local direction of the current node
    node: the current node that we want to know the length of in a direction

    method:
    1. check all nodes that the current node has an outgoing edge to (aka do step one for all child nodes)
    2. substract the size of the component if its not on the face we are intrested in, doubled if it is the complete opposite
    3. take the biggest length, add the current nodes size to it and return it
    """
    length = [0]
    faces = ['RIGHT', 'TOP', 'LEFT', 'BOTTOM']
    rot = dict(graph.nodes(data=True))[node]['rotation']
    component_size = 1 # usefull for later expansion if needed

    # oriantation fixing by changing direction, 45 degrees maybe needs to be expanded upon later using comonentsize
    if (direction != 'FRONT' and direction != 'BACK') and rot != "DEG_0":
        match rot: 
            case "DEG_90":
                direction = faces[(faces.index(direction)+1)%4]
            case "DEG_180":
                direction = faces[(faces.index(direction)+2)%4]
            case "DEG_270":
                direction = faces[(faces.index(direction)+3)%4]
            case "DEG_125":
                direction = faces[(faces.index(direction)+1)%4]
            case "DEG_225":
                direction = faces[(faces.index(direction)+2)%4]
            case "DEG_315":
                direction = faces[(faces.index(direction)+3)%4]
    

    for index, child in enumerate(graph.successors(node)):

        face = list(graph.edges(node, data=True))[index][2]["face"]


        # if this is in the right way keep going that way
        if face == direction:
            length.append(find_length(graph, 'FRONT', child))


        # completly the wrong way subtract 2 to penalize
        elif (direction =='LEFT' and face == 'RIGHT') or (direction =='RIGHT' and face == 'LEFT') \
            or (direction =='BOTTOM' and face == 'TOP') or (direction =='TOP' and face == 'BOTTOM'):

            length.append(find_length(graph, 'BACK', child)- 2) 


        # back means it needs to do an 180 or the same direction twice if its not the front
        elif direction == 'BACK' and face != 'FRONT':
            length.append(find_length(graph, face, child)-1)

        else:
            length.append(find_length(graph, 'RIGHT', child)-1)



    return max(length)+component_size

def analyze_proportion(individual: DiGraph) -> NamedGraphPropertiesT[float]:
    """
    Measures the proportionality or balance of the robot's shape (M6).

    Calculation Method: The specific formula for 'Proportion' (M6) is not detailed in this source material,
    but it is noted that this descriptor ranges in value from 0 to 1 [1].
    Proportion was observed to drop drastically for fitness S1, which was dominated by single-limb, disproportional robots [5].
    """
# w,l,h is the width, length and height hopefully this is helpfull for porortions
    w,l,h = give_dim(graph)
    print(w,l,h)
    


    return {"proportion": 0.0}


def analyze_symmetry(individual: DiGraph) -> NamedGraphPropertiesT[float]:
    """
    Measures the symmetry of the robot's structure (M7).

    Calculation Method: The specific formula for 'Symmetry' (M7) is not detailed in this source material,
    but it is noted that this descriptor ranges in value from 0 to 1 [1].
    Symmetry tended to be higher when a penalty for long limbs (S3 fitness) was applied [7].
    The results suggest that higher symmetry is correlated with lower average speed [8].
    The main idea is to approximate the morphological symmetry comparing the sizes of the subtrees attached.
    Formula used: symmetry = 1 - std(subtree_sizes) / max(subtree_sizes)

    """

    # Find the core
    core_nodes = [n for n, d in individual.nodes(data=True) if d.get("type") == "CORE"]
    core = core_nodes[0]
    # Find the neighbors
    neighbors = list(individual.successors(core)) + list(individual.predecessors(core))
    if len(neighbors) <= 1:
        return {"symmetry": 0.0}
    subtree_sizes = []
    visited_global = set([core])
    for n in neighbors:
        queue = [n]
        visited_local = set([core])
        count = 0
        while queue:
            node = queue.pop()
            if node in visited_local:
                continue
            visited_local.add(node)
            visited_global.add(node)
            count += 1
            for succ in individual.successors(node):
                if succ not in visited_local:
                    queue.append(succ)
        subtree_sizes.append(count)
    if not subtree_sizes:
        return {"symmetry": 0.0}
    max_size = max(subtree_sizes)
    if max_size == 0:
        return {"symmetry": 0.0}

    symmetry_value = 1 - (np.std(subtree_sizes) / max_size)
    symmetry_value = float(np.clip(symmetry_value, 0.0, 1.0))

    return {"symmetry": symmetry_value}


def analyze_size(individual: DiGraph) -> NamedGraphPropertiesT[float]:
    """
    Measures the overall size of the robot's morphology (M8).

    Calculation Method: The specific formula for 'Size' (M8) is not detailed in this source material,
    but it is noted that this descriptor ranges in value from 0 to 1 [1].
    All behavior-oriented searches tended to explore larger Size, as a large body can more easily produce a large displacement for high speed [5].
    """
    counts = analyze_module_counts(individual)
    not_none = counts["not-none"]
    max_size = 50  
    size_ratio = not_none / max_size if max_size > 0 else 0.0

    return {"size": float(min(1.0, size_ratio))}


def analyze_sensors(individual: DiGraph) -> NamedGraphPropertiesT[float]:
    """
    Measures the ratio of sensors to available slots in the morphology (M9).

    Calculation Method: This is a new descriptor introduced in the study [1]. It is defined by the equation [9]:

    C = { c / c_max, if c_max > 0
        { 0, otherwise

    Where 'c' is the number of sensors in the morphology, and 'c_max' is the number of free slots in the morphology [9].
    The descriptor ranges in value from 0 to 1 [1].
    """
    return {"sensors": 0.0}


if __name__ == "__main__":
    from ariel_experiments.utils.initialize import generate_random_individual

    graph = generate_random_individual()
    console.print("not none:", analyze_module_counts(graph)["not-none"])

    # feel free to test and expand here
