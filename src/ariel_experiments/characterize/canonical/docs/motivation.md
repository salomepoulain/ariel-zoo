Our diversity metric is in essence how literally morpholically different individuals are, which can be represented in a tree-like graph where the edges are labeled as the attachment-faces, and the nodes contain the moduletypes and rotation states.

However, comparing 2 graphs together is rather quite slow if something like tree-edit distance would be used (least amount of operations needed to turn graphs into each other). Aditionally, the morphological-to-tree mapping is a 1 to many relationship, as (for example) attaching something to a face with no internal rotation state, results in the same morphology as attaching it to a different face with an additional internal rotation state.

To tackle this problem, a **canonicalization** method has been impelemented that can take any tree-like graph, and give a standardized form, by minimzing the module rotation values. It does so from the root on.

In order to now compare graphs together, inspiration from chemistry was taken to divide an individual into (canonical) sub-graphs. Options for these subgraphs can either be starting from the root to the leaf, and canonicalizing each step. Or using a node-neighbouring approach, by collecting all (canonical) neighbourhoods from all the nodes. A canonical sub-graph is thus a way to get a 1:1 mapping from morphological body-part to graph.

By storing the subtrees using a string format [which may have its own grammar rules for easy human-readibility, also inspired by chemistry's canonical SMILES] a certain **tree-hash** **fingerprint** is obtained per individual. The fingerprint may be seen as an N dimensional **vector**, with N being the unique tree-hashes, and the values representing the counts.

This opens up **many** possibilities to cross-compare individuals or compare individuals against a population.

cross-comparison between the **fingerprint vectors** may be done using cosine similarity, or tanimoto approaches (intersection / union).

population-based approaches could either mean forming a symmetric matrix using the methods named above,
or collecting a treehash-archive from an entire population and then obtaining a **TF-IDF vector** for the fingerprint of an individual.

_TF-IDF is a score that reveals how "interesting" a word (treehash) is by balancing its frequency in a specific document (fingerprint) against its rarity everywhere else (archive). Hence, this is an interesting way to see if an individual contains novel fragments in its morphology_

Calculated numerical values from these TF-IDF vectors, may be obtained to use as a fitness score. examples are: l1 norm, mean, shannon-entropy. Information Theory can motivate why a certain choice would be made.

The TF-IDF method is also interesting for novelty-based EA's, as you can directly archive all tree-hashes that have been introduced across generations, and thus more easily reward individuals with 'novel' morphological parts. This method is also a lot cheaper computationally speaking compared to the matrix method.
