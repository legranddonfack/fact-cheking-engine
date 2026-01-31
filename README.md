# GradientSemantic
## How the Approach works:
1. If the type is inconsistent (TBox inconsistency)\
   If a derefied fact violates class hierarchy rules or property constraints, we should consider the hasTruthValue to be 0.0 and stop checking for that triplet.
2. If the type is consistent (TBox consistency)\
   If the triplet respects the hierarchy and domains/scopes, this means that the fact is possible, but not necessarily true. We continue the analysis in the reference graph:
   * Direct verification: We check whether the triplet exists as is in the graph. If it is present, the score is 1.0.
   * Proximity analysis (Random Walks): If the fact is not explicit, we use Random Walks to measure the ‘proximity’ between the subject and the object in the graph. Strong indirect connectivity increases the truthfulness score.
   * Semantic coherence: We check whether the two entities share common neighbours or types of links similar to those observed for other valid pairs in the training data.

3. Using Training Data for the final score\
   Training Data is used to calibrate our scoring function.
   * For consistent facts that are not in the graph, we will obtain indices (e.g. ‘distance of 2’, ‘typicality of 0.6’).
   * By observing the facts marked ‘1.0’ or ‘0.0’ in the training data, we determine the weights to give to each index. For example, we might learn that a fact with high ‘Support’ (the pattern appears often in the KG) is almost always true, even if the distance in the graph is large.
   Step-by-step summary of the algorithm:
   1. De-reify the fact to obtain (s, p, o).
   2. Test consistency (TBox): If violation (inconsistency) -> Score 0.0 (STOP).
   3. Check existence (ABox): If present in refKG.nt -> Score 1.0 (STOP).
   4. Calculate the indices (Features): If absent, calculate proximity, support and consistency via Jena/SPARQL queries.
   5. Weight (Training Data): Apply the weights learned on the training set to produce a final score between 0.0 and 1.0.

# How to run:
- Requirements: 
  * Java 17 or above
  * Maven 3.5.x

   ### To compile the project:
  - `mvn clean compile`
  ### To build a jar file:
  - `mvn clean package`\
  The generated jar file should be in the folder ./target
  - run `java -jar fact-checking-1.0-SNAPSHOT.jar ./data ./data/fokg-sw-train-2024.nt ./data/fokg-sw-test-2024.nt ./result.ttl`
  to execute, assuming the training and the test data are in the `./data` directory.
  - Place both `classHierarchy.nt` and `reference-kg.nt` in the <data_dir>