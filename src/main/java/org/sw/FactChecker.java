package org.sw;

import org.sw.entities.DomainRangeInfo;
import org.sw.entities.FeatureVector;
import org.sw.entities.FeatureWeights;
import org.sw.entities.LabeledFact;
import org.sw.entities.ReifiedStatement;
import org.apache.jena.datatypes.RDFDatatype;
import org.apache.jena.datatypes.xsd.XSDDatatype;
import org.apache.jena.rdf.model.*;
import org.apache.jena.query.*;
import org.apache.jena.tdb2.TDB2Factory;
import org.apache.jena.dboe.base.file.Location;
import org.apache.jena.util.FileManager;
import org.apache.jena.vocabulary.RDF;
import org.apache.jena.vocabulary.RDFS;
import java.io.*;
import java.util.*;

public class FactChecker {

    static {
        // Initialize ARQ before ANY Jena/TDB operations
        ARQ.init();
    }

    // Constants
    private static final Property HAS_TRUTH_VALUE = ResourceFactory.createProperty(
        "http://swc2017.aksw.org/hasTruthValue");
    private static final RDFDatatype DOUBLE_TYPE = XSDDatatype.XSDdouble;
    
    private static final Property RDF_SUBJECT = RDF.subject;
    private static final Property RDF_PREDICATE = RDF.predicate;
    private static final Property RDF_OBJECT = RDF.object;
    private static final Resource RDF_STATEMENT = RDF.Statement;
    
    // Data
    private Dataset dataset;
    private Model referenceKG;
    private Model classHierarchy;
    
    // Caches for performance
    private Map<String, Set<String>> subclassCache = new HashMap<>();
    private Map<String, Set<String>> typeCache = new HashMap<>();
    private Map<String, DomainRangeInfo> domainRangeCache = new HashMap<>();

    // Store learned weights per predicate
    private Map<String, FeatureWeights> predicateWeights = new HashMap<>();
    private Map<String, Double> predicateBaselines = new HashMap<>();

    // Default weights for unseen predicates
    private FeatureWeights defaultWeights = new FeatureWeights();
    
    public FactChecker(String dataDir) {
        // Initialize TDB2 dataset
        Location location = Location.create(dataDir + "/tdb");
        dataset = TDB2Factory.connectDataset(location);
        
        // Load reference data
        loadReferenceData(dataDir);
        
        // Build caches
        buildSubclassCache();
        buildDomainRangeCache();
        initializeDefaultWeights();
    }

    private void initializeDefaultWeights() {
        defaultWeights = new FeatureWeights();
        defaultWeights.normalize();
    }
    
    private void loadReferenceData(String dataDir) {
        referenceKG = loadModel(dataDir + "/reference-kg.nt");
        classHierarchy = loadModel(dataDir + "/classHierarchy.nt");

        // Load into TDB
        dataset.begin(ReadWrite.WRITE);
        try {
            Model tdbModel = dataset.getDefaultModel();
            tdbModel.add(referenceKG);
            tdbModel.add(classHierarchy);
            dataset.commit();
        } finally {
            dataset.end();
        }

        System.out.println("Reference KG size: " + referenceKG.size() + " triples");
        System.out.println("Class hierarchy size: " + classHierarchy.size() + " triples");
    }

    /**
     *
     * @param filePath
     * @return
     */
    private Model loadModel(String filePath) {
        Model model = ModelFactory.createDefaultModel();
        FileManager.get().readModel(model, filePath);
        return model;
    }
    
    private void buildSubclassCache() {
        StmtIterator iter = classHierarchy.listStatements(null, RDFS.subClassOf, (RDFNode)null);
        while (iter.hasNext()) {
            Statement stmt = iter.next();
            String subclass = stmt.getSubject().getURI();
            String superclass = stmt.getObject().asResource().getURI();
            
            subclassCache.computeIfAbsent(subclass, k -> new HashSet<>()).add(superclass);
        }
    }
    
    private void buildDomainRangeCache() {
        // Extract domain/range from reference KG
        StmtIterator domainIter = referenceKG.listStatements(null, RDFS.domain, (RDFNode)null);
        while (domainIter.hasNext()) {
            Statement stmt = domainIter.next();
            String predicate = stmt.getSubject().getURI();
            String domain = stmt.getObject().asResource().getURI();
            
            DomainRangeInfo info = domainRangeCache.computeIfAbsent(predicate, 
                k -> new DomainRangeInfo());
            info.domains.add(domain);
        }
        
        StmtIterator rangeIter = referenceKG.listStatements(null, RDFS.range, (RDFNode)null);
        while (rangeIter.hasNext()) {
            Statement stmt = rangeIter.next();
            String predicate = stmt.getSubject().getURI();
            String range = stmt.getObject().asResource().getURI();
            
            DomainRangeInfo info = domainRangeCache.computeIfAbsent(predicate, 
                k -> new DomainRangeInfo());
            info.ranges.add(range);
        }
    }
    
    /**
     * Learn model weights from training data
     */
    public void train(String trainingFile) {
        Model trainingModel = loadModel(trainingFile);
        List<LabeledFact> labeledFacts = extractLabeledFacts(trainingModel);
        
        System.out.println("Training on " + labeledFacts.size() + " labeled facts");

        // Train predicate-specific models
        trainPerPredicate(labeledFacts);

        System.out.println("=== Training Complete ===");
        System.out.println("Trained models for " + predicateWeights.size() + " predicates");
        System.out.println("Default model ready for " + (labeledFacts.size() - countTrainedFacts(labeledFacts)) + " unseen predicates");
    }

    /**
     * Train separate model for each predicate
     */
    public void trainPerPredicatee(List<LabeledFact> trainingData) {
        System.out.println("Training per-predicate models...");

        // Group by predicate
        Map<String, List<LabeledFact>> byPredicate = new HashMap<>();
        for (LabeledFact fact : trainingData) {
            String predicateURI = fact.getPredicate().getURI();
            byPredicate.computeIfAbsent(predicateURI, k -> new ArrayList<>())
                    .add(fact);
        }

        System.out.println("Found " + byPredicate.size() + " distinct predicates");

        // Train model for each predicate with enough data
        for (Map.Entry<String, List<LabeledFact>> entry : byPredicate.entrySet()) {
            String predicateURI = entry.getKey();
            List<LabeledFact> predicateFacts = entry.getValue();

            if (predicateFacts.size() < 10) {
                // Too few examples, use default
                System.out.println("  " + predicateURI + ": Insufficient data (" +
                        predicateFacts.size() + " facts), using default");
                continue;
            }

            System.out.println("  Training " + predicateURI + " (" +
                    predicateFacts.size() + " facts)");

            // Extract features
            List<FeatureVector> vectors = new ArrayList<>();
            List<Double> truths = new ArrayList<>();

            for (LabeledFact fact : predicateFacts) {
                FeatureVector fv = extractFeatures(
                        fact.getSubject(),
                        fact.getPredicate(),
                        fact.getObject()
                );
                vectors.add(fv);
                truths.add(fact.getTruthValue());
            }

            // Learn weights
            FeatureWeights weights = learnWeightsGradient(vectors, truths);
            predicateWeights.put(predicateURI, weights);

            // Calculate baseline (average truth for this predicate)
            double baseline = truths.stream().mapToDouble(Double::doubleValue).average().orElse(0.5);
            predicateBaselines.put(predicateURI, baseline);

            System.out.println("    Baseline: " + String.format("%.3f", baseline) +
                    ", Learned weights: " + weights.toString());
        }
    }


    private void trainPerPredicate(List<LabeledFact> trainingData) {
        System.out.println("\n--- Training per-predicate models ---");

        // Group facts by predicate
        Map<String, List<LabeledFact>> factsByPredicate = new HashMap<>();
        for (LabeledFact fact : trainingData) {
            String predicateURI = fact.getPredicate().getURI();
            factsByPredicate.computeIfAbsent(predicateURI, k -> new ArrayList<>())
                    .add(fact);
        }

        System.out.println("Found " + factsByPredicate.size() + " distinct predicates");

        // Statistics
        int totalTrained = 0;
        int skippedDueToSize = 0;

        // Train model for each predicate
        for (Map.Entry<String, List<LabeledFact>> entry : factsByPredicate.entrySet()) {
            String predicateURI = entry.getKey();
            List<LabeledFact> predicateFacts = entry.getValue();

            // Skip predicates with insufficient data
            if (predicateFacts.size() < 20) {
                skippedDueToSize++;
                continue;
            }

            // Extract features for all facts with this predicate
            List<FeatureVector> featureVectors = new ArrayList<>();
            List<Double> truthValues = new ArrayList<>();

            int extracted = 0;
            for (LabeledFact fact : predicateFacts) {
                try {
                    FeatureVector features = extractFeatures(
                            fact.getSubject(),
                            fact.getPredicate(),
                            fact.getObject()
                    );
                    featureVectors.add(features);
                    truthValues.add(fact.getTruthValue());
                    extracted++;

                    // Progress indicator
                    if (extracted % 100 == 0) {
                        System.out.printf("Extracted features for %d/%d facts\n",
                                extracted, predicateFacts.size());
                    }
                } catch (Exception e) {
                    System.err.println("Error extracting features for fact: " + e.getMessage());
                }
            }

            if (extracted < 10) {
                System.out.printf("Too few valid features (%d), skipping\n", extracted);
                skippedDueToSize++;
                continue;
            }

            // Learn optimal weights for this predicate
            FeatureWeights weights = learnWeightsGradient(featureVectors, truthValues);

            // Calculate baseline (average truth for this predicate)
            double baseline = truthValues.stream().mapToDouble(Double::doubleValue).average().orElse(0.5);

            // Store the learned model
            predicateWeights.put(predicateURI, weights);
            predicateBaselines.put(predicateURI, baseline);

            totalTrained++;

            System.out.printf("Trained: baseline=%.3f, weights=%s\n",
                    baseline, weights.toString());
        }

        System.out.println("Training Summary:");
        System.out.println("Predicates trained: " + totalTrained);
        System.out.println("Predicates skipped (insufficient data): " + skippedDueToSize);
        System.out.println("Total predicates processed: " + factsByPredicate.size());
    }

    /**
     *
     * @param subject
     * @param predicate
     * @param object
     * @return
     */
    public double checkFact(Resource subject, Property predicate, RDFNode object) {
        // Check TBox consistency
        if (!isTBoxConsistent(subject, predicate, object)) {
            return 0.0; // STOP: Semantic inconsistency
        }
        
        // Check existence in ABox
        if (existsInKG(subject, predicate, object)) {
            return 1.0; // STOP: Explicit fact
        }
        
        // Extract features for analysis
        FeatureVector features = extractFeatures(subject, predicate, object);

        // Calculate final score with learned model
        return calculateScoreWithModel(features, predicate);
    }

    /**
     * TBox consistency check (types, domains, ranges)
     * @param subject
     * @param predicate
     * @param object
     * @return
     */
    private boolean isTBoxConsistent(Resource subject, Property predicate, RDFNode object) {
        // 1. Check if it's a type assertion
        if (predicate.equals(RDF.type)) {
            return isValidType(subject, object);
        }
        
        // 2. Check domains
        if (!checkDomains(subject, predicate)) {
            return false;
        }
        
        // 3. Check ranges (if object is resource)
        if (object.isResource()) {
            if (!checkRanges(object.asResource(), predicate)) {
                return false;
            }
        }
        
        return true;
    }
    
    private boolean isValidType(Resource subject, RDFNode type) {
        if (!type.isResource()) {
            return false; // Type must be a resource
        }
        
        String typeURI = type.asResource().getURI();
        Set<String> subjectTypes = getTypesWithInheritance(subject);
        
        // Exact type found
        if (subjectTypes.contains(typeURI)) {
            return true;
        }
        
        // Check inheritance: is subject subclass of type?
        for (String subjectType : subjectTypes) {
            if (isSubclassOf(subjectType, typeURI)) {
                return true;
            }
        }
        
        return false;
    }
    
    private boolean checkDomains(Resource subject, Property predicate) {
        DomainRangeInfo info = domainRangeCache.get(predicate.getURI());
        if (info == null || info.domains.isEmpty()) {
            return true; // No domain constraint
        }
        
        Set<String> subjectTypes = getTypesWithInheritance(subject);
        if (subjectTypes.isEmpty()) {
            return false; // Cannot check without types
        }
        
        // Check if any subject type satisfies a domain
        for (String domain : info.domains) {
            for (String subjectType : subjectTypes) {
                if (subjectType.equals(domain) || isSubclassOf(subjectType, domain)) {
                    return true;
                }
            }
        }
        
        return false; // No type satisfies domains
    }
    
    private boolean checkRanges(Resource object, Property predicate) {
        DomainRangeInfo info = domainRangeCache.get(predicate.getURI());
        if (info == null || info.ranges.isEmpty()) {
            return true; // No range constraint
        }
        
        Set<String> objectTypes = getTypesWithInheritance(object);
        if (objectTypes.isEmpty()) {
            return false; // Cannot check without types
        }
        
        // Check if any object type satisfies a range
        for (String range : info.ranges) {
            for (String objectType : objectTypes) {
                if (objectType.equals(range) || isSubclassOf(objectType, range)) {
                    return true;
                }
            }
        }
        
        return false; // No type satisfies ranges
    }

    /**
     * Check existence in ABox (reference KG)
     * @param subject
     * @param predicate
     * @param object
     * @return
     */
    private boolean existsInKG(Resource subject, Property predicate, RDFNode object) {
        String query = buildASKQuery(subject, predicate, object);
        dataset.begin(ReadWrite.READ);
        try (QueryExecution qexec = QueryExecutionFactory.create(query, dataset)) {
            return qexec.execAsk();
        } finally {
            dataset.end();
        }
    }
    
    private String buildASKQuery(Resource subject, Property predicate, RDFNode object) {
        if (object.isLiteral()) {
            return String.format(
                "ASK { <%s> <%s> \"%s\" }",
                subject.getURI(),
                predicate.getURI(),
                object.asLiteral().getLexicalForm()
            );
        } else {
            return String.format(
                "ASK { <%s> <%s> <%s> }",
                subject.getURI(),
                predicate.getURI(),
                object.asResource().getURI()
            );
        }
    }

    /**
     * Extract features for analysis
     * @param subject
     * @param predicate
     * @param object
     * @return
     */
    private FeatureVector extractFeatures(Resource subject, Property predicate, RDFNode object) {
        FeatureVector features = new FeatureVector();
        
        // 1. Predicate support (frequency in KG)
        features.predicateSupport = calculatePredicateSupport(predicate);
        
        // 2. Entity connectivity
        features.subjectConnectivity = calculateConnectivity(subject);
        if (object.isResource()) {
            features.objectConnectivity = calculateConnectivity(object.asResource());
        }
        
        // 3. Random Walk proximity
        if (object.isResource()) {
            features.rwProximity = calculateRandomWalkProximity(subject, object.asResource());
        }
        
        // 4. Graph distance (shortest path)
        if (object.isResource()) {
            features.pathDistance = calculatePathDistance(subject, object.asResource());
        }
        
        // 5. Semantic coherence (common neighbors)
        if (object.isResource()) {
            features.semanticCoherence = calculateSemanticCoherence(subject, object.asResource());
        }
        
        // 6. Typicality (type similarity)
        features.typicality = calculateTypicality(subject);
        
        return features;
    }
    
    private double calculatePredicateSupport(Property predicate) {
        String query = String.format(
            "SELECT (COUNT(*) as ?count) WHERE { ?s <%s> ?o }",
            predicate.getURI()
        );

        dataset.begin(ReadWrite.READ);
        try (QueryExecution qexec = QueryExecutionFactory.create(query, dataset)) {
            ResultSet results = qexec.execSelect();
            if (results.hasNext()) {
                int count = results.next().getLiteral("count").getInt();
                // Normalize: log(count+1) / log(max+1)
                return Math.log(count + 1) / Math.log(1000 + 1); // max estimated 1000
            }
        } finally {
            dataset.end();
        }
        return 0.0;
    }
    
    private double calculateConnectivity(Resource entity) {
        String query = String.format(
            "SELECT (COUNT(*) as ?count) WHERE { { <%s> ?p ?o } UNION { ?s ?p <%s> } }",
            entity.getURI(), entity.getURI()
        );

        dataset.begin(ReadWrite.READ);
        try (QueryExecution qexec = QueryExecutionFactory.create(query, dataset)) {
            ResultSet results = qexec.execSelect();
            if (results.hasNext()) {
                int count = results.next().getLiteral("count").getInt();
                // Logarithmic normalization
                return Math.log(count + 1) / Math.log(10000 + 1);
            }
        } finally {
            dataset.end();
        }
        return 0.0;
    }
    
    private double calculateRandomWalkProximity(Resource subject, Resource object) {
        // Simulate random walks of length 2
        String query = String.format(
            "SELECT (COUNT(*) as ?count) WHERE {\n" +
            "  <%s> ?p1 ?intermediate .\n" +
            "  ?intermediate ?p2 <%s> .\n" +
            "}",
            subject.getURI(), object.getURI()
        );

        dataset.begin(ReadWrite.READ);
        try (QueryExecution qexec = QueryExecutionFactory.create(query, dataset)) {
            ResultSet results = qexec.execSelect();
            if (results.hasNext()) {
                int count = results.next().getLiteral("count").getInt();
                // Normalize
                return Math.min(1.0, count / 10.0);
            }
        } finally {
            dataset.end();
        }
        return 0.0;
    }
    
    private double calculatePathDistance(Resource subject, Resource object) {
        // Find shortest paths of length 1, 2, 3
        for (int length = 1; length <= 3; length++) {
            if (existsPathOfLength(subject, object, length)) {
                // Shorter path = higher score
                return 1.0 / length;
            }
        }
        return 0.0;
    }
    
    private boolean existsPathOfLength(Resource source, Resource target, int length) {
        if (length == 1) {
            return existsDirectRelation(source, target);
        }
        
        // For length > 1, approximate recursive query
        String query = buildPathQueryOfLength(source, target, length);
        dataset.begin(ReadWrite.READ);
        try (QueryExecution qexec = QueryExecutionFactory.create(query, dataset)) {
            ResultSet results = qexec.execSelect();
            return results.hasNext();
        } finally {
            dataset.end();
        }
    }
    
    private boolean existsDirectRelation(Resource subject, Resource object) {
        String query = String.format(
            "ASK { { <%s> ?p <%s> } UNION { <%s> ?p <%s> } }",
            subject.getURI(), object.getURI(), object.getURI(), subject.getURI()
        );

        dataset.begin(ReadWrite.READ);
        try (QueryExecution qexec = QueryExecutionFactory.create(query, dataset)) {
            return qexec.execAsk();
        } finally {
            dataset.end();
        }
    }
    
    private String buildPathQueryOfLength(Resource source, Resource target, int length) {
        StringBuilder sb = new StringBuilder();
        sb.append("SELECT ?node0");
        
        for (int i = 1; i < length; i++) {
            sb.append(" ?node").append(i);
        }
        
        sb.append(" WHERE {\n");
        sb.append("  <").append(source.getURI()).append("> ?p0 ?node0 .\n");
        
        for (int i = 0; i < length - 2; i++) {
            sb.append("  ?node").append(i).append(" ?p").append(i+1).append(" ?node").append(i+1).append(" .\n");
        }
        
        sb.append("  ?node").append(length-2).append(" ?p").append(length-1)
          .append(" <").append(target.getURI()).append("> .\n");
        sb.append("}");
        
        return sb.toString();
    }
    
    private double calculateSemanticCoherence(Resource subject, Resource object) {
        // Count common neighbors
        String query = String.format(
            "SELECT (COUNT(DISTINCT ?neighbor) as ?count) WHERE {\n" +
            "  { <%s> ?p1 ?neighbor . <%s> ?p2 ?neighbor }\n" +
            "  UNION\n" +
            "  { ?neighbor ?p1 <%s> . ?neighbor ?p2 <%s> }\n" +
            "}",
            subject.getURI(), object.getURI(), subject.getURI(), object.getURI()
        );

        dataset.begin(ReadWrite.READ);
        try (QueryExecution qexec = QueryExecutionFactory.create(query, dataset)) {
            ResultSet results = qexec.execSelect();
            if (results.hasNext()) {
                int count = results.next().getLiteral("count").getInt();
                return Math.min(1.0, count / 5.0);
            }
        } finally {
            dataset.end();
        }
        return 0.0;
    }
    
    private double calculateTypicality(Resource subject) {
        // Check if this predicate-type pattern is common
        Set<String> subjectTypes = getTypesWithInheritance(subject);
        
        if (subjectTypes.isEmpty()) {
            return 0.5; // Default value
        }
        
        // Simplified: more typed org.sw.entities = more typical
        return Math.min(1.0, subjectTypes.size() / 10.0);
    }

    /**
     * Calculate score with learned model
     * @param features
     * @param predicate
     * @return
     */
    private double calculateScoreWithModel(FeatureVector features, Property predicate) {
        // Get weights for this specific predicate
        FeatureWeights weights;
        if (predicateWeights.containsKey(predicate.getURI())) {
            weights = predicateWeights.get(predicate.getURI());
            System.out.println("Using trained weights for predicate: " + predicate.getURI());
        } else {
            // Use default weights for unseen predicates
            weights = new FeatureWeights(); // default weights
            System.out.println("Warning: Using default weights for predicate: " +  predicate.getURI());
        }

        // Base score for predicate
        double baseScore = predicateBaselines.getOrDefault(predicate.getURI(), 0.5);
        // Weighted linear combination
        double score = baseScore;
        score += features.predicateSupport * weights.wSupport;
        score += features.subjectConnectivity * weights.wSubjectConnectivity;
        score += features.objectConnectivity * weights.wObjectConnectivity;
        score += features.rwProximity * weights.wRWProximity;
        score += features.pathDistance * weights.wPathDistance;
        score += features.semanticCoherence * weights.wCoherence;
        score += features.typicality * weights.wTypicality;

        // optional: Apply sigmoid for [0,1]
        // return sigmoid(score);
        return score <= 0 ? 0.0 : score > 1.0 ? 1.0 : score;
    }
    
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Gradient descent learning
     * @param vectors
     * @param truths
     * @return
     */
    private FeatureWeights learnWeightsGradient(List<FeatureVector> vectors, List<Double> truths) {
        int nFeatures = 7; // Number of features
        double learningRate = 0.01;
        int iterations = 500;
        
        // Initialize weights
        FeatureWeights weights = new FeatureWeights();
        
        for (int iter = 0; iter < iterations; iter++) {
            double totalError = 0;
            double[] gradient = new double[nFeatures];
            
            for (int i = 0; i < vectors.size(); i++) {
                FeatureVector f = vectors.get(i);
                double prediction = predictWithWeights(f, weights, 0.5);
                double error = prediction - truths.get(i);
                totalError += error * error;
                
                // Calculate gradient
                int idx = 0;
                gradient[idx++] += error * f.predicateSupport;
                gradient[idx++] += error * f.subjectConnectivity;
                gradient[idx++] += error * f.objectConnectivity;
                gradient[idx++] += error * f.rwProximity;
                gradient[idx++] += error * f.pathDistance;
                gradient[idx++] += error * f.semanticCoherence;
                gradient[idx++] += error * f.typicality;
            }
            
            // Update weights
            weights.wSupport -= learningRate * gradient[0] / vectors.size();
            weights.wSubjectConnectivity -= learningRate * gradient[1] / vectors.size();
            weights.wObjectConnectivity -= learningRate * gradient[2] / vectors.size();
            weights.wRWProximity -= learningRate * gradient[3] / vectors.size();
            weights.wPathDistance -= learningRate * gradient[4] / vectors.size();
            weights.wCoherence -= learningRate * gradient[5] / vectors.size();
            weights.wTypicality -= learningRate * gradient[6] / vectors.size();
            
            if (iter % 100 == 0) {
                System.out.printf("Iteration %d, Error: %.6f\n", iter, totalError / vectors.size());
            }
        }

        // Store learned weights (for all predicates)
        weights.normalize();
        return weights;
    }
    
    private double predictWithWeights(FeatureVector f, FeatureWeights w, double baseline) {
        double score = baseline;
        score += f.predicateSupport * w.wSupport;
        score += f.subjectConnectivity * w.wSubjectConnectivity;
        score += f.objectConnectivity * w.wObjectConnectivity;
        score += f.rwProximity * w.wRWProximity;
        score += f.pathDistance * w.wPathDistance;
        score += f.semanticCoherence * w.wCoherence;
        score += f.typicality * w.wTypicality;
        
        // alternative: return sigmoid(score);
        return score <= 0 ? 0.0 : score > 1.0 ? 1.0 : score;
    }

    /**
     * Predict score using predicate-specific model
     */
    public double predictWithPredicateModel(FeatureVector features, String predicateURI) {
        FeatureWeights weights;
        double baseline;

        if (predicateWeights.containsKey(predicateURI)) {
            weights = predicateWeights.get(predicateURI);
            baseline = predicateBaselines.get(predicateURI);
        } else {
            weights = defaultWeights;
            baseline = 0.5; // Default baseline
            System.out.println("Using default model for: " + predicateURI);
        }

        return predictWithWeights(features, weights, baseline);
    }

    private Set<String> getTypesWithInheritance(Resource resource) {
        String uri = resource.getURI();
        
        if (typeCache.containsKey(uri)) {
            return typeCache.get(uri);
        }
        
        Set<String> allTypes = new HashSet<>();
        
        // Direct types
        String query = String.format(
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n" +
                "SELECT ?type WHERE {\n" +
                "  <%s> rdf:type ?type .\n" +
                "}\n",
            uri
        );
        dataset.begin(ReadWrite.READ);
        try (QueryExecution qexec = QueryExecutionFactory.create(query, dataset)) {
            ResultSet results = qexec.execSelect();
            while (results.hasNext()) {
                String type = results.next().getResource("type").getURI();
                allTypes.add(type);
                addSuperclasses(type, allTypes);
            }
        } finally {
            dataset.end();
        }
        
        typeCache.put(uri, allTypes);
        return allTypes;
    }
    
    private void addSuperclasses(String type, Set<String> types) {
        Set<String> superclasses = subclassCache.get(type);
        if (superclasses != null) {
            for (String superclass : superclasses) {
                if (types.add(superclass)) {
                    addSuperclasses(superclass, types);
                }
            }
        }
    }
    
    private boolean isSubclassOf(String subclass, String superclass) {
        if (subclass.equals(superclass)) return true;
        
        Set<String> directParents = subclassCache.get(subclass);
        if (directParents == null) return false;
        
        if (directParents.contains(superclass)) return true;
        
        for (String parent : directParents) {
            if (isSubclassOf(parent, superclass)) {
                return true;
            }
        }
        
        return false;
    }

    /**
     * Process test file and generate results
     * @param testFile
     * @param outputFile
     * @throws IOException
     */
    public void processTestFile(String testFile, String outputFile) throws IOException {
        Model testModel = loadModel(testFile);
        List<ReifiedStatement> statements = extractReifiedStatements(testModel);
        
        try (FileWriter writer = new FileWriter(outputFile)) {
            for (ReifiedStatement stmt : statements) {
                System.out.print("checking statement " + stmt.getSubject().getURI());
                System.out.print(" " + stmt.getPredicate().getURI());
                System.out.println(" " + stmt.getObject().asResource().getURI());
                double truthValue = checkFact(
                    stmt.getSubject(),
                    stmt.getPredicate(),
                    stmt.getObject()
                );
                writer.write(String.format("<%s> <%s> \"%.1f\"^^<%s> .\n",
                    stmt.getStatementIRI().getURI(),
                    HAS_TRUTH_VALUE.getURI(),
                    truthValue,
                    DOUBLE_TYPE.getURI()));
            }
        }
    }

    /**
     * De-reification
     * @param model
     * @return List<ReifiedStatement>
     */
    private List<ReifiedStatement> extractReifiedStatements(Model model) {
        List<ReifiedStatement> statements = new ArrayList<>();
        ResIterator iter = model.listSubjectsWithProperty(RDF.type, RDF_STATEMENT);
        
        while (iter.hasNext()) {
            Resource stmtResource = iter.next();
            Statement subj = stmtResource.getProperty(RDF_SUBJECT);
            Statement pred = stmtResource.getProperty(RDF_PREDICATE);
            Statement obj = stmtResource.getProperty(RDF_OBJECT);
            
            if (subj != null && pred != null && obj != null) {
                statements.add(new ReifiedStatement(
                    stmtResource,
                    subj.getObject().asResource(),
                    pred.getObject().as(Property.class),
                    obj.getObject()
                ));
            }
        }
        return statements;
    }

    /**
     * Extract facts from training data model essentially
     * @param model
     * @return
     */
    private List<LabeledFact> extractLabeledFacts(Model model) {
        List<LabeledFact> facts = new ArrayList<>();
        List<ReifiedStatement> statements = extractReifiedStatements(model);
        
        for (ReifiedStatement stmt : statements) {
            Statement truthStmt = stmt.getStatementIRI().getProperty(HAS_TRUTH_VALUE);
            if (truthStmt != null) {
                facts.add(new LabeledFact(stmt, truthStmt.getDouble()));
            }
        }
        return facts;
    }

    private int countTrainedFacts(List<LabeledFact> facts) {
        int count = 0;
        for (LabeledFact fact : facts) {
            if (predicateWeights.containsKey(fact.getPredicate().getURI())) {
                count++;
            }
        }
        return count;
    }

}