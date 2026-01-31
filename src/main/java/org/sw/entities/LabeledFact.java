package org.sw.entities;

import org.apache.jena.rdf.model.Property;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.rdf.model.Resource;

public class LabeledFact {
	private final ReifiedStatement statement;
	private final double truthValue;

	public LabeledFact(ReifiedStatement statement, double truthValue) {
		this.statement = statement;
		this.truthValue = truthValue;
	}

	public Resource getSubject() { return statement.getSubject(); }
	public Property getPredicate() { return statement.getPredicate(); }
	public RDFNode getObject() { return statement.getObject(); }
	public double getTruthValue() { return truthValue; }
}