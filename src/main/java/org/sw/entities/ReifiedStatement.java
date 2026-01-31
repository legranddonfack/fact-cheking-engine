package org.sw.entities;

import org.apache.jena.rdf.model.Property;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.rdf.model.Resource;

public class ReifiedStatement {
	private final Resource statementIRI;
	private final Resource subject;
	private final Property predicate;
	private final RDFNode object;

	public ReifiedStatement(Resource statementIRI, Resource subject,
							Property predicate, RDFNode object) {
		this.statementIRI = statementIRI;
		this.subject = subject;
		this.predicate = predicate;
		this.object = object;
	}

	public Resource getStatementIRI() { return statementIRI; }
	public Resource getSubject() { return subject; }
	public Property getPredicate() { return predicate; }
	public RDFNode getObject() { return object; }
}