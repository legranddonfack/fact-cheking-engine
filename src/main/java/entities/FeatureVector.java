package entities;

public class FeatureVector {
	public double predicateSupport;
	public double subjectConnectivity;
	public double objectConnectivity;
	public double rwProximity;
	public double pathDistance;
	public double semanticCoherence;
	public double typicality;

	public String predicateURI;

	@Override
	public String toString() {
		return "FeatureVector{" +
				"predicateSupport=" + predicateSupport +
				", subjectConnectivity=" + subjectConnectivity +
				", objectConnectivity=" + objectConnectivity +
				", rwProximity=" + rwProximity +
				", pathDistance=" + pathDistance +
				", semanticCoherence=" + semanticCoherence +
				", typicality=" + typicality +
				", predicateURI=" + predicateURI +
				'}';
	}
}