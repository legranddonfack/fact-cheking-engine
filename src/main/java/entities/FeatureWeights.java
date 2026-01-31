package entities;

public class FeatureWeights {
	public double wSupport = 0.1;
	public double wSubjectConnectivity = 0.1;
	public double wObjectConnectivity = 0.1;
	public double wRWProximity = 0.2;
	public double wPathDistance = 0.2;
	public double wCoherence = 0.2;
	public double wTypicality = 0.1;

	public void normalize() {
		double sum = wSupport + wSubjectConnectivity + wObjectConnectivity +
				wRWProximity + wPathDistance + wCoherence + wTypicality;

		if (sum > 0) {
			wSupport /= sum;
			wSubjectConnectivity /= sum;
			wObjectConnectivity /= sum;
			wRWProximity /= sum;
			wPathDistance /= sum;
			wCoherence /= sum;
			wTypicality /= sum;
		}
	}

	@Override
	public String toString() {
		return "FeatureWeights{" +
				"wSupport=" + wSupport +
				", wSubjectConnectivity=" + wSubjectConnectivity +
				", wObjectConnectivity=" + wObjectConnectivity +
				", wRWProximity=" + wRWProximity +
				", wPathDistance=" + wPathDistance +
				", wCoherence=" + wCoherence +
				", wTypicality=" + wTypicality +
				'}';
	}
}