#include <iostream>
#include <bits/stdc++.h>

extern "C" {
	struct OPFClassifier {
		int x, y;
		size_t n;
		std::function<double(std::vector<double>, std::vector<double>)> fDist;
	};

	void init(OPFClassifier* opf, const std::string cost) {
		
	}

	void fit(double* X_, double* Y_) {
		std::vector<double> x, y;
	}

	void classify() {

	}
}