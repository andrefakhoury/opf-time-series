#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cassert>
using namespace std;

double euclideanDistance(vector<double> const& x, vector<double> const& y) {
	assert(x.size() == y.size());
	double sum = 0;
	for (int i = 0; i < (int) x.size(); i++) {
		double d = x[i] - y[i];
		sum += d*d;
	}
	return sqrt(sum);		
}

void readTable(const string& filename, vector<vector<double>>& X, vector<int>& y) {
	ifstream f;
	f.open(filename);
	
	string line;
	while(getline(f, line)) {
		stringstream ss;
		ss << line;
		
		int label;
		ss >> label;
		y.push_back(label);

		X.emplace_back();

		double val;
		while (ss >> val) {
			X.back().push_back(val);
		}
	}

	assert(X.size() == y.size());
	for (int i = 0; i < (int) X.size(); i++) {
		assert(X[i].size() == X[0].size());
	}

	f.close();
}

void readDF(const string& dfName, vector<vector<double>>& X, vector<int>& y, vector<vector<double>>& X_test, vector<int>& y_test) {
	readTable("../data/UCRArchive_2018/" + dfName + "/" + dfName + "_TRAIN.tsv", X, y);
	readTable("../data/UCRArchive_2018/" + dfName + "/" + dfName + "_TEST.tsv", X_test, y_test);

	// coordinate-compression on y/y_test to [1..k]
	vector<int> labels = y;
	labels.insert(labels.begin(), y_test.begin(), y_test.end());
	sort(labels.begin(), labels.end());
	auto getLabel = [&](int l)->int {
		return lower_bound(labels.begin(), labels.end(), l) - labels.begin() + 1;
	};
	for (int& l : y) l = getLabel(l);
	for (int& l : y_test) l = getLabel(l);
}

double error(vector<int> const& a, vector<int> const& b) {
	int total = 0;
	for (int i = 0; i < (int) a.size(); i++) {
		total += a[i] == b[i];
	}
	return 1 - 1.0 * total / a.size();
}

#endif