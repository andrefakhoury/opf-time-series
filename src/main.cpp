#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <climits>
#include <cfloat>
#include <numeric>
#include <iomanip>
#include <queue>
#include <cmath>
#include <fstream>
#include <sstream>
#include "dtw.hpp"
#include "utils.hpp"
#include "OPFClassifier.cpp"
using namespace std;

int main() {
	ios::sync_with_stdio(false); cin.tie(nullptr);

	vector<vector<double>> X, X_test;
	vector<int> y, y_test;

	readDF("DodgerLoopGame", X, y, X_test, y_test);

	OPFClassifier opf(0);
	opf.fit(X, y);

	auto preds = opf.classify(X_test);
	cout << fixed << setprecision(4) << error(preds, y_test) << "\n";
}
