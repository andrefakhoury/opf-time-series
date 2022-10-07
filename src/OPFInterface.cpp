#include <string>
#include <iostream>
#include "OPFClassifier.cpp"

using namespace std;


struct OPFInterface {
	vector<int> *label, *orderedNodes, *prototypes;
	vector<double> *cost;
	vector<vector<double>> *X;
	int F_DISTANCE;
};

void py_to_cpp(OPFInterface* opfInter, OPFClassifier** opfClass) {
	*opfClass = new OPFClassifier(opfInter->F_DISTANCE);
	(*opfClass)->label = vector<int>(*(opfInter->label));
	(*opfClass)->orderedNodes = vector<int>(*(opfInter->orderedNodes));
	(*opfClass)->prototypes = vector<int>(*(opfInter->prototypes));
	(*opfClass)->cost = vector<double>(*(opfInter->cost));
	(*opfClass)->X = vector<vector<double>>(*(opfInter->X));
}

void cpp_to_py(OPFClassifier* opfClass, OPFInterface* opfInter) {
	opfInter->F_DISTANCE = opfClass->F_DISTANCE;
	opfInter->label = new vector<int>(opfClass->label);
	opfInter->orderedNodes = new vector<int>(opfClass->orderedNodes);
	opfInter->prototypes = new vector<int>(opfClass->prototypes);
	opfInter->cost = new vector<double>(opfClass->cost);
	opfInter->X = new vector<vector<double>>(opfClass->X);
}

extern "C" {

void init(char* s, OPFInterface* opf) {
	OPFClassifier opfClass(s);
	cpp_to_py(&opfClass, opf);
}

void fit(double* X_, int* y_, size_t X_n, size_t X_m, size_t y_n, OPFInterface* opf) {
	vector<vector<double>> X(X_n, vector<double>(X_m));
	vector<int> y(y_n);

	for (size_t i = 0; i < X_n; i++)
		for (size_t j = 0; j < X_m; j++)
			X[i][j] = X_[i * X_m + j];
	for (size_t i = 0; i < y_n; i++)
		y[i] = y_[i];

	OPFClassifier* opfClass;
	py_to_cpp(opf, &opfClass);
	opfClass->fit(X, y);
	cpp_to_py(opfClass, opf);
}

void classify(double* X_, size_t X_n, size_t X_m, int* y_, OPFInterface* opf) {
	vector<vector<double>> X(X_n, vector<double>(X_m));
	for (size_t i = 0; i < X_n; i++)
		for (size_t j = 0; j < X_m; j++)
			X[i][j] = X_[i * X_m + j];

	OPFClassifier* opfClass;
	py_to_cpp(opf, &opfClass);

	vector<int> y = opfClass->classify(X);


	copy(y.begin(), y.end(), y_);
	cpp_to_py(opfClass, opf);
}

size_t getPrototypeSize(OPFInterface* opf) {
	return opf->prototypes->size();
}

int getPrototypeAt(size_t i, OPFInterface* opf) {
	return opf->prototypes->at(i);
}

}
