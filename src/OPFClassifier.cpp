#ifndef OPF_HPP
#define OPF_HPP

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
#include <cassert>
#include "UCR_DTW.hpp"
#include "utils.hpp"
using namespace std;


class DSU {
	vector<int> par, sz;
public:
	DSU(int n) : par(n), sz(n) {
		iota(par.begin(), par.end(), 0);
		fill(sz.begin(), sz.end(), 1);
	}

	int find(int x) {
		return x == par[x] ? x : par[x] = find(par[x]);
	}

	bool same(int u, int v) {
		return find(u) == find(v);
	}

	void merge(int u, int v) {
		u = find(u), v = find(v);
		if (u == v) return;
		if (sz[u] > sz[v]) swap(u, v);
		par[u] = v;
		sz[v] += sz[u];
	}
};

struct OPFClassifier {
	vector<vector<double>> X;
	vector<int> label, orderedNodes, prototypes;
	vector<double> cost;
	enum ENUM_DISTANCE {
		EUCLIDEAN=0, DTW_DISTANCE=1, DTW_PRUNED=2 
	} F_DISTANCE;

	OPFClassifier(const string& distance_method="euclidean-distance") {
		if (distance_method == "dtw-distance") {
			F_DISTANCE = DTW_DISTANCE;
		} else if (distance_method == "dtw-pruned") {
			F_DISTANCE = DTW_PRUNED;
		} else if (distance_method == "euclidean-distance") {
			F_DISTANCE = EUCLIDEAN;
		} else {
			assert(0);
		}
	}

	OPFClassifier(int F_DIST) : F_DISTANCE{static_cast<ENUM_DISTANCE>(F_DIST)} {}

	double F(vector<double> const& x, vector<double> const& y) {
		if (F_DISTANCE == DTW_DISTANCE) return ucr_dtw(x, y, 0.1 * x.size());
		else if (F_DISTANCE == DTW_PRUNED) return ucr_dtw(x, y, 0.1 * x.size());
		else if (F_DISTANCE == EUCLIDEAN) return euclideanDistance(x, y);
		else assert(0);
	}

	vector<int> findPrototypes(vector<vector<pair<int, double>>> const& adj, vector<int> const& y) {
		const int n = adj.size();

		vector<bool> selected(n, false);
		vector<int> parent(n, -1);
		vector<double> cost(n, 1e18);
		cost[0] = 0;

		vector<int> prototypes;
		for (int it = 0; it < n; it++) {
			int v = -1;
			for (int j = 0; j < n; j++) {
				if (selected[j]) continue;
				if (v == -1 || cost[j] < cost[v])
					v = j;
			}

			selected[v] = true;
			if (parent[v] != -1 && y[v] != y[parent[v]]) {
				prototypes.push_back(parent[v]);
				prototypes.push_back(v);
			}
			for (auto [to, w] : adj[v]) {
				if (w < cost[to]) {
					cost[to] = w;
					parent[to] = v;
				}
			}
		}

		sort(prototypes.begin(), prototypes.end());
		prototypes.erase(unique(prototypes.begin(), prototypes.end()), prototypes.end());
		if (prototypes.empty())
			prototypes = {0};

		return prototypes;
	}

	void fit(vector<vector<double>> const& X_, vector<int> const& y) {
		const int n = y.size();
		X = X_;

		vector<vector<pair<int, double>>> adj(n, vector<pair<int, double>>(n));
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				adj[i][j] = {j, F(X[i], X[j])};
				adj[j][i] = {i, adj[i][j].second};
			}
		}

		prototypes = findPrototypes(adj, y);
		cost.assign(n, 1e20);
		label.assign(n, -1);
		
		priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
		for (auto u : prototypes) {
			cost[u] = 0;
			label[u] = y[u];
			pq.emplace(0, u);
		}
		
		while(!pq.empty()) {
			auto [d, u]	= pq.top();
			pq.pop();
			if (d > cost[u]) {
				continue;
			}

			for (auto [v, w] : adj[u]) {
				double newCost = max(cost[u], w);
				if (cost[v] > newCost) {
					cost[v] = newCost;
					label[v] = label[u];
					pq.emplace(w, v);
				}
			}
		}

		orderedNodes.assign(n, 0);
		iota(orderedNodes.begin(), orderedNodes.end(), 0);
		sort(orderedNodes.begin(), orderedNodes.end(), [&](int i, int j) {
			return cost[i] < cost[j];
		});		
	}

	int classifyOneVertex(vector<double> const& x) {
		int best = orderedNodes[0];
		double bestCost = max(cost[best], F(X[best], x));

		for (int i = 1; i < (int) X.size(); i++) {
			int curNode = orderedNodes[i];
			if (cost[curNode] > bestCost) {
				break;
			}

			double curCost = max(cost[curNode], F(X[curNode], x));
			if (curCost < bestCost) {
				best = curNode;
				bestCost = curCost;				
			}
		}

		return label[best];
	}

	vector<int> classify(vector<vector<double>> const& X) {
		vector<int> preds(X.size(), -1);
		for (int i = 0; i < (int) X.size(); i++) {
			preds[i] = classifyOneVertex(X[i]);
		}
		return preds;
	}
};

#endif