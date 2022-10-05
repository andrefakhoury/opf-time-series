#include <bits/stdc++.h>
#include "dtw.hpp"
using namespace std;

#define pb push_back
#define eb emplace_back
#define mk make_pair
#define fi first
#define se second
#define mset(a, b) memset(a, b, sizeof(a))
using ll = long long;
using pii = pair<int, int>;
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
template<class Ty> Ty randint(Ty a, Ty b) { return uniform_int_distribution<Ty>(a, b)(rng); }

template<class T> void DBG(T&& x) { cerr << x << ' '; }
template<class T, class...Args> void DBG(T&& x, Args&&... args) { DBG(x); DBG(args...); }
#define DBG(...) { cerr << "[" << #__VA_ARGS__ << "]: "; DBG(__VA_ARGS__); cerr << endl; }
template<class num> inline void rd(num& x) { cin >> x; }
template <class Ty, class... Args> inline void rd(Ty& x, Args&... args) { rd(x); rd(args...); }
template<class num> inline void print(num&& x) { cout << x; }
template <class Ty, class... Args> inline void print(Ty&& x, Args&&... args) { print(x); print(' '); print(args...); }
#define print(...) print(__VA_ARGS__), print('\n')

#define dist(x,y) ((x-y)*(x-y))

#define INF 1e20       //Pseudo Infitinte number for this code

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

double euclideanDistance(vector<double> const& x, vector<double> const& y) {
	double sum = 0;
	for (int i = 0; i < (int) x.size(); i++) {
		double d = x[i] - y[i];
		sum += d*d;
	}
	return sqrt(sum);		
}

class OPFClassifier {
private:
	vector<vector<double>> X;
	vector<int> label, orderedNodes;
	vector<double> cost;
	
	double F(vector<double> const& x, vector<double> const& y) {
		return pruneddtw(x, y, 1.0);
	}

public:
	OPFClassifier() {}

	vector<int> findPrototypes(vector<vector<pair<int, double>>> const& adj, vector<int> const& y) {
		const int n = adj.size();
		
		DSU uf(n);
		vector<tuple<double, int, int>> edges;
		for (int i = 0; i < n; i++) {
			for (auto& [u, w] : adj[i]) {
				edges.emplace_back(w, i, u);
			}
		}

		sort(edges.begin(), edges.end());

		vector<int> prototypes;
		for (auto& [w, u, v] : edges) {
			if (!uf.same(u, v)) {
				uf.merge(u, v);
				if (y[u] != y[v]) {
					prototypes.push_back(u);
					prototypes.push_back(v);
				}
			}
		}

		sort(prototypes.begin(), prototypes.end());
		prototypes.erase(unique(prototypes.begin(), prototypes.end()), prototypes.end());

		return prototypes;
	}

	void fit(vector<vector<double>> const& X_, vector<int> const& y) {
		const int n = y.size();
		X = X_;

		vector<vector<pair<int, double>>> adj(n);
		for (int i = 0; i < n; i++) {
			adj[i].resize(n);
			for (int j = 0; j < n; j++) {
				adj[i][j] = {j, F(X[i], X[j])};
			}
		}

		auto prototypes = findPrototypes(adj, y);
		cost.assign(n, DBL_MAX);
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

	f.close();
}

double error(vector<int> const& a, vector<int> const& b) {
	int total = 0;
	for (int i = 0; i < (int) a.size(); i++) {
		total += a[i] == b[i];
	}
	return 1 - 1.0 * total / a.size();
}

int main() {
	ios::sync_with_stdio(false); cin.tie(nullptr);

	vector<vector<double>> X, X_test;
	vector<int> y, y_test;
	readTable("../data/UCRArchive_2018/WordSynonyms/WordSynonyms_TRAIN.tsv", X, y);
	readTable("../data/UCRArchive_2018/WordSynonyms/WordSynonyms_TEST.tsv", X_test, y_test);

	OPFClassifier opf;
	opf.fit(X, y);

	auto preds = opf.classify(X_test);
	cout << fixed << setprecision(4) << error(preds, y_test) << "\n";
}
