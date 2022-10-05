#include <bits/stdc++.h>
using namespace std;

#define dist(x,y) ((x-y)*(x-y))
const double INF = 1e20;

double dtw(vector<double> const& A, vector<double> const& B, double w) {
	const int m = A.size();
	const int r = m * w;

    /// Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).
	vector<double> cost(2 * r + 1, INF), costPrev(2 * r + 1, INF);

	int k = 0;
	for (int i = 0; i < m; i++) {
		k = max(0, r-i);

		for (int j = max(0, i - r); j <= min(m - 1, i + r); j++, k++) {
			if (i == 0 && j == 0) {
				cost[k] = dist(A[0], B[0]);
				continue;
			}

			double x = INF, y = INF, z = INF;
			if (j >= 1 && k >= 1) y = cost[k-1];
			if (i >= 1 && k + 1 <= 2 * r) x = costPrev[k+1];
			if (i >= 1 && j >= 1) z = costPrev[k];
			cost[k] = min({x, y, z}) + dist(A[i], B[j]);
		}

        /// Move current array to previous array.
		swap(cost, costPrev);
    }

    /// the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
    return costPrev[k - 1];
}

double pruneddtw(vector<double> const& A, vector<double> const& B, double w) {
	const int m = A.size();
	const int r = m * w;

	vector<double> cost(2 * r + 1, INF), costPrev(2 * r + 1, INF);
	vector<double> ubPartials(m);

	for (int j = m - 1; j >= 0; j--) {
		if (j == m - 1) ubPartials[j] = dist(A[j], B[j]);
		else ubPartials[j] = ubPartials[j+1] + dist(A[j], B[j]);
	}

	double UB = ubPartials[0];
	int lp = -1, ec = 0, sc = 0;
	for (int i = 0; i < m; i++) {
		bool foundSC = false;
		bool prunnedEC = false;
		int nextEC = i + r + 1;
		int iniJ = max({0, i - r, sc});

		for (int j = iniJ; j <= min(m - 1, i + r); j++) {
			if (i == 0 && j == 0) {
				cost[j] = dist(A[0], B[0]);
				foundSC = true;
				continue;
			}

			double x = INF, y = INF, z = INF;
			if (j != iniJ) y = cost[j-1];
			if (i != 0 && j != i + r && j < lp) x = costPrev[j];
			if (i != 0 && j != 0 && j <= lp) z = costPrev[j-1];

			cost[j] = min({x, y, z}) + dist(A[i], B[j]);

			/// Pruning criteria
			if (!foundSC && cost[j] <= UB) {
				sc = j;
				foundSC = true;
			}
			
			if (cost[j] > UB) {				
				if (j > ec) {
					lp = j;
					prunnedEC = true;
					break;
				}
			} else {
				nextEC = j+1;
			}
        }
		UB = ubPartials[i+1] + cost[i];
		swap(cost, costPrev);
		
		/// Pruning statistics update
		if(sc > 0) costPrev[sc-1] = INF;
		if (!prunnedEC) lp = i + r + 1;
		ec = nextEC;
    }
	
    /// the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
	return costPrev[m-1];
}