#ifndef UCR_DTW_HPP
#define UCR_DTW_HPP

#include <numeric>
#include <vector>
#include <deque>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <iomanip>
using namespace std;

#define dist(x,y) ((x-y)*(x-y))
const double INF = 1e20;

/// Finding the envelop of min and max value for LB_Keogh
/// Implementation idea is intoruduced by Danial Lemire in his paper
/// "Faster Retrieval with a Two-Pass Dynamic-Time-Warping Lower Bound", Pattern Recognition 42(9), 2009.
void lower_upper_lemire(vector<double> const& t, int len, int r, vector<double>& l, vector<double>& u) {
	deque<int> du, dl;
	du.push_back(0);
	dl.push_back(0);

	for (int i = 1; i < len; i++) {
		if (i > r) {
			u[i-r-1] = t[du.front()];
			l[i-r-1] = t[dl.front()];
		}
		if (t[i] > t[i-1]) {
			du.pop_back();
			while (!du.empty() && t[i] > t[du.back()])
				du.pop_back();
		} else {
			dl.pop_back();
			while (!dl.empty() && t[i] < t[dl.back()])
				dl.pop_back();
		}
		du.push_back(i);
		dl.push_back(i);
		if (i == 2 * r + 1 + du.front())
			du.pop_front();
		else if (i == 2 * r + 1 + dl.front())
			dl.pop_front();
	}
	for (int i = len; i < len+r+1; i++) {
		u[i-r-1] = t[du.front()];
		l[i-r-1] = t[dl.front()];
		if (i-du.front() >= 2 * r + 1)
			du.pop_front();
		if (i-dl.front() >= 2 * r + 1)
			dl.pop_front();
	}
}

/// Calculate quick lower bound
/// Usually, LB_Kim take time O(m) for finding top,bottom,fist and last.
/// However, because of z-normalization the top and bottom cannot give siginifant benefits.
/// And using the first and last points can be computed in constant time.
/// The prunning power of LB_Kim is non-trivial, especially when the query is not long, say in length 128.
double lb_kim_hierarchy(vector<double>& t, vector<double> const& q, int j, int len, double mean, double std, double bsf = INF) {
	/// 1 point at front and back
	double d, lb;
	double x0 = (t[j] - mean) / std;
	double y0 = (t[(len-1+j)] - mean) / std;
	lb = dist(x0,q[0]) + dist(y0,q[len-1]);
	if (lb >= bsf)   return lb;

	/// 2 points at front
	double x1 = (t[(j+1)] - mean) / std;
	d = min(dist(x1,q[0]), dist(x0, q[1]));
	d = min(d, dist(x1, q[1]));
	lb += d;
	if (lb >= bsf)   return lb;

	/// 2 points at back
	double y1 = (t[(len-2+j)] - mean) / std;
	d = min(dist(y1,q[len-1]), dist(y0, q[len-2]) );
	d = min(d, dist(y1,q[len-2]));
	lb += d;
	if (lb >= bsf)   return lb;

	/// 3 points at front
	double x2 = (t[(j+2)] - mean) / std;
	d = min(dist(x0,q[2]), dist(x1, q[2]));
	d = min(d, dist(x2,q[2]));
	d = min(d, dist(x2,q[1]));
	d = min(d, dist(x2,q[0]));
	lb += d;
	if (lb >= bsf)   return lb;

	/// 3 points at back
	double y2 = (t[(len-3+j)] - mean) / std;
	d = min(dist(y0,q[len-3]), dist(y1, q[len-3]));
	d = min(d, dist(y2,q[len-3]));
	d = min(d, dist(y2,q[len-2]));
	d = min(d, dist(y2,q[len-1]));
	lb += d;

	return lb;
}

/// LB_Keogh 1: Create Envelop for the query
/// Note that because the query is known, envelop can be created once at the begenining.
///
/// Variable Explanation,
/// order : sorted indices for the query.
/// uo, lo: upper and lower envelops for the query, which already sorted.
/// t     : a circular array keeping the current data.
/// j     : index of the starting location in t
/// cb    : (output) current bound at each position. It will be used later for early abandoning in DTW.
double lb_keogh_cumulative(vector<int>& order, vector<double>& t, vector<double>& uo, vector<double>& lo, vector<double>& cb, int j, int len, double mean, double std, double best_so_far = INF) {
	double lb = 0;
	for (int i = 0; i < len && lb < best_so_far; i++) {
		double x = (t[(order[i]+j)] - mean) / std;
		double d = 0;
		if (x > uo[i])
			d = dist(x,uo[i]);
		else if(x < lo[i])
			d = dist(x,lo[i]);
		lb += d;
		cb[order[i]] = d;
	}
	return lb;
}

/// LB_Keogh 2: Create Envelop for the data
/// Note that the envelops have been created (in main function) when each data point has been read.
///
/// Variable Explanation,
/// tz: Z-normalized data
/// qo: sorted query
/// cb: (output) current bound at each position. Used later for early abandoning in DTW.
/// l,u: lower and upper envelop of the current data
double lb_keogh_data_cumulative(vector<int>& order, vector<double>& tz, vector<double>& qo, vector<double>& cb, vector<double>& l, vector<double>& u, int I, int len, double mean, double std, double best_so_far = INF) {
	double lb = 0;
	for (int i = 0; i < len && lb < best_so_far; i++) {
		double uu = (u[I+order[i]]-mean)/std;
		double ll = (l[I+order[i]]-mean)/std;
		double d = 0;
		
		if (qo[i] > uu)
			d = dist(qo[i], uu);
		else if(qo[i] < ll)
			d = dist(qo[i], ll);

		lb += d;
		cb[order[i]] = d;
	}
	return lb;
}

/// Calculate Dynamic Time Wrapping distance
/// A,B: data and query, respectively
/// cb : cummulative bound used for early abandoning
/// r  : size of Sakoe-Chiba warpping band

double dtw(vector<double> const& A, vector<double> const& B, vector<double>& cb, int m, int r, double bsf = INF) {
	/// Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).
	vector<double> cost(2 * r + 1, INF), costPrev(2 * r + 1, INF);

	int k = 0;
	for (int i = 0; i < m; i++) {
		k = max(0, r-i);
		double min_cost = INF;

		for (int j = max(0, i - r); j <= min(m - 1, i + r); j++, k++) {
			if (i == 0 && j == 0) {
				cost[k] = dist(A[0], B[0]);
				min_cost = cost[k];
				continue;
			}

			double x = INF, y = INF, z = INF;
			if (j >= 1 && k >= 1) y = cost[k-1];
			if (i >= 1 && k + 1 <= 2 * r) x = costPrev[k+1];
			if (i >= 1 && j >= 1) z = costPrev[k];
			cost[k] = min({x, y, z}) + dist(A[i], B[j]);

			min_cost = min(min_cost, cost[k]);
		}

		if (i+r < m-1 && min_cost + cb[i+r+1] >= bsf) {
			return min_cost + cb[i+r+1];
		}

		/// Move current array to previous array.
		swap(cost, costPrev);
	}

	/// the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
	return costPrev[k - 1];
}

double ucr_dtw(vector<double> const& vdata, vector<double> q, int r) {
	const int m = q.size();

	r = min(r, m);

	/// For every EPOCH points, all cummulative values, such as ex (sum), ex2 (sum square), will be restarted for reducing the floating point error.
	const int EPOCH = min(100000, (int) vdata.size());
	vector<double> buffer(EPOCH), u_buff(EPOCH), l_buff(EPOCH);

	// z normalization
	auto z_norm = [](vector<double>& q) {
		const int m = q.size();
		double ex = 0, ex2 = 0;
		for (int i = 0; i < m; i++) {
			ex += q[i];
			ex2 += q[i] * q[i];
		}
		double mean = ex / m;
		double std = sqrt(ex2/m - mean*mean);
		for (int i = 0; i < m; i++)
			q[i] = (q[i] - mean) / std;
	};
	z_norm(q);

	// Create envelop of the query: lower envelop, l, and upper envelop, u
	vector<double> u(m), l(m);
	lower_upper_lemire(q, m, r, l, u);

	/// Sort the query one time by abs(z-norm(q[i]))
	vector<int> order(m);
	iota(order.begin(), order.end(), 0);
	sort(order.begin(), order.end(), [&q](int i, int j) {
		return q[i] > q[j];
	});

	vector<double> qo(m), uo(m), lo(m);
	for (int i = 0; i < m; i++) {
		qo[i] = q[order[i]];
		uo[i] = u[order[i]];
		lo[i] = l[order[i]];
	}

	double bsf = INF;
	vector<double> cb(m), cb1(m), cb2(m);
	int i = 0, j = 0;
	double ex = 0, ex2 = 0;
	bool done = false;
	int it = 0, ep = 0, k = 0;
	long long I;    /// the starting index of the data in current chunk of size EPOCH
	int idx_data = 0;
	vector<double> t(m * 2), tz(m);

	while(!done)
	{
		/// Read first m-1 points
		ep=0;
		if (it==0)
		{   for(k=0; k<m-1; k++)
				if (idx_data != (int) vdata.size())
					buffer[k] = vdata[idx_data++];
		}
		else
		{   for(k=0; k<m-1; k++)
				buffer[k] = buffer[EPOCH-m+1+k];
		}

		/// Read buffer of size EPOCH or when all data has been read.
		ep=m-1;
		while(ep<EPOCH)
		{   if (idx_data == (int) vdata.size())
				break;
			buffer[ep] = vdata[idx_data++];
			ep++;
		}

		/// Data are read in chunk of size EPOCH.
		/// When there is nothing to read, the loop is end.
		if (ep<=m-1)
		{   done = true;
		} else
		{   lower_upper_lemire(buffer, ep, r, l_buff, u_buff);

			/// Do main task here..
			ex=0; ex2=0;
			for(i=0; i<ep; i++)
			{
				/// A bunch of data has been read and pick one of them at a time to use
				double d = buffer[i];

				/// Calcualte sum and sum square
				ex += d;
				ex2 += d*d;

				/// t is a circular array for keeping current data
				t[i%m] = d;

				/// Double the size for avoiding using modulo "%" operator
				t[(i%m)+m] = d;

				/// Start the task when there are more than m-1 points in the current chunk
				if( i >= m-1 )
				{
					double mean = ex/m;
					double std = sqrt(ex2/m-mean*mean);

					/// compute the start location of the data in the current circular array, t
					j = (i+1)%m;
					/// the start location of the data in the current chunk
					I = i-(m-1);

					/// Use a constant lower bound to prune the obvious subsequence
					double lb_kim = lb_kim_hierarchy(t, q, j, m, mean, std, bsf);
					if (lb_kim < bsf) {
						/// Use a linear time lower bound to prune; z_normalization of t will be computed on the fly.
						/// uo, lo are envelop of the query.
						double lb_k = lb_keogh_cumulative(order, t, uo, lo, cb1, j, m, mean, std, bsf);
						if (lb_k < bsf)
						{
							/// Take another linear time to compute z_normalization of t.
							/// Note that for better optimization, this can merge to the previous function.
							for(k=0;k<m;k++)
							{   tz[k] = (t[(k+j)] - mean)/std;
							}

							/// Use another lb_keogh to prune
							/// qo is the sorted query. tz is unsorted z_normalized data.
							/// l_buff, u_buff are big envelop for all data in this chunk
							double lb_k2 = lb_keogh_data_cumulative(order, tz, qo, cb2, l_buff, u_buff, I, m, mean, std, bsf);
							if (lb_k2 < bsf)
							{
								/// Choose better lower bound between lb_keogh and lb_keogh2 to be used in early abandoning DTW
								/// Note that cb and cb2 will be cumulative summed here.
								if (lb_k > lb_k2)
								{
									cb[m-1]=cb1[m-1];
									for(k=m-2; k>=0; k--)
										cb[k] = cb[k+1]+cb1[k];
								}
								else
								{
									cb[m-1]=cb2[m-1];
									for(k=m-2; k>=0; k--)
										cb[k] = cb[k+1]+cb2[k];
								}

								/// Compute DTW and early abandoning if possible
								double dist = dtw(tz, q, cb, m, r, bsf);
								bsf = min(bsf, dist);
							}
						}
					}

					/// Reduce obsolute points from sum and sum square
					ex -= t[j];
					ex2 -= t[j]*t[j];
				}
			}

			/// If the size of last chunk is less then EPOCH, then no more data and terminate.
			if (ep<EPOCH)
				done=true;
			else
				it++;
		}
	}

	return sqrt(bsf);
}


#endif