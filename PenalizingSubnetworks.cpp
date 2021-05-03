/*                                                                            
    Copyright 2021
    Alexander Belyi <alexander.belyi@gmail.com>,
    Stanislav Sobolevsky <sobolevsky@nyu.edu>
 
    This file is part of BestPartition project.

    BestPartition is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    BestPartition is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with BestPartition.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "Matrix.h"
#include "PenalizingSubnetworks.h"

#include "ClpSimplex.hpp"
#ifdef CPLEX_AVAILABLE
#include <ilcplex/ilocplex.h>
#endif

#include <deque>
#include <iostream>
#include <map>
#include <stack>
#include <tuple>
#include <vector>
using namespace std;


void PositiveBFS(const Matrix& m, const MatrixInt& fixedEdges, int v, vector<int>& conncomp, int label)
{
	deque<size_t> q;
	q.push_back(v);
    conncomp[v] = label;
	while (!q.empty()) {
		v = q.front();
		q.pop_front();
		for (size_t i = 0; i < m[v].size(); ++i)
			if (conncomp[i] == 0 && (m[v][i] > EPS || (!fixedEdges.empty() && fixedEdges[v][i] == 1))) {
		        conncomp[i] = label;
				q.push_back(i);
            }
	}
}

vector<int> PositiveConnectedComponents(const Matrix& m, const MatrixInt& fixedEdges)
{
    vector<int> conncomp(m.size(), 0);
    int label = 1;
    for (size_t i = 0; i < m.size(); ++i)
        if (conncomp[i] == 0) {
            PositiveBFS(m, fixedEdges, i, conncomp, label);
            ++label;
        }
    return conncomp;
}

vector<int> PositiveConnectedComponents(const Matrix& m)
{
    PositiveConnectedComponents(m, MatrixInt());
}

bool OnlyPositiveEdgesInPositiveConnComp(const Matrix& Q, const MatrixInt& fixedEdges)
{
    vector<int> conncomp = PositiveConnectedComponents(Q, fixedEdges);
    for (size_t i = 0; i < Q.size(); ++i)
        for (size_t j = 0; j < Q.size(); ++j)
            if (conncomp[i] == conncomp[j] && (Q[i][j] < -EPS || (!fixedEdges.empty() && fixedEdges[i][j] == 0)))
                return false;
    return true;
}

bool OnlyPositiveEdgesInPositiveConnComp(const Matrix& Q)
{
    return OnlyPositiveEdgesInPositiveConnComp(Q, MatrixInt());
}

tuple<
    vector<double>,
    map<pair<size_t, size_t>, vector<pair<size_t, double>>>,
    vector<vector<size_t>>
    >
FindAllShortPenalizingChains(const Matrix& Q, const MatrixInt& fixedEdges, int max_chain_len)
{
    size_t var_num = 0;
    vector<double> vars;
    //could be  matrix instead of map
    map<pair<size_t, size_t>, vector<pair<size_t, double>>> edges;
    vector<vector<size_t>> paths;
    size_t n = Q.size();
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if (fixedEdges[i][j] == 0 || (Q[i][j] < -EPS && fixedEdges[i][j] != 1)) {
                double cur_min_penalty = INF;
                if (fixedEdges[i][j] == -1)
                    cur_min_penalty = -Q[i][j];
                for (size_t k1 = 0; k1 < n; ++k1)
                    if (k1 != i && k1 != j &&
                       (fixedEdges[i][k1] == 1 || (Q[i][k1] > EPS && fixedEdges[i][k1] != 0))) {
                        if (fixedEdges[i][k1] == -1)
                            cur_min_penalty = min(cur_min_penalty, Q[i][k1]);
                        if (fixedEdges[j][k1] == 1 || (Q[j][k1] > EPS && fixedEdges[j][k1] != 0)) {
                            double penalty = cur_min_penalty;
                            if (fixedEdges[j][k1] == -1)
                                penalty = min(penalty, Q[j][k1]);
                            vars.push_back(penalty);
                            if (fixedEdges[i][j] == -1)
                                edges[{i, j}].push_back({var_num, penalty});
                            if (fixedEdges[i][k1] == -1) {
                                if (i < k1)
                                    edges[{i, k1}].push_back({var_num, penalty});
                                else
                                    edges[{k1, i}].push_back({var_num, penalty});
                            }
                            if (fixedEdges[j][k1] == -1) {
                                if (j < k1)
                                    edges[{j, k1}].push_back({var_num, penalty});
                                else
                                    edges[{k1, j}].push_back({var_num, penalty});
                            }
                            ++var_num;
                            paths.push_back({i, k1, j});
                        }
                        if (max_chain_len >= 4)
                        for (size_t k2 = 0; k2 < n; ++k2)
                            if (k2 != i && k2 != j && k2 != k1 &&
                               (fixedEdges[k1][k2] == 1 || (Q[k1][k2] > EPS && fixedEdges[k1][k2] != 0))) {
                                if (fixedEdges[k1][k2] == -1)
                                    cur_min_penalty = min(cur_min_penalty, Q[k1][k2]);
                                if (fixedEdges[j][k2] == 1 || (Q[j][k2] > EPS && fixedEdges[j][k2] != 0)) {
                                    double penalty = cur_min_penalty;
                                    if (fixedEdges[k2][j] == -1)
                                        penalty = min(penalty, Q[k2][j]);
                                    vars.push_back(penalty);
                                    if (fixedEdges[i][j] == -1)
                                        edges[{i, j}].push_back({var_num, penalty});
                                    if (fixedEdges[i][k1] == -1) {
                                        if (i < k1)
                                            edges[{i, k1}].push_back({var_num, penalty});
                                        else
                                            edges[{k1, i}].push_back({var_num, penalty});
                                    }
                                    if (fixedEdges[k1][k2] == -1) {
                                        if (k1 < k2)
                                            edges[{k1, k2}].push_back({var_num, penalty});
                                        else
                                            edges[{k2, k1}].push_back({var_num, penalty});
                                    }
                                    if (fixedEdges[k2][j] == -1) {
                                        if (j < k2)
                                            edges[{j, k2}].push_back({var_num, penalty});
                                        else
                                            edges[{k2, j}].push_back({var_num, penalty});
                                    }
                                    ++var_num;
                                    paths.push_back({i, k1, k2, j});
                                }
                                if (max_chain_len >= 5)
                                for (size_t k3 = 0; k3 < n; ++k3)
                                    if (k3 != i && k3 != j && k3 != k1 && k3 != k2 &&
                                       (fixedEdges[k2][k3] == 1 || (Q[k2][k3] > EPS && fixedEdges[k2][k3] != 0))) {
                                        if (fixedEdges[k2][k3] == -1)
                                            cur_min_penalty = min(cur_min_penalty, Q[k2][k3]);
                                        if (fixedEdges[j][k3] == 1 || (Q[j][k3] > EPS && fixedEdges[j][k3] != 0)) {
                                            double penalty = cur_min_penalty;
                                            if (fixedEdges[k3][j] == -1)
                                                penalty = min(penalty, Q[k3][j]);
                                            vars.push_back(penalty);
                                            if (fixedEdges[i][j] == -1)
                                                edges[{i, j}].push_back({var_num, penalty});
                                            if (fixedEdges[i][k1] == -1) {
                                                if (i < k1)
                                                    edges[{i, k1}].push_back({var_num, penalty});
                                                else
                                                    edges[{k1, i}].push_back({var_num, penalty});
                                            }
                                            if (fixedEdges[k1][k2] == -1) {
                                                if (k1 < k2)
                                                    edges[{k1, k2}].push_back({var_num, penalty});
                                                else
                                                    edges[{k2, k1}].push_back({var_num, penalty});
                                            }
                                            if (fixedEdges[k2][k3] == -1) {
                                                if (k2 < k3)
                                                    edges[{k2, k3}].push_back({var_num, penalty});
                                                else
                                                    edges[{k3, k2}].push_back({var_num, penalty});
                                            }
                                            if (fixedEdges[k3][j] == -1) {
                                                if (j < k3)
                                                    edges[{j, k3}].push_back({var_num, penalty});
                                                else
                                                    edges[{k3, j}].push_back({var_num, penalty});
                                            }
                                            ++var_num;
                                            paths.push_back({i, k1, k2, k3, j});
                                        }
                                        if (max_chain_len >= 6)
                                        for (size_t k4 = 0; k4 < n; ++k4)
                                            if (k4 != i && k4 != j && k4 != k1 && k4 != k2 && k4 != k3 &&
                                                (fixedEdges[k3][k4] == 1 || (Q[k3][k4] > EPS && fixedEdges[k3][k4] != 0))) {
                                                if (fixedEdges[k3][k4] == -1)
                                                    cur_min_penalty = min(cur_min_penalty, Q[k3][k4]);
                                                if (fixedEdges[j][k4] == 1 || (Q[j][k4] > EPS && fixedEdges[j][k4] != 0))
                                                {
                                                    double penalty = cur_min_penalty;
                                                    if (fixedEdges[k4][j] == -1)
                                                        penalty = min(penalty, Q[k4][j]);
                                                    vars.push_back(penalty);
                                                    if (fixedEdges[i][j] == -1)
                                                        edges[{i, j}].push_back({var_num, penalty});
                                                    if (fixedEdges[i][k1] == -1) {
                                                        if (i < k1)
                                                            edges[{i, k1}].push_back({var_num, penalty});
                                                        else
                                                            edges[{k1, i}].push_back({var_num, penalty});
                                                    }
                                                    if (fixedEdges[k1][k2] == -1) {
                                                        if (k1 < k2)
                                                            edges[{k1, k2}].push_back({var_num, penalty});
                                                        else
                                                            edges[{k2, k1}].push_back({var_num, penalty});
                                                    }
                                                    if (fixedEdges[k2][k3] == -1) {
                                                        if (k2 < k3)
                                                            edges[{k2, k3}].push_back({var_num, penalty});
                                                        else
                                                            edges[{k3, k2}].push_back({var_num, penalty});
                                                    }
                                                    if (fixedEdges[k3][k4] == -1) {
                                                        if (k3 < k4)
                                                            edges[{k3, k4}].push_back({var_num, penalty});
                                                        else
                                                            edges[{k4, k3}].push_back({var_num, penalty});
                                                    }
                                                    if (fixedEdges[k4][j] == -1) {
                                                        if (j < k4)
                                                            edges[{j, k4}].push_back({var_num, penalty});
                                                        else
                                                            edges[{k4, j}].push_back({var_num, penalty});
                                                    }
                                                    ++var_num;
                                                    paths.push_back({i, k1, k2, k3, k4, j});
                                                }
                                            }
                                    }
                            }
                    }
            }
    return {vars, edges, paths};
}

tuple<
    vector<double>,
    map<pair<size_t, size_t>, vector<pair<size_t, double>>>,
    vector<vector<size_t>>
    >
UpdateAllShortPenalizingChains(const Matrix& Q,
                                const MatrixInt& fixedEdges,
                                const vector<PenalizingChain>& chains)
{
    size_t var_num = 0;
    vector<double> vars;
    map<pair<size_t, size_t>, vector<pair<size_t, double>>> edges;
    vector<vector<size_t>> paths;
    for (const auto& chain : chains) {
        double penalty = INF;
        bool good_chain = true;
        for (size_t i = 0; i < chain.size(); ++i) {
            Edge e = chain[i];
            if (fixedEdges[e.node1][e.node2] == -1)
                penalty = min(penalty, abs(Q[e.node1][e.node2]));
            if ((fixedEdges[e.node1][e.node2] == 1 && e.weight < 0) ||
               (fixedEdges[e.node1][e.node2] == 0 && e.weight > 0)) {
                good_chain = false;
                break;
            }
        }
        if (good_chain) {
            vars.push_back(penalty);
            for (int i = 0; i < chain.size(); ++i)
            {
                Edge e = chain[i];
                if (fixedEdges[e.node1][e.node2] == -1)
                    edges[{min(e.node1, e.node2), max(e.node1, e.node2)}].push_back({var_num, penalty});
            }
            paths.push_back(chain.chain);
            ++var_num;
        }
    }
    return {vars, edges, paths};
}

PenalizingChain ConstructChain(const vector<size_t>& path, const Matrix& Q, double penalty)
{
    size_t len = path.size();
    vector<double> weights(len);
    for (size_t i = 0; i + 1 < len; ++i)
        weights[i] = Q[path[i]][path[i+1]];
    weights[len-1] = Q[path[0]][path[len-1]];
    return {path, weights, penalty};
}

CoinPackedMatrix CreateCionMatrix(vector<double>& vars, map<pair<size_t, size_t>, vector<pair<size_t, double>>>& edges)
{
    vector<double> elem;
    vector<int> rowInd;
    vector<int> colInd;
    int rInd = 0;
    for (auto it = edges.begin(); it != edges.end(); ++it, ++rInd)
        for (size_t i = 0; i < it->second.size(); ++i)
        {
            elem.push_back(it->second[i].second);
            rowInd.push_back(rInd);
            colInd.push_back(it->second[i].first);
        }
    
    CoinBigIndex numElem = CoinBigIndex(elem.size()); // Number of non-zero elements
    bool storeByCols = false;
    CoinPackedMatrix M(storeByCols, rowInd.data(), colInd.data(), elem.data(), numElem);
    return M;
}

double AddPenalizingChains_clp(const vector<PenalizingChain>& old_chains,
                               vector<PenalizingChain>& chains,
                               const Matrix& Q,
                               const MatrixInt& fixedEdges,
                               int max_chain_len,
                               bool only_nonzero_solution,
                               int text_level)
{
    vector<double> vars;
    map<pair<size_t, size_t>, vector<pair<size_t, double>>> edges;
    vector<vector<size_t>> paths;
    clock_t t1 = clock();
    if (old_chains.size() == 0)
        tie(vars, edges, paths) = FindAllShortPenalizingChains(Q, fixedEdges, max_chain_len);
    else
        tie(vars, edges, paths) = UpdateAllShortPenalizingChains(Q, fixedEdges, old_chains);
    if (text_level > 0) {
        cout << "constructing chains: " << double(clock() - t1) / CLOCKS_PER_SEC << endl;
        t1 = clock();
    }
    CoinPackedMatrix M = CreateCionMatrix(vars, edges);
    if (text_level > 0) {
        cout << "constructing matrix: " << double(clock() - t1) / CLOCKS_PER_SEC << endl;
        t1 = clock();
    }
    vector<double> rowUB(edges.size());
    vector<double> rowLB(edges.size(), 0);
    int ind = 0;
    for (auto it = edges.begin(); it != edges.end(); ++it, ++ind)
        rowUB[ind] = abs(Q[it->first.first][it->first.second]);
    if (text_level > 0) {
        cout << "setting UBs and LBs: " << double(clock() - t1) / CLOCKS_PER_SEC << endl;
        t1 = clock();
    }
    ClpSimplex solver;
    solver.setLogLevel(max(0, text_level-1));
    solver.loadProblem(M, NULL, NULL, vars.data(), rowLB.data(), rowUB.data());
    int optimizationDirection = -1;
    solver.setOptimizationDirection(optimizationDirection);
    solver.dual();
    if (text_level > 0) {
        cout << "solving: " << double(clock() - t1) / CLOCKS_PER_SEC << endl;
        t1 = clock();
    }
    if (text_level > 0 && solver.isAbandoned())
        cerr << "Numerical problems found" << endl;
    if (text_level > 0 && solver.isProvenPrimalInfeasible())
        cerr << "Primal Infeasible" << endl;
    if (solver.isProvenOptimal()) {
        int ncol = solver.getNumCols();
        const double *solution = solver.getColSolution();
        for (int i = 0; i < ncol; ++i) {
            if (!only_nonzero_solution || solution[i] > EPS)
                chains.push_back(ConstructChain(paths[i], Q, vars[i] * solution[i]));
            if (text_level > 1)
                cout << solution[i] << ' ';
        }
        if (text_level > 1)
            cout << endl;
    }
    double penalty = 2.0 * optimizationDirection * solver.rawObjectiveValue();
    return penalty;
}

#ifdef CPLEX_AVAILABLE
double AddPenalizingChains_cplex(const vector<PenalizingChain>& old_chains,
                                 vector<PenalizingChain>& chains,
                                 const Matrix& Q,
                                 const MatrixInt& fixedEdges,
                              	 int max_chain_len,
                                 bool only_nonzero_solution,
                                 int text_level)
{
    vector<double> penalties; //coefficients in obj function
    map<pair<size_t, size_t>, vector<pair<size_t, double>>> edges;
    vector<vector<size_t>> paths;
    if (chains.size() == 0)
        tie(penalties, edges, paths) = FindAllShortPenalizingChains(Q, fixedEdges, max_chain_len);
    else
        tie(penalties, edges, paths) = UpdateAllShortPenalizingChains(Q, fixedEdges, old_chains);
    IloEnv env;
    env.setOut(env.getNullStream());
    env.setNormalizer(false);
    try {
        IloNumArray coeffs(env, penalties.size());
        for (size_t i = 0; i < penalties.size(); ++i)
            coeffs[i] = penalties[i];
        IloNumVarArray vars(env, penalties.size(), 0, IloInfinity);
        IloObjective obj = IloMaximize(env);
        obj.setLinearCoefs(vars, coeffs);
        IloModel model(env);
        model.add(obj);
        for (const auto& p : edges) {
            const auto& e = p.first;
            const auto& col = p.second;
            IloNumExpr row(env);
            for (size_t i = 0; i < col.size(); ++i)
                row += col[i].second * vars[col[i].first];
            model.add(row <= abs(Q[e.first][e.second]));
            row.end();
        }
        IloCplex cplex(model);
        cplex.setParam(IloCplex::Param::Threads, 1);
        if (!cplex.solve())
            env.error() << "Failed to optimize LP" << endl;
        if (text_level > 0 && cplex.getStatus() == IloAlgorithm::Status::Infeasible)
            env.error() << "Infeasibility proven (or none better than cutoff)" << endl;
        if (cplex.getStatus() == IloAlgorithm::Status::Unbounded)
            env.error() << "Continuous solution unbounded" << endl;
        if (cplex.getStatus() == IloAlgorithm::Status::InfeasibleOrUnbounded)
            env.error() << "Problem InfeasibleOrUnbounded" << endl;
        if (cplex.getStatus() == IloAlgorithm::Status::Error)
            env.error() << "Problems found" << endl;
        if (cplex.getStatus() == IloAlgorithm::Status::Unknown ||
           cplex.getStatus() == IloAlgorithm::Status::Feasible)
            env.error() << "Probably some limit reached" << endl;
        if (cplex.getStatus() == IloAlgorithm::Status::Optimal) {
            IloNumArray solution(env);
            cplex.getValues(solution, vars);
            for (size_t i = 0; i < solution.getSize(); ++i) {
                if (!only_nonzero_solution || solution[i] > EPS)
                    chains.push_back(ConstructChain(paths[i], Q, penalties[i] * solution[i]));
                if (text_level > 1)
                    cout << solution[i] << ' ';
            }
            if (text_level > 1)
                cout << endl;
            double penalty = 2.0 * cplex.getObjValue();
            env.end();
            return penalty;
        }
    }
    catch (IloException& e) {
        env.error() << "Concert exception caught: " << e << endl;
    }
    catch (...) {
        env.error() << "Unknown exception caught" << endl;
    }
    env.end();
    return -1;
}
#endif

double AddPenalizingChains_simplex(const vector<PenalizingChain>& old_chains,
                                   vector<PenalizingChain>& new_chains,
                                   const Matrix& Q,
                                   const MatrixInt& fixedEdges,
                                   int max_chain_len,
                                   bool only_nonzero_solution,
                                   bool use_cplex,
                                   int text_level)
{
#ifdef CPLEX_AVAILABLE
    if (use_cplex)
        return AddPenalizingChains_cplex(old_chains, new_chains, Q, fixedEdges, max_chain_len, only_nonzero_solution, text_level);
    else
#endif
        return AddPenalizingChains_clp(old_chains, new_chains, Q, fixedEdges, max_chain_len, only_nonzero_solution, text_level);
}

vector<Edge> NegativeOrExcludedEdges(const Matrix& m, const MatrixInt& fixedEdges)
{
    vector<Edge> e;
    for (size_t i = 0; i < m.size() - 1; ++i)
        for (size_t j = i + 1; j < m.size(); ++j)
            if (fixedEdges[i][j] == 0)
                e.emplace_back(i, j, -INF);
            else if (m[i][j] < -EPS)
                e.emplace_back(i, j, m[i][j]);
    return e;
}

vector<int> ShortestHeavyUnblockedPath(const Matrix& m, const MatrixInt& fixedEdges,
                                       double min_weight, size_t from, size_t to, size_t max_len)
{
    vector<int> prevs(m.size(), -1);
    vector<int> dist(m.size(), -1);
    deque<size_t> q;
    q.push_back(from);
    prevs[from] = from;
    dist[from] = 0;
    while (!q.empty()) {
        size_t i = q.front();
        if (i == to || dist[i] > max_len)
            return prevs;
        q.pop_front();
        for (size_t j = 0; j < m.size(); ++j) {
            if (prevs[j] == -1 && fixedEdges[i][j] != 0 &&
               (m[i][j] > min_weight || fixedEdges[i][j] == 1)) {
                q.push_back(j);
                prevs[j] = i;
                dist[j] = dist[i] + 1;
            }
        }
    }
    return prevs;
}

vector<size_t> Traverse(const vector<int>& prevs, size_t from, size_t to)
{
	if (prevs[to] == -1)
		return vector<size_t>();
	stack<size_t> s;
	s.push(to);
	while (to != from) {
		to = prevs[to];
		s.push(to);
	}
	vector<size_t> path(s.size());
	for (size_t i = 0; !s.empty(); ++i) {
		path[i] = s.top();
		s.pop();
	}
	return path;
}

vector<size_t> GetPositivePath(size_t from, size_t to, size_t len, const Matrix& m, const MatrixInt& fixedEdges)
{
    //look for positive path
    vector<int> prevs = ShortestHeavyUnblockedPath(m, fixedEdges, EPS, from, to, len);
    vector<size_t> path = Traverse(prevs, from, to);
    return path;
}

double GetPathsPenalty(const vector<size_t>& path, const Matrix& m, const MatrixInt& fixedEdges)
{
    size_t from = path[0];
    size_t to = path.back();
    double min_score = INF;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        if (fixedEdges[path[i]][path[i+1]] == 0)
            return 0;
        else if (fixedEdges[path[i]][path[i+1]] == -1)
	        min_score = min(min_score, m[path[i]][path[i+1]]);
    }
    return min_score;
}

double GetChainsPenalty(const vector<size_t>& path, const Matrix& m, const MatrixInt& fixedEdges)
{
    size_t from = path[0];
    size_t to = path.back();
    if (fixedEdges[from][to] == 1)
        return 0;
    double penalty = GetPathsPenalty(path, m, fixedEdges);
    if (fixedEdges[from][to] == -1)
	    penalty = min(penalty, -m[from][to]);
    return penalty;
}

void UpdatePathScore(const vector<size_t>& path, Matrix& m, double penalty)
{
    int from = path[0];
    int to = path.back();
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        m[path[i]][path[i+1]] -= penalty;
        m[path[i+1]][path[i]] -= penalty;
        if (m[path[i]][path[i+1]] < 0) {
            m[path[i]][path[i+1]] = 0;
            m[path[i+1]][path[i]] = 0;
        }
    }
    m[from][to] += penalty;
    m[to][from] += penalty;
    if (m[from][to] > 0) {
        m[from][to] = 0;
        m[to][from] = 0;
    }
}

double AddPenalizingChains_fast(size_t len,
                                vector<PenalizingChain>& chains,
                                Matrix& Q,
                                const MatrixInt& fixedEdges,
                                int text_level)
{
    double total_penalty = 0;
    vector<Edge> edges = NegativeOrExcludedEdges(Q, fixedEdges);
    //sort(edges.begin(), edges.end());
    random_shuffle(edges.begin(), edges.end());
    for (size_t i = 0; i < edges.size(); /* empty */) {
        Edge& edge = edges[i];
        vector<size_t> path = GetPositivePath(edge.node1, edge.node2, len, Q, fixedEdges);
        if (path.size() != len + 1) {
            ++i;
            continue;
        }
        double penalty = GetChainsPenalty(path, Q, fixedEdges);
        chains.push_back(ConstructChain(path, Q, penalty));
        UpdatePathScore(path, Q, penalty);
        total_penalty += 2 * penalty;
        if (fixedEdges[edge.node1][edge.node2] == -1 && Q[edge.node1][edge.node2] > -EPS)
            ++i;
        if (text_level > 2)
            cout << "Path of length "<< len
                 << ", total residual weight left = " << Sum(Sum(Q, 1, Positive))
                 << endl;
    }
    return total_penalty;
}
