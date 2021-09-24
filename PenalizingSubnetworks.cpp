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

#include <algorithm>
#include <deque>
#include <iostream>
#include <map>
#include <random>
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
    return PositiveConnectedComponents(m, MatrixInt());
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

PenalizingChain ConstructChain(const vector<size_t>& path, const Matrix& Q, double penalty)
{
    size_t len = path.size();
    vector<double> weights(len);
    for (size_t i = 0; i + 1 < len; ++i)
        weights[i] = Q[path[i]][path[i+1]];
    weights[len-1] = Q[path[0]][path[len-1]];
    return {path, weights, penalty};
}

double AddPenalizingChainsHeuristic(size_t chain_len,
                                    vector<PenalizingChain>& chains,
                                    Matrix& Q,
                                    const MatrixInt& fixedEdges,
                                    int text_level)
{
    double total_penalty = 0;
    vector<Edge> edges = NegativeOrExcludedEdges(Q, fixedEdges);
    //sort(edges.begin(), edges.end());
    mt19937 rng(7);
    shuffle(edges.begin(), edges.end(), rng);
    for (size_t i = 0; i < edges.size(); /* empty */) {
        Edge& edge = edges[i];
        vector<size_t> path = GetPositivePath(edge.node1, edge.node2, chain_len, Q, fixedEdges);
        if (path.size() != chain_len + 1) {
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
            cout << "Path of length "<< chain_len
                 << ", total residual weight left = " << Sum(Sum(Q, 1, Positive))
                 << endl;
    }
    return total_penalty;
}

void AddVariableToEdges(map<pair<size_t, size_t>, vector<pair<size_t, double>>>& edges,
                        vector<size_t> nodes, size_t var_num, double penalty, const MatrixInt& fixedEdges)
{
    for (size_t i = 0; i+1 < nodes.size(); ++i) {
        size_t u = nodes[i];
        size_t v = nodes[i+1];
        if (fixedEdges[u][v] == -1) {
            if (u < v)
                edges[{u, v}].push_back({var_num, penalty});
            else
                edges[{v, u}].push_back({var_num, penalty});
        }
    }
}

void FindSimplePenalizingStars(const Matrix& Q, const MatrixInt& fixedEdges, vector<double>& vars,
    map<pair<size_t, size_t>, vector<pair<size_t, double>>>& edges)
{
    map<pair<size_t, size_t>, vector<size_t>> paths;
    size_t n = Q.size();
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if (fixedEdges[i][j] == 0 || (Q[i][j] < -EPS && fixedEdges[i][j] == -1))
                for (size_t k = j + 1; k < n; ++k)
                    if ((fixedEdges[i][k] == 0 || (Q[i][k] < -EPS && fixedEdges[i][k] == -1)) &&
                        (fixedEdges[j][k] == 0 || (Q[j][k] < -EPS && fixedEdges[j][k] == -1))) {
                        set<set<size_t>> stars;
                        for (size_t center = 0; center < n; ++center)
                            if (center != i && center != j && center != k) {
                                vector<size_t> path_i = paths.count({center, i}) > 0 ?
                                    paths[{center, i}] :
                                    paths[{center, i}] = GetPositivePath(center, i, n, Q, fixedEdges);
                                vector<size_t> path_j = paths.count({center, j}) > 0 ?
                                    paths[{center, j}] :
                                    paths[{center, j}] = GetPositivePath(center, j, n, Q, fixedEdges);
                                vector<size_t> path_k = paths.count({center, k}) > 0 ?
                                    paths[{center, k}] :
                                    paths[{center, k}] = GetPositivePath(center, k, n, Q, fixedEdges);
                                if ((path_i[1] == path_j[1] && path_k[1] == path_i[1]) ||
                                    (path_i.size() < path_j.size() && i == path_j[path_i.size() - 1]) ||
                                    (path_i.size() > path_j.size() && j == path_i[path_j.size() - 1]) ||
                                    (path_k.size() < path_j.size() && k == path_j[path_k.size() - 1]) ||
                                    (path_k.size() > path_j.size() && j == path_k[path_j.size() - 1]) ||
                                    (path_i.size() < path_k.size() && i == path_k[path_i.size() - 1]) ||
                                    (path_i.size() > path_k.size() && k == path_i[path_k.size() - 1]))
                                    continue;
                                double penalty = GetPathsPenalty(path_i, Q, fixedEdges);
                                penalty = min(penalty, GetPathsPenalty(path_j, Q, fixedEdges));
                                penalty = min(penalty, GetPathsPenalty(path_k, Q, fixedEdges));
                                if (fixedEdges[i][j] == -1)
                                    penalty = min(penalty, -Q[i][j]);
                                if (fixedEdges[i][k] == -1)
                                    penalty = min(penalty, -Q[i][k]);
                                if (fixedEdges[j][k] == -1)
                                    penalty = min(penalty, -Q[j][k]);
                                size_t var_num = vars.size();
                                vars.push_back(2.0 * penalty);
                                for (size_t ind = 1; ind < path_i.size() || ind < path_j.size() || ind < path_k.size(); ++ind) {
                                    if (ind < path_i.size() &&
                                        (ind >= path_j.size() || path_i[ind] != path_j[ind]) &&
                                        (ind >= path_k.size() || path_i[ind] != path_k[ind]) )
                                        AddVariableToEdges(edges, {path_i[ind-1], path_i[ind]}, var_num, penalty, fixedEdges);
                                    if (ind < path_j.size() &&
                                        (ind >= path_k.size() || path_j[ind] != path_k[ind]) )
                                        AddVariableToEdges(edges, {path_j[ind-1], path_j[ind]}, var_num, penalty, fixedEdges);
                                    if (ind < path_k.size())
                                        AddVariableToEdges(edges, {path_k[ind-1], path_k[ind]}, var_num, penalty, fixedEdges);
                                }
                                AddVariableToEdges(edges, {i, k, j, i}, var_num, penalty, fixedEdges);
                            }
                    }
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
                            AddVariableToEdges(edges, {i, k1, j, i}, var_num, penalty, fixedEdges);
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
                                    AddVariableToEdges(edges, {i, k1, k2, j, i}, var_num, penalty, fixedEdges);
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
                                            AddVariableToEdges(edges, {i, k1, k2, k3, j, i}, var_num, penalty, fixedEdges);
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
                                                    AddVariableToEdges(edges, {i, k1, k2, k3, k4, j, i}, var_num, penalty, fixedEdges);
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
            for (size_t i = 0; i < chain.size(); ++i) {
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

CoinPackedMatrix CreateCionMatrix(const vector<double>& vars,
                                  const map<pair<size_t, size_t>, vector<pair<size_t, double>>>& edges)
{
    vector<double> elem;
    vector<int> rowInd;
    vector<int> colInd;
    int rInd = 0;
    for (auto it = edges.begin(); it != edges.end(); ++it, ++rInd)
        for (size_t i = 0; i < it->second.size(); ++i) {
            elem.push_back(it->second[i].second);
            rowInd.push_back(rInd);
            colInd.push_back(it->second[i].first);
        }
    CoinBigIndex numElem = CoinBigIndex(elem.size()); // Number of non-zero elements
    bool storeByCols = false;
    CoinPackedMatrix M(storeByCols, rowInd.data(), colInd.data(), elem.data(), numElem);
    return M;
}

double FindMaxPenaltyWithCLP(vector<double>& solution,
                             const vector<double>& var_coeffs,
                             const map<pair<size_t, size_t>, vector<pair<size_t, double>>>& edges,
                             const Matrix& Q,
                             int text_level)
{
    if (var_coeffs.empty() || edges.empty())
        return 0;
    clock_t start_time = clock();
    CoinPackedMatrix M = CreateCionMatrix(var_coeffs, edges);
    if (text_level > 0) {
        cout << "constructing matrix: " << double(clock() - start_time) / CLOCKS_PER_SEC << endl;
        start_time = clock();
    }
    vector<double> rowUB(edges.size());
    vector<double> rowLB(edges.size(), 0);
    int ind = 0;
    for (auto it = edges.begin(); it != edges.end(); ++it, ++ind)
        rowUB[ind] = abs(Q[it->first.first][it->first.second]);
    if (text_level > 0) {
        cout << "setting UBs and LBs: " << double(clock() - start_time) / CLOCKS_PER_SEC << endl;
        start_time = clock();
    }
    ClpSimplex solver;
    solver.setLogLevel(max(0, text_level-1));
    solver.loadProblem(M, NULL, NULL, var_coeffs.data(), rowLB.data(), rowUB.data());
    int optimizationDirection = -1;
    solver.setOptimizationDirection(optimizationDirection);
    solver.dual();
    if (text_level > 0) {
        cout << "solving: " << double(clock() - start_time) / CLOCKS_PER_SEC << endl;
        start_time = clock();
    }
    if (text_level > 0 && solver.isAbandoned())
        cerr << "Numerical problems found" << endl;
    if (text_level > 0 && solver.isProvenPrimalInfeasible())
        cerr << "Primal Infeasible" << endl;
    if (solver.isProvenOptimal()) {
        const double *solution_ptr = solver.getColSolution();
        solution.assign(solution_ptr, solution_ptr + solver.getNumCols());
        double penalty = 2.0 * optimizationDirection * solver.rawObjectiveValue();
        return penalty;
    } else {
        cerr << "Optimal solution wan't found" << endl;
        return 0;
    }
}

#ifdef CPLEX_AVAILABLE
double FindMaxPenaltyWithCPLEX(vector<double>& solution,
                               const vector<double>& var_coeffs,
                               const map<pair<size_t, size_t>, vector<pair<size_t, double>>>& edges,
                               const Matrix& Q,
                               int text_level)
{
    if (var_coeffs.empty() || edges.empty())
        return 0;
    IloEnv env;
    env.setOut(env.getNullStream());
    env.setNormalizer(false);
    try {
        IloNumArray coeffs(env, var_coeffs.size());
        for (size_t i = 0; i < var_coeffs.size(); ++i)
            coeffs[i] = var_coeffs[i];
        IloNumVarArray vars(env, var_coeffs.size(), 0, IloInfinity);
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
            IloNumArray ilo_solution(env);
            cplex.getValues(ilo_solution, vars);
            solution.assign(ilo_solution.getSize(), 0.0);
            for (size_t i = 0; i < ilo_solution.getSize(); ++i)
                solution[i] = ilo_solution[i];
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
    return 0;
}
#endif

double AddPenalizingChainsLP(const vector<PenalizingChain>& old_chains,
                             vector<PenalizingChain>& new_chains,
                             const Matrix& Q,
                             const MatrixInt& fixedEdges,
                             int max_chain_len,
                             bool only_nonzero_solution,
                             bool prefer_cplex,
                             int text_level)
{
    vector<double> penalties; //coefficients in obj function
    map<pair<size_t, size_t>, vector<pair<size_t, double>>> edges;
    vector<vector<size_t>> paths;
    clock_t start_time = clock();
    if (old_chains.size() == 0)
        tie(penalties, edges, paths) = FindAllShortPenalizingChains(Q, fixedEdges, max_chain_len);
    else
        tie(penalties, edges, paths) = UpdateAllShortPenalizingChains(Q, fixedEdges, old_chains);
    if (text_level > 0)
        cout << "constructing chains: " << double(clock() - start_time) / CLOCKS_PER_SEC << endl;
    vector<double> solution;
    double penalty = 0;
#ifdef CPLEX_AVAILABLE
    if (prefer_cplex)
        penalty = FindMaxPenaltyWithCPLEX(solution, penalties, edges, Q, text_level);
    else
#endif
        penalty = FindMaxPenaltyWithCLP(solution, penalties, edges, Q, text_level);
    for (size_t i = 0; i < solution.size(); ++i) {
        if (!only_nonzero_solution || solution[i] > EPS)
            new_chains.push_back(ConstructChain(paths[i], Q, penalties[i] * solution[i]));
        if (text_level > 1)
            cout << solution[i] << ' ';
    }
    if (text_level > 1)
        cout << endl;
    return penalty;
}

double GetPenaltyUsingChainsAndStars(const Matrix& Q,
                                     const MatrixInt& fixedEdges,
                                     int max_chain_len,
                                     bool prefer_cplex,
                                     int text_level)
{
    vector<double> penalties;
    map<pair<size_t, size_t>, vector<pair<size_t, double>>> edges;
    vector<vector<size_t>> paths;
    clock_t start_time = clock();
    tie(penalties, edges, paths) = FindAllShortPenalizingChains(Q, fixedEdges, max_chain_len);
    if (text_level > 0) {
        cout << "constructing chains: " << double(clock() - start_time) / CLOCKS_PER_SEC << endl;
        start_time = clock();
    }
    FindSimplePenalizingStars(Q, fixedEdges, penalties, edges);
    if (text_level > 0)
        cout << "constructing stars: " << double(clock() - start_time) / CLOCKS_PER_SEC << endl;
    vector<double> solution;
#ifdef CPLEX_AVAILABLE
    if (prefer_cplex)
        return FindMaxPenaltyWithCPLEX(solution, penalties, edges, Q, text_level);
    else
#endif
        return FindMaxPenaltyWithCLP(solution, penalties, edges, Q, text_level);
}
