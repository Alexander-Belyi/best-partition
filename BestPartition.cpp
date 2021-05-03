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

#include "BestPartition.h"
#include "BranchAndBound.h"
#include "PenalizingSubnetworks.h"
#include "ILP.h"
#include "Matrix.h"

#include <iostream>
#include <optional>
#include <vector>
using namespace std;

double TrivialEstimate(const Matrix& Q)
{
    return Sum(Sum(Q, 1, Positive)) + Sum(diag(Q), Negative);
}

double EstimateUB_trivial(const Graph& graph)
{
    return TrivialEstimate(graph.GetModularityMatrix());
}

double EstimateUB_chains_fast(const Graph& graph, const int text_level)
{
	Matrix Q = graph.GetModularityMatrix();
	double estimate = TrivialEstimate(Q);
    size_t n = Q.size();
    //get rid of the loop edges
    for (size_t i = 0; i < n; ++i)
    	Q[i][i] = 0;
    if (text_level > 0)
        cout << "Initial estimate = " << estimate << endl;
    MatrixInt fixedEdges(n, vector<int>(n, -1));
    for (size_t i = 0; i < n; ++i)
        fixedEdges[i][i] = 1;
    vector<PenalizingChain> chains;
    for (size_t path_len = 2; !OnlyPositiveEdgesInPositiveConnComp(Q); ++path_len)
        estimate -= AddPenalizingChains_fast(path_len, chains, Q, fixedEdges, text_level);
    return estimate;
}

double EstimateUB_chains_simplex(const Graph& graph, const int text_level)
{
	Matrix Q = graph.GetModularityMatrix();
	double estimate = TrivialEstimate(Q);
    size_t n = Q.size();
    //get rid of the loop edges
    for (size_t i = 0; i < n; ++i)
    	Q[i][i] = 0;
    if (text_level > 0)
        cout << "Initial estimate = " << estimate << endl;
    MatrixInt fixedEdges(n, vector<int>(n, -1));
    for (size_t i = 0; i < n; ++i)
        fixedEdges[i][i] = 1;
    vector<PenalizingChain> chains;
    estimate -= AddPenalizingChains_simplex(chains, chains, Q, fixedEdges, 4, false, false, text_level);
    return estimate;
}

bool are_all_integers(const vector<double>& v)
{
    const double e = 100 * EPS;
    //we already assume all elements to be in [0, 1] range
    for(double el : v)
        if(abs(el) > e && abs(el - 1) > e)
            return false;
    return true;
}

double EstimateUB_relax_LP(const Graph& graph, optional<unsigned int> time_limit, const int text_level)
{
    Matrix Q = graph.GetModularityMatrix();
    double diag_sum = Sum(diag(Q));
    //get rid of the loop edges
    for (size_t i = 0; i < Q.size(); ++i)
        Q[i][i] = 0;
    auto [estimate, solution] = SolveRelaxedLP(Q, true, time_limit, text_level);
    if (text_level > 0 && are_all_integers(solution))
        cout << "Relaxed problem solved in integers resolving the network" << endl;
    return diag_sum + estimate;
}

SolutionInfo BestPartitionILP(const Graph& graph, optional<unsigned int> time_limit, const int text_level)
{
    SolutionInfo info;
    Matrix Q = graph.GetModularityMatrix();
    double diag_sum = Sum(diag(Q));
    //get rid of the loop edges
    for (size_t i = 0; i < Q.size(); ++i)
        Q[i][i] = 0;
    auto&& [max_mod, solution, num_nodes] = SolveILP(Q, true, time_limit, text_level);
    info.num_visited_nodes = num_nodes;
    info.optimal_solution = max_mod + diag_sum;
    return info;
}

SolutionInfo BestPartitionBnB(const Graph& graph, BnBParameters parameters, optional<unsigned int> time_limit, const int text_level)
{
    Matrix Q = graph.GetModularityMatrix();
    if (Q.empty()) {
        cerr << "Error in EstimateMaxMod: modularity matrix is empty." << endl;
        return SolutionInfo();
    }
    double best_known_score = graph.Modularity();
    SolutionInfo info;
    info.run_time = clock();
    info.trivial_estimate = TrivialEstimate(Q);
    info.num_visited_nodes = 0;
    
    int n = int(Q.size());
    double diag_sum = Sum(diag(Q));
    if (text_level > 1)
        cout << "sum of elements on the diagonal = " << diag_sum << endl;
    //get rid of the loop edges
    for (int i = 0; i < n; ++i)
        Q[i][i] = 0;
    if (best_known_score <= EPS + diag_sum) { // we were not given any good partition
        best_known_score = diag_sum;
        for (int i = 0; i < n-1; ++i)
            for (int j = i+1; j < n; ++j)
                if (Q[i][j] > EPS)
                    best_known_score = max(best_known_score, Q[i][j] + diag_sum);
        if (best_known_score <= EPS + diag_sum) { // there are no positive edges
            info.combos_solution = 
                info.chains_estimate = 
                    info.optimal_solution = 
                        info.best_estimate = best_known_score;
            info.run_time = (clock() - info.run_time) / CLOCKS_PER_SEC;
            return info;
        }
    }
    info.combos_solution = best_known_score;
    MatrixInt fixedEdges(n, vector<int>(n, -1));
    for (int i = 0; i < n; ++i)
        fixedEdges[i][i] = 1;
    for (int i = 0; i < 3; ++i) {
        info.chains_estimate = EstimateUB_chains_fast(graph, text_level);
        if (info.combos_solution + EPS >= info.chains_estimate) {
            info.optimal_solution = info.best_estimate = info.combos_solution;
            info.run_time = (clock() - info.run_time) / CLOCKS_PER_SEC;
            if (text_level > 0)
                cout << "Solved just using chains heuristic." << endl;
            return info;
        }
        if (text_level > 0)
            cout << "Chains=" << parameters.max_chain_len <<" estimate = " << info.chains_estimate << endl;
    }
    vector<PenalizingChain> chains;
    double penalty;
    tie(chains, penalty) = GetPenalizingChains(Q, fixedEdges, vector<PenalizingChain>(), parameters.initial_mode, parameters, text_level);
    info.chains_estimate = info.trivial_estimate - penalty;
    if (info.combos_solution + EPS >= info.chains_estimate) {
        info.optimal_solution = info.best_estimate = info.combos_solution;
        info.run_time = (clock() - info.run_time) / CLOCKS_PER_SEC;
        if (text_level > 0)
            cout << "Solved with simplex with chains of length 4." << endl;
        return info;
    }
    if (text_level > 0)
        cout << "Chains=" << parameters.max_chain_len <<" estimate = " << info.chains_estimate << endl;
    vector<Edge> sortedEdges = SortEdgesByPenalty(Q, fixedEdges, chains, parameters, penalty, text_level);
    int visited_nodes_counter = 0;
    MatrixInt solution(n, vector<int>(n, -1));
    double obtained_score = best_known_score - diag_sum;
    info.best_estimate = BranchAndBoundDFS(clock(), time_limit, parameters, 0, 0, penalty, sortedEdges, 0,
        graph.GetModularityMatrix(), Q, fixedEdges, chains, Sum(Sum(Q, 1, Positive)), obtained_score, solution, visited_nodes_counter, text_level);
    info.best_estimate += diag_sum;
    if (text_level > 0)
        cout << "b&b visited " << visited_nodes_counter << " nodes.\n";
    info.optimal_solution = diag_sum + obtained_score;
    info.num_visited_nodes = visited_nodes_counter;
    info.run_time = (clock() - info.run_time) / CLOCKS_PER_SEC;
    return info;
}
