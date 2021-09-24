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

#include "BranchAndBound.h"
#include "Combo.h"
#include "PenalizingSubnetworks.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <utility>
#include <vector>

using namespace std;

pair<vector<PenalizingChain>, double> GetPenalizingChains(Matrix Q, const MatrixInt& fixedEdges,
                                                            const vector<PenalizingChain>& old_chains,
                                                            BnBParameters::ChainSearchMode mode,
                                                            const BnBParameters& params,
                                                            int text_level)
{
    double penalty = 0;
    vector<PenalizingChain> chains;
    if (mode == BnBParameters::SIMPLEX) {
        // Disabling this stops reusing chains, so we will run our cycles to find all chains again
        // on random graphs this reduses by half number of visited nodes, but doubles total time
        if (params.reuse_chains)
            penalty += AddPenalizingChainsLP(old_chains, chains, Q, fixedEdges, params.max_chain_len, params.only_nonzero_solution, params.solver == BnBParameters::CPLEX, text_level);
        else
            penalty += AddPenalizingChainsLP(chains, chains, Q, fixedEdges, params.max_chain_len, params.only_nonzero_solution, params.solver == BnBParameters::CPLEX, text_level);
    } else {
        for (auto& chain : old_chains) {
            bool good_chain = true;
            double cur_penalty = INF;
            for (size_t i = 0; i < chain.size(); ++i) {
                Edge e = chain[i];
                if ((fixedEdges[e.node1][e.node2] == 1 && e.weight < 0) ||
                   (fixedEdges[e.node1][e.node2] == 0 && e.weight > 0)) {
                    cur_penalty = 0;
                    good_chain = false;
                    break;
                } else if (fixedEdges[e.node1][e.node2] == -1)
                    cur_penalty = min(cur_penalty, abs(Q[e.node1][e.node2]));
            }
            if (good_chain && cur_penalty > EPS) {
                chains.push_back(chain);
                // we don't really use this new cur_penalty, it was mostly to check that chain is still valid
                if (chain.penalty > cur_penalty) {
                    if (chain.penalty > cur_penalty + 1e-6)
                        cerr << "ASSERTION FAILED: chain.penalty=" << chain.penalty << " > cur_penalty=" << cur_penalty << endl;
                } else
                    cur_penalty = chain.penalty;
                penalty += 2 * cur_penalty;
                for (size_t i = 0; i < chain.size(); ++i) {
                    Edge e = chain[i];
                    if (fixedEdges[e.node1][e.node2] == -1) {
                        if (Q[e.node1][e.node2] > EPS)
                            Q[e.node1][e.node2] = Q[e.node2][e.node1] = Q[e.node1][e.node2] - cur_penalty;
                        else if (Q[e.node1][e.node2] < EPS)
                            Q[e.node1][e.node2] = Q[e.node2][e.node1] = Q[e.node1][e.node2] + cur_penalty;
                    }
                }
            }
        }
        for (size_t path_len = 2; !OnlyPositiveEdgesInPositiveConnComp(Q, fixedEdges); ++path_len)
            penalty += AddPenalizingChainsHeuristic(path_len, chains, Q, fixedEdges, text_level);
    }
    return {chains, penalty};
}

double FixEdge(int v1, int v2, int select, MatrixInt& fixedEdges, Matrix& Q, vector<pair<pair<int, int>, double>>& updatedEdges)
{
    double penalty = 0;
    if (fixedEdges[v1][v2] == -1) {
        updatedEdges.push_back({{v1, v2}, Q[v1][v2]});
        if (select == 0 && Q[v1][v2] > EPS) {
        	penalty = 2 * Q[v1][v2];
            Q[v1][v2] = Q[v2][v1] = 0;
        } else if (select == 1 && Q[v1][v2] < EPS) {
            penalty = -2 * Q[v1][v2];
            Q[v1][v2] = Q[v2][v1] = 0;
        }
    } else if (fixedEdges[v1][v2] != select)
        cerr << "inconsistent edge selection!" << endl;
    fixedEdges[v1][v2] = fixedEdges[v2][v1] = select;
    return penalty;
}

pair<vector<pair<pair<int, int>, double>>, double> FixEdges(int v1, int v2, MatrixInt& fixedEdges, Matrix& Q, int select)
{
    vector<pair<pair<int, int>, double>> updatedEdges;
    double penalty_from_fixing = 0;
    int n = int(fixedEdges.size());
    vector<int> v1in, v1out, v2in, v2out;
    for (size_t i = 0; i < n; ++i) {
        if (fixedEdges[i][v1] == 1)
            v1in.push_back(i);
        if (fixedEdges[i][v2] == 1)
            v2in.push_back(i);
        if (fixedEdges[i][v1] == 0)
            v1out.push_back(i);
        if (fixedEdges[i][v2] == 0)
            v2out.push_back(i);
    }
    for (int in1node : v1in) {
        for (int in2node : v2in)
            penalty_from_fixing += FixEdge(in1node, in2node, select, fixedEdges, Q, updatedEdges);
        if (select == 1)
            for (int out2node : v2out)
                penalty_from_fixing += FixEdge(in1node, out2node, 0, fixedEdges, Q, updatedEdges);
    }
    if (select == 1)
        for (int out1node : v1out)
            for (int in2node : v2in)
                penalty_from_fixing += FixEdge(out1node, in2node, 0, fixedEdges, Q, updatedEdges);
    return {updatedEdges, penalty_from_fixing};
}

double BranchAndBoundDFS(clock_t start_time, optional<unsigned int> time_limit, const BnBParameters& parameters, int depth, double penalty_from_fixing, double optimistic_penalty,
              vector<Edge>& sortedEdges, int edge_number, const Matrix& orig_Q, Matrix& Q, MatrixInt& fixedEdges, const vector<PenalizingChain>& chains,
              double sum_positive, double& best_known_score, MatrixInt& solution, int& visited_nodes_counter, int text_level)
{
    vector<PenalizingChain> new_chains;
    double penalty = optimistic_penalty;
    double estimate = sum_positive - penalty;
    if (estimate / best_known_score <= 1.0 + 100*EPS) {
        if ((depth & 3) == 3)
            tie(new_chains, penalty) = GetPenalizingChains(Q, fixedEdges, chains, parameters.seldom_recalc_mode, parameters, text_level);
        else
            tie(new_chains, penalty) = GetPenalizingChains(Q, fixedEdges, chains, parameters.default_mode, parameters, text_level);
        penalty += penalty_from_fixing;
        estimate = sum_positive - penalty;
        if (time_limit.has_value() && double(clock() - start_time) / CLOCKS_PER_SEC > time_limit.value())
            return estimate;
        if (text_level > 1)
            cout << "best_known_score = " << best_known_score << ", current estimate = " << estimate << endl;        
        if (estimate / best_known_score <= 1.0 + 100*EPS)
            return best_known_score;
    }
    else
        new_chains = chains;
    ++visited_nodes_counter;
    if (text_level > 0 && visited_nodes_counter % 1000 == 1)
        cout << "entering " << visited_nodes_counter << " node, spent " << double(clock() - start_time) / CLOCKS_PER_SEC << " seconds" << endl;
    while (edge_number < sortedEdges.size() && fixedEdges[sortedEdges[edge_number].node1][sortedEdges[edge_number].node2] != -1)
        ++edge_number;
    if (edge_number < sortedEdges.size()) {
        Edge& e = sortedEdges[edge_number];
        if (text_level > 2)
            cout << "fixing edge " << e.node1 << " - " << e.node2 << endl;
        int selections[] = {1, 0};
        double estimates[2];
        for (int selection_index = 0; selection_index < 2; ++selection_index) {
            auto&& [updatedEdges, fixing_penalty] = FixEdges(e.node1, e.node2, fixedEdges, Q, selections[selection_index]);
            double new_optimistic_penalty;
            if (parameters.use_optimistic_estimates && parameters.default_mode == BnBParameters::SIMPLEX)
                new_optimistic_penalty = penalty + fixing_penalty;
            else
                new_optimistic_penalty = sum_positive;
            estimates[selection_index] = BranchAndBoundDFS(start_time, time_limit, parameters, depth+1, penalty_from_fixing + fixing_penalty,
                                                            new_optimistic_penalty, sortedEdges, edge_number + 1, orig_Q, Q, fixedEdges, new_chains,
                                                            sum_positive, best_known_score, solution, visited_nodes_counter, text_level);
            for (auto& p : updatedEdges) {
                fixedEdges[p.first.first][p.first.second] = fixedEdges[p.first.second][p.first.first] = -1;
                Q[p.first.first][p.first.second] = Q[p.first.second][p.first.first] = p.second;
            }
        }
        return max(estimates[0], estimates[1]);
    } else {
        size_t n = Q.size();
        double cur_score = 0;
        for (size_t i = 0; i + 1 < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                if (fixedEdges[i][j] == 1)
                    cur_score += orig_Q[i][j];
        cur_score *= 2;
        if (cur_score - best_known_score > -EPS) {
            best_known_score = cur_score;
            solution = fixedEdges;
        }
        else
            cerr << "ERROR: something went wrong, shouldn't reach this section in file " << __FILE__ << " line " << __LINE__ << "!" << endl;
        return best_known_score;
    }
}

vector<Edge> SortEdgesByPenalty(Matrix& Q, MatrixInt& fixedEdges, const vector<PenalizingChain>& chains, const BnBParameters& parameters, double penalty, int text_level)
{
    vector<Edge> sortedEdges;
    vector<double> orig_scores;
    size_t n = Q.size();
    while(true) {
        vector<Edge> curEdges;
        bool all_positive_excluded = true;
        for (size_t i = 0; i + 1 < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                if (fixedEdges[i][j] == -1 && (!parameters.sort_only_positive_edges || Q[i][j] > 0)) {
                    if (Q[i][j] > 0)
                        all_positive_excluded = false;
                    double edge_score;
                    if (parameters.edge_sorting_order == BnBParameters::WEIGHT)
                        edge_score = Q[i][j];
                    else {
                        fixedEdges[i][j] = fixedEdges[j][i] = 0;
                        double score = Q[i][j];
                        Q[i][j] = Q[j][i] = 0;
                        double new_penalty = GetPenalizingChains(Q, fixedEdges, chains, parameters.recalc_for_sorting_mode, parameters, text_level).second;
                        Q[i][j] = Q[j][i] = score;
                        fixedEdges[i][j] = fixedEdges[j][i] = -1;
                        if (parameters.edge_sorting_order == BnBParameters::PENALTY_DIFFERENCE)
                            edge_score = 2*max(0.0, Q[i][j]) + new_penalty - penalty;
                        else
                            edge_score = 4*Q[i][j] + new_penalty - penalty;
                    }
                    curEdges.push_back({i, j, edge_score});
                }
        if (parameters.edge_sorting_order == BnBParameters::WEIGHT || !(parameters.edge_sorting_order & BnBParameters::RECURSIVE)) {
            sort(curEdges.begin(), curEdges.end(), std::greater<Edge>());
            return curEdges;
        }
        if (all_positive_excluded)
            break;
        Edge best_edge = *max_element(curEdges.begin(), curEdges.end());
        fixedEdges[best_edge.node1][best_edge.node2] = fixedEdges[best_edge.node2][best_edge.node1] = 0;
        double score = Q[best_edge.node1][best_edge.node2];
        Q[best_edge.node1][best_edge.node2] = Q[best_edge.node2][best_edge.node1] = 0;
        if (parameters.edge_sorting_order == BnBParameters::PENALTY_DIFFERENCE)
            penalty = best_edge.weight + penalty - 2*max(0.0, score);
        else
            penalty = best_edge.weight + penalty - 4*score;
        sortedEdges.push_back(best_edge);
        orig_scores.push_back(score);
    }
    for (size_t i = 0; i < orig_scores.size(); ++i) {
        Q[sortedEdges[i].node1][sortedEdges[i].node2] = Q[sortedEdges[i].node2][sortedEdges[i].node1] = orig_scores[i];
        fixedEdges[sortedEdges[i].node1][sortedEdges[i].node2] = fixedEdges[sortedEdges[i].node2][sortedEdges[i].node1] = -1;
    }
    return sortedEdges;
}
