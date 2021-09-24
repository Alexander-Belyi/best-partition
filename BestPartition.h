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

#ifndef BEST_PARTITION_H
#define BEST_PARTITION_H

#include "Graph.h"

#include <optional>
#include <string>
#include <vector>

struct SolutionInfo
{
    double run_time = 0;
    double combos_solution = 0;
    double trivial_estimate = 0;
    double chains_estimate = 0;
    double best_estimate = 0; // if time limit exceeded this will be found instead of optimal_solution
    double optimal_solution = 0;
    int num_visited_nodes = 0;
    SolutionInfo& operator+=(const SolutionInfo& other)
    {
        run_time += other.run_time;
        combos_solution += other.combos_solution;
        trivial_estimate += other.trivial_estimate;
        chains_estimate += other.chains_estimate;
        best_estimate += other.best_estimate;
        optimal_solution += other.optimal_solution;
        num_visited_nodes += other.num_visited_nodes;
        return *this;
    }
};

inline std::string to_string(const SolutionInfo& info)
{
    return  "Time = " + std::to_string(info.run_time) +
            ", trivial estimate = " + std::to_string(info.trivial_estimate) +
            ", chains estimate = " + std::to_string(info.chains_estimate) +
            ", best estimate = " + std::to_string(info.best_estimate) +
            ", Combo's score = " + std::to_string(info.combos_solution) +
            ", optimum = " + std::to_string(info.optimal_solution) +
            ", visited " + std::to_string(info.num_visited_nodes) + " nodes";
}

struct BnBParameters
{
    enum SortingOrder {PENALTY_DIFFERENCE=1, WEIGHT=2, COMBINED=4, RECURSIVE=8};
    enum ChainSearchMode {FAST, SIMPLEX};
    enum SimplexSolver {CLP, CPLEX};
    
    SimplexSolver solver = CLP;
    ChainSearchMode initial_mode = SIMPLEX;
    ChainSearchMode default_mode = FAST;
    ChainSearchMode seldom_recalc_mode = SIMPLEX;
    ChainSearchMode recalc_for_sorting_mode = FAST;
    int max_chain_len = 4;
    bool reuse_chains = true;
    bool only_nonzero_solution = true; //makes sense to disable this only if reuse_chains==true
    SortingOrder edge_sorting_order = WEIGHT;
    bool sort_only_positive_edges = true;
    bool use_optimistic_estimates = default_mode == SIMPLEX &&
                                    seldom_recalc_mode == SIMPLEX &&
                                    only_nonzero_solution && reuse_chains;
};

double EstimateUB_trivial(const Graph& graph);
double EstimateUB_chains_fast(const Graph& graph, const int text_level = 0);
double EstimateUB_chains_simplex(const Graph& graph, const int text_level = 0);
double EstimateUB_chains_and_stars(const Graph& graph, const int text_level = 0);
double EstimateUB_relax_LP(const Graph& graph, std::optional<unsigned int> time_limit, const int text_level = 0);
SolutionInfo BestPartitionILP(const Graph& graph, std::optional<unsigned int> time_limit, const int text_level = 0);
SolutionInfo BestPartitionBnB(const Graph& graph, BnBParameters parameters, std::optional<unsigned int> time_limit, const int text_level = 0);

#endif //BEST_PARTITION_H
