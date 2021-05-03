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

#ifndef BRANCh_AND_BOUND_H
#define BRANCh_AND_BOUND_H

#include "BestPartition.h"
#include "Matrix.h"
#include "PenalizingSubnetworks.h"

#include <optional>
#include <utility>
#include <vector>

std::pair<std::vector<PenalizingChain>, double> GetPenalizingChains(Matrix Q, const MatrixInt& fixedEdges,
                                                                    const std::vector<PenalizingChain>& old_chains,
                                                                    BnBParameters::ChainSearchMode mode,
                                                                    const BnBParameters& params,
                                                                    int text_level);

double BranchAndBoundDFS(clock_t start_time, std::optional<unsigned int> time_limit, const BnBParameters& parameters, int depth, double penalty_from_fixing, double optimistic_penalty,
              std::vector<Edge>& sortedEdges, int edge_number, const Matrix& orig_Q, Matrix& Q, MatrixInt& fixedEdges, const std::vector<PenalizingChain>& chains,
              double sum_positive, double& best_known_score, MatrixInt& solution, int& visited_nodes_counter, int text_level);

std::vector<Edge> SortEdgesByPenalty(Matrix& Q, MatrixInt& fixedEdges, const std::vector<PenalizingChain>& chains, const BnBParameters& parameters, double penalty, int text_level);

#endif //BRANCh_AND_BOUND_H
