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

#ifndef PENALIZING_SUBNETWORKS_H
#define PENALIZING_SUBNETWORKS_H

#include <vector>

struct Edge
{
    size_t node1, node2;
    double weight;
    Edge(size_t n1, size_t n2, double w) : node1(n1), node2(n2), weight(w) {}
    bool operator<(const Edge& e) const
    {
        return weight < e.weight;
    }
    bool operator>(const Edge& e) const
    {
        return weight > e.weight;
    }
};

struct PenalizingChain
{
    std::vector<size_t> chain;
    std::vector<double> weights;
    double penalty;
    Edge operator[](size_t index) const
    {
        if(index + 1 == chain.size())
            return {chain[0], chain[chain.size()-1], weights[index]};
        return {chain[index], chain[index+1], weights[index]};
    }
    size_t size() const
    {
        return chain.size();
    }
};


double AddPenalizingChains_fast(size_t len,
                           std::vector<PenalizingChain>& chains,
                           Matrix& Q,
                           const MatrixInt& fixedEdges,
                           int text_level);

double AddPenalizingChains_simplex(const std::vector<PenalizingChain>& old_chains,
                                   std::vector<PenalizingChain>& chains,
                                   const Matrix& Q,
                                   const MatrixInt& fixedEdges,
                                   int max_chain_len,
                                   bool only_nonzero_solution,
                                   bool use_cplex,
                                   int text_level);

bool OnlyPositiveEdgesInPositiveConnComp(const Matrix& Q, const MatrixInt& fixedEdges);

bool OnlyPositiveEdgesInPositiveConnComp(const Matrix& Q);

#endif //PENALIZING_SUBNETWORKS_H
