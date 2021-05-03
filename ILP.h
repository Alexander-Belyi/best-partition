/*
 Copyright 2021
 Alexander Belyi <alexander.belyi@gmail.com>
 
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

#ifndef ILP_h
#define ILP_h

#include "Matrix.h"

#include <optional>
#include <tuple>
#include <vector>

std::tuple<double, std::vector<double>> SolveRelaxedLP(const Matrix& Q, bool use_cplex, std::optional<unsigned int> time_limit = std::nullopt, int text_level = 0);

std::tuple<double, MatrixInt, int> SolveILP(const Matrix& Q, bool use_cplex, std::optional<unsigned int> time_limit = std::nullopt, int text_level = 0);

#endif /* ILP_h */
