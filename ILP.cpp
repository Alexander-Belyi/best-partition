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

#include "ILP.h"
#include "Matrix.h"
#include "ClpSimplex.hpp"
#include "OsiClpSolverInterface.hpp"
#include "CbcModel.hpp"
#ifdef CPLEX_AVAILABLE
#include <ilcplex/ilocplex.h>
#endif

#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <tuple>
#include <vector>

using namespace std;

CoinPackedMatrix CreateCoinMatrix(const Matrix& Q)
{
    int n = int(Q.size());
    vector<double> elem;
    vector<int> rowInd;
    vector<int> colInd;
    int row_cnt = 0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            for (int k = j + 1; k < n; ++k) {
                vector<pair<int, int>> pairs = {{i, j}, {i, k}, {j, k}};
                for (int x = 0; x < 3; ++x) {
                    bool skip = true; //skip redundant constraint
                    for (int y = 0; y < 3; ++y)
                        if (y != x && Q[pairs[y].first][pairs[y].second] >= 0)
                            skip = false;
                    if (!skip) {
                        for (int y = 0; y < 3; ++y) {
                            elem.push_back(1 - 2 * (y == x));
                            colInd.push_back(GetEdgeNumber(pairs[y].first, pairs[y].second, n));
                            rowInd.push_back(row_cnt);
                        }
                        ++row_cnt;
                    }
                }
            }
    
    bool storeByCols = false;
    CoinPackedMatrix M(storeByCols, rowInd.data(), colInd.data(), elem.data(), int(elem.size()));
    return M;
}

tuple<double, vector<double>> SolveLP_clp(const Matrix& Q, optional<unsigned int> time_limit, int text_level)
{
    int n = int(Q.size());
    if (n < 2)
        return {0, vector<double>()};
    if (n == 2) {
        if (Q[0][1] > 0)
            return {Q[0][1], vector<double>{1.0}};
        else
            return {0.0, vector<double>{0.0}};
    }
    int n_edges = n * (n-1) / 2;
    int var_num = n_edges; // number of variables is the number of edges
    vector<double> vars(var_num);
    for (int i = 0; i < n;++i)
        for (int j = i + 1; j < n; ++j)
            vars[GetEdgeNumber(i, j, n)] = Q[i][j];
    CoinPackedMatrix M = CreateCoinMatrix(Q);
    vector<double> rowUB(M.getNumRows(), 1.0);
    vector<double> rowLB(M.getNumRows(), -1.0);
    ClpSimplex solver;
    solver.setLogLevel(max(text_level-1, 0));
    solver.loadProblem(M, NULL, NULL, vars.data(), rowLB.data(), rowUB.data());
    int optimizationDirection = -1;
    solver.setOptimizationDirection(optimizationDirection);
    if (time_limit.has_value())
        solver.setMaximumSeconds(time_limit.value());
    solver.initialSolve();
    int ncol = solver.getNumCols();
    vector<double> solution(solver.getColSolution(), solver.getColSolution() + ncol);
    
    if (text_level > 1 && solver.isProvenOptimal()) {
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                cout << i << " -> " << j << ": " << solution[GetEdgeNumber(i, j, n)] << endl;
    }
    if (text_level > 0 && solver.isAbandoned())
        cerr << "Numerical problems found" << endl;
    if (text_level > 0 && solver.isProvenPrimalInfeasible())
        cerr << "Primal Infeasible" << endl;
    if (text_level > 0 && solver.isProvenDualInfeasible())
        cerr << "Dual Infeasible" << endl;
    return {2.0 * optimizationDirection * solver.rawObjectiveValue(), solution};
}

#ifdef CPLEX_AVAILABLE
tuple<double, vector<double>> SolveLP_cplex(const Matrix& Q, optional<unsigned int> time_limit, int text_level)
{
    int n = int(Q.size());
    if (n < 2)
        return {0, vector<double>()};
    if (n == 2) {
        if (Q[0][1] > 0)
            return {Q[0][1], vector<double>{1.0}};
        else
            return {0.0, vector<double>{0.0}};
    }
    IloEnv env;
    env.setOut(env.getNullStream());
    env.setNormalizer(false);
    try {
        IloModel model(env);
        IloObjective obj = IloMaximize(env);
        
        int num_edges = n * (n-1) / 2;
        int edge_num = 0;
        IloNumArray coeffs(env, num_edges);
        for (int i = 0; i < n-1; ++i)
            for (int j = i + 1; j < n; ++j) {
                coeffs[edge_num] = Q[i][j];
                ++edge_num;
            }
        IloNumVarArray vars(env, num_edges, 0, 1);
        obj.setLinearCoefs(vars, coeffs);
        model.add(obj);
        IloRangeArray constrs(env);
        for (int i = 0; i < n-2; ++i)
            for (int j = i + 1; j < n-1; ++j)
                for (int k = j + 1; k < n; ++k) {
                    int ij = GetEdgeNumber(i, j, n);
                    int jk = GetEdgeNumber(j, k, n);
                    int ik = GetEdgeNumber(i, k, n);
                    if (Q[i][j] >= 0 || Q[j][k] >= 0)
                        constrs.add( vars[ij] + vars[jk] - vars[ik] <= 1);
                    if (Q[i][j] >= 0 || Q[i][k] >= 0)
                        constrs.add( vars[ij] - vars[jk] + vars[ik] <= 1);
                    if (Q[j][k] >= 0 || Q[i][k] >= 0)
                        constrs.add(-vars[ij] + vars[jk] + vars[ik] <= 1);
                }
        model.add(constrs);
        IloCplex cplex(model);
        cplex.setParam(IloCplex::Param::Threads, 1);
        if (time_limit.has_value())
            cplex.setParam(IloCplex::Param::TimeLimit, time_limit.value());
        if (!cplex.solve())
            env.error() << "Failed to optimize LP" << endl;
        if (cplex.getStatus() == IloAlgorithm::Status::Infeasible)
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
            vector<double> res(solution.getSize());
            double acheived_mod = 0;
            for (int i = 0; i < n-1; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    int ind = GetEdgeNumber(i, j, n);
                    res[ind] = solution[ind];
                    acheived_mod += 2.0 * solution[ind] * Q[i][j];
                    if (text_level > 1)
                        cout << i << " -> " << j << ": " << solution[ind] << endl;
                }
            }
            env.end();
            return {acheived_mod, res};
        }
    }
    catch (IloException& e) {
        env.error() << "Concert exception caught: " << e << endl;
    }
    catch (...) {
        env.error() << "Unknown exception caught" << endl;
    }
    env.end();

    return {-1, vector<double>()};
}
#endif

tuple<double, MatrixInt, int> SolveIP_clp(const Matrix& Q, optional<unsigned int> time_limit, int text_level)
{
    int n = int(Q.size());
    if (n < 2)
        return {0, MatrixInt(), 0};
    if (n == 2) {
        if (Q[0][1] > 0)
            return {Q[0][1], MatrixInt(2, vector<int>(2, 1)), 0};
        else
            return {0.0, MatrixInt(2, vector<int>(2, 0)), 0};
    }
    int n_edges = n * (n-1) / 2;
    int var_num = n_edges; // number of variables is the number of edges
    vector<double> vars(var_num);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            vars[GetEdgeNumber(i, j, n)] = Q[i][j];
    CoinPackedMatrix M = CreateCoinMatrix(Q);
    vector<double> colUB(var_num, 1.0);
    vector<double> colLB(var_num, 0);
    vector<double> rowUB(M.getNumRows(), 1.0);
    vector<double> rowLB(M.getNumRows(), -1.0);
    vector<int> integers(var_num);
    iota(integers.begin(), integers.end(), 0);
    OsiClpSolverInterface solver;
    solver.loadProblem(M, colLB.data(), colUB.data(), vars.data(), rowLB.data(), rowUB.data());
    solver.setInteger(integers.data(), var_num);
    int optimizationDirection = -1;
    solver.setObjSense(optimizationDirection);
    //if (time_limit.has_value())
    //    solver.setMaximumSeconds(time_limit.value());
    solver.branchAndBound();

    vector<double> solution(solver.getColSolution(), solver.getColSolution() + var_num);
    MatrixInt connectivityMatrix(n, vector<int>(n));
    if (solver.isProvenOptimal()) {
        for (int i = 0; i < n-1; ++i) {
            connectivityMatrix[i][i] = 1;
            for (int j = i + 1; j < n; ++j) {
                connectivityMatrix[i][j] = connectivityMatrix[j][i] = solution[GetEdgeNumber(i, j, n)];
                if (text_level > 1)
                    cout << i << " -> " << j << ": " << connectivityMatrix[i][j] << endl;
            }
        }
    }
    if (text_level > 0 && solver.isAbandoned())
        cerr << "Numerical problems found" << endl;
    if (text_level > 0 && solver.isProvenPrimalInfeasible())
        cerr << "Primal Infeasible" << endl;
    return {2 * solver.getObjValue(), connectivityMatrix, -1};
}

tuple<double, MatrixInt, int> SolveIP_cbc(const Matrix& Q, optional<unsigned int> time_limit, int text_level)
{
    int n = int(Q.size());
    if (n < 2)
        return {0, MatrixInt(), 0};
    if (n == 2) {
        if (Q[0][1] > 0)
            return {Q[0][1], MatrixInt(2, vector<int>(2, 1)), 0};
        else
            return {0.0, MatrixInt(2, vector<int>(2, 0)), 0};
    }
    int n_edges = n * (n-1) / 2;
    int var_num = n_edges; // number of variables is the number of edges
    vector<double> vars(var_num);
    for (int i = 0; i < n;++i)
        for (int j = i + 1; j < n; ++j)
            vars[GetEdgeNumber(i, j, n)] = Q[i][j];
    CoinPackedMatrix M = CreateCoinMatrix(Q);
    vector<double> colUB(var_num, 1.0);
    vector<double> colLB(var_num, 0);
    vector<double> rowUB(M.getNumRows(), 1.0);
    vector<double> rowLB(M.getNumRows(), -1.0);
    vector<int> integers(var_num);
    iota(integers.begin(), integers.end(), 0);
    OsiClpSolverInterface solver;
    solver.loadProblem(M, colLB.data(), colUB.data(), vars.data(), rowLB.data(), rowUB.data());
    solver.setInteger(integers.data(), var_num);
    int optimizationDirection = -1;
    CbcModel model(solver);
    model.setLogLevel(max(text_level-1, 0));
    model.setObjSense(optimizationDirection);
    if (time_limit.has_value())
        model.setMaximumSeconds(time_limit.value());
    model.branchAndBound();
    if (model.isProvenOptimal()) {
        vector<double> solution(model.bestSolution(), model.bestSolution() + var_num);
        MatrixInt connectivityMatrix(n, vector<int>(n));
        for (int i = 0; i < n-1; ++i) {
            connectivityMatrix[i][i] = 1;
            for (int j = i + 1; j < n; ++j) {
                connectivityMatrix[i][j] = connectivityMatrix[j][i] = solution[GetEdgeNumber(i, j, n)];
                if (text_level > 1)
                    cout << i << " -> " << j << ": " << connectivityMatrix[i][j] << endl;
            }
        }
        return {2 * model.getObjValue(), connectivityMatrix, model.getNodeCount()};
    }
    if (model.isProvenInfeasible())
        cerr << "Infeasibility proven (or none better than cutoff)" << endl;
    if (model.isProvenDualInfeasible())
        cerr << "ProvenDualInfeasible" << endl;
    if (model.isAbandoned())
        cerr << "Numerical problems found" << endl;
    if (model.isContinuousUnbounded())
        cerr << "Continuous solution unbounded" << endl;
    if (model.isNodeLimitReached())
        cerr << "Node limit reached" << endl;
    if (model.isSecondsLimitReached())
        cerr << "Time limit reached" << endl;
    if (model.isSolutionLimitReached())
        cerr << "Solution limit reached" << endl;
    return {0, MatrixInt(), 0};
}

#ifdef CPLEX_AVAILABLE
tuple<double, MatrixInt, int> SolveIP_cplex(const Matrix& Q, optional<unsigned int> time_limit, int text_level)
{
    int n = int(Q.size());
    if (n < 2)
        return {0, MatrixInt(), 0};
    if (n == 2) {
        if (Q[0][1] > 0)
            return {Q[0][1], MatrixInt(2, vector<int>(2, 1)), 0};
        else
            return {0.0, MatrixInt(2, vector<int>(2, 0)), 0};
    }
    IloEnv env;
    env.setOut(env.getNullStream());
    env.setNormalizer(false);
    try {
        IloModel model(env);
        IloObjective obj = IloMaximize(env);
        int num_edges = n * (n-1) / 2;
        int edge_num = 0;
        IloNumArray coeffs(env, num_edges);
        for (int i = 0; i < n-1; ++i)
            for (int j = i + 1; j < n; ++j) {
                coeffs[edge_num] = Q[i][j];
                ++edge_num;
            }
        IloNumVarArray vars(env, num_edges, 0, 1, ILOBOOL);
        obj.setLinearCoefs(vars, coeffs);
        model.add(obj);
        IloRangeArray constrs(env);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                for (int k = j + 1; k < n; ++k) {
                    int ij = GetEdgeNumber(i, j, n);
                    int jk = GetEdgeNumber(j, k, n);
                    int ik = GetEdgeNumber(i, k, n);
                    if (Q[i][j] >= 0 || Q[j][k] >= 0)
                        constrs.add( vars[ij] + vars[jk] - vars[ik] <= 1);
                    if (Q[i][j] >= 0 || Q[i][k] >= 0)
                        constrs.add( vars[ij] - vars[jk] + vars[ik] <= 1);
                    if (Q[j][k] >= 0 || Q[i][k] >= 0)
                        constrs.add(-vars[ij] + vars[jk] + vars[ik] <= 1);
                }
        model.add(constrs);
        IloCplex cplex(model);
        cplex.setParam(IloCplex::Param::Threads, 1);
        if (time_limit.has_value())
            cplex.setParam(IloCplex::Param::TimeLimit, time_limit.value());
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
            MatrixInt connectivityMatrix(n, vector<int>(n));
            double achieved_mod = 0;
            for (int i = 0; i < n-1; ++i) {
                connectivityMatrix[i][i] = 1;
                for (int j = i + 1; j < n; ++j) {
                    connectivityMatrix[i][j] = connectivityMatrix[j][i] = IloRound(solution[GetEdgeNumber(i, j, n)]);
                    achieved_mod += 2.0 * connectivityMatrix[i][j] * Q[i][j];
                    if (text_level > 1)
                        cout << i << " -> " << j << ": " << connectivityMatrix[i][j] << endl;
                }
            }
            int visited_nodes = int(cplex.getNnodes());
            env.end();
            return {achieved_mod, connectivityMatrix, visited_nodes};
        }
    }
    catch (IloException& e) {
        env.error() << "Concert exception caught: " << e << endl;
    }
    catch (...) {
        env.error() << "Unknown exception caught" << endl;
    }
    env.end();
    return {0, MatrixInt(), 0};
}
#endif

tuple<double, MatrixInt, int> SolveILP(const Matrix& Q, bool use_cplex, optional<unsigned int> time_limit, int text_level)
{
#ifdef CPLEX_AVAILABLE
    if (use_cplex)
        return SolveIP_cplex(Q, time_limit, text_level);
    else
#endif
        return SolveIP_cbc(Q, time_limit, text_level);
}

tuple<double, vector<double>> SolveRelaxedLP(const Matrix& Q, bool use_cplex, optional<unsigned int> time_limit, int text_level)
{
#ifdef CPLEX_AVAILABLE
    if (use_cplex)
        return SolveLP_cplex(Q, time_limit, text_level);
    else
#endif
        return SolveLP_clp(Q, time_limit, text_level);
}
