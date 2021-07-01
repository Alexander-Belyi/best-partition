/*                                                                            
    Copyright 2021
    Alexander Belyi <alexander.belyi@gmail.com>                                      
                                                                            
    This is the main file of BestPartition project.

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
#include "Graph.h"
#include "Combo.h"
#include "BestPartition.h"

#include <ctime>
#include <cmath>
#include <iostream>
#include <fstream>
#include <optional>
using namespace std;

string tests_folder = "../test-networks/";

ComboAlgorithm combo(7, 0, 0);

vector<SolutionInfo> run_CPP_test(Graph& G, BnBParameters bnb_params,
	bool compare_with_ILP = false, int num_combo_runs = 2, int text_level = 0)
{
	clock_t time_start = clock();
	cout.precision(6);
	double mod_combo = 0;
	vector<size_t> communities;
	for (int i = 0; i < num_combo_runs; ++i) {
		combo.Run(G, nullopt, i & 1);
		if (mod_combo < G.Modularity()) {
			mod_combo = G.Modularity();
			communities = G.Communities();
		}
	}
	G.SetCommunities(communities);
	time_start = clock();
	SolutionInfo info = BestPartitionBnB(G, bnb_params, nullopt, text_level);
	info.run_time = double(clock() - time_start) / CLOCKS_PER_SEC;
	vector<SolutionInfo> res = {info};
	if (compare_with_ILP) {
		time_start = clock();
		SolutionInfo IP_info = BestPartitionILP(G, nullopt, 0);
		IP_info.run_time = double(clock() - time_start) / CLOCKS_PER_SEC;
		if (abs(IP_info.optimal_solution - info.optimal_solution) > EPS)
			cerr << "ERROR: optimal_solution by IP != optimal_solution" << endl;
		if (abs(info.optimal_solution - info.best_estimate) > EPS)
			cerr << "ERROR: optimal_solution != best_estimate" << endl;
		res = {info, IP_info};
	}
	return res;
}

int run_CPP_rand_test_nets(int test_set, bool compare_with_ILP = false, int num_combo_runs = 2, int text_level = 0)
{
	string path = tests_folder+"/Jaehn/cpp_random_graphs/test_set_" + to_string(test_set) + "/";
	set<string> file_ends;
	if (test_set == 3)
		file_ends = {"_40.edgelist", "_80.edgelist"};
	else
		file_ends = {".edgelist"};
	int max_net_size = 20;
	if (test_set == 2)
		max_net_size = 24;
	BnBParameters bnb_params;
	for (const string& file_name_end : file_ends) {
		for (int n = 10; n <= max_net_size; ++n) {
			clock_t batch_start = clock();
			vector<SolutionInfo> sum_infos(1);
			if (compare_with_ILP)
				sum_infos.assign(2, SolutionInfo());
			for (int q : {1, 2, 3, 5, 10, 50, 100})
				for (int i = 0; i < 5; ++i) {
					string file_name = path + to_string(n) + '_' + to_string(q) + '_' + to_string(i) + file_name_end;
					Graph G = ReadGraphFromFile(file_name, 1.0, true);
					int info_index = 0;
					for (auto& info : run_CPP_test(G, bnb_params, compare_with_ILP, num_combo_runs, text_level)) {
						sum_infos[info_index] += info;
						++info_index;
					}
				}
			cout << "Network size = " << n
				 << ". Batch Time: " << double(clock() - batch_start) / CLOCKS_PER_SEC
				 << ", ";
			for (auto& sum_info : sum_infos)
				cout << to_string(sum_info) << ";\n";
			if (compare_with_ILP)
				cout << endl;
		}
	}
	cout << endl;
	return 0;
}

int run_CPP_rw_test_nets(int test_set, bool compare_with_ILP = false, int num_combo_runs = 2, int text_level = 0)
{
	string path = tests_folder+"/Jaehn/cpp_real_world_graphs/Grotschel-Wakabayashi/";
	vector<string> file_names = {"wild_cats", "cars", "workers", "cetacea", "micro", "UNO", "UNO_1a", "UNO_1b", "UNO_2a", "UNO_2b"};
	BnBParameters bnb_params;
	if (test_set == 2) {
		path = tests_folder+"/Jaehn/cpp_real_world_graphs/Oosten/";
		file_names = {"KKV", "SUL", "SEI", "MCC", "BOC"};
		bnb_params.edge_sorting_order = BnBParameters::PENALTY_DIFFERENCE;
		bnb_params.default_mode = BnBParameters::SIMPLEX;
		bnb_params.reuse_chains = false;
	}
    for (const string& net_name : file_names) {
		clock_t batch_start = clock();
		vector<SolutionInfo> sum_infos(1);
		if (compare_with_ILP)
			sum_infos.assign(2, SolutionInfo());
		string file_name = path + net_name + ".edgelist";
		Graph G = ReadGraphFromFile(file_name, 1.0, true);
		int info_index = 0;
		for (auto& info : run_CPP_test(G, bnb_params, compare_with_ILP, num_combo_runs, text_level)) {
			sum_infos[info_index] += info;
			++info_index;
		}
		cout << "Network name = " << net_name
			 << ", size = " << G.Size()
		 	 << ". Batch Time: " << double(clock() - batch_start) / CLOCKS_PER_SEC
		 	 << ", ";
		for (auto& sum_info : sum_infos)
			cout << to_string(sum_info) << "; ";
		cout << endl;
	}
	cout << endl;
	return 0;
}

int run_Miyauchi_nets()
{
	string path = tests_folder+"/Miyauchi/Modularity/";
	vector<string> network_file_names = {
		"Zachary Karate.net", // 0.1s, 0 nodes
		"Dolphins Social Network.net", // 5.6s, 337 nodes
		"Les Miserables_unit.net", // 7.3s, 74 nodes
		"Political Books.net", // 105, IP: 0.52723659380607923, IP Time: 101.6184; bnb: Time = 30.2, trivial estimate = 0.845149, chains estimate = 0.527913, best estimate = 0.527237, Combo's score = 0.527237, optimum = 0.527237, visited 137 nodes
		"American College Football.net", // 115, IP: 0.60456956268345896, IP Time: 33.25; bnb: Time = 41.41, trivial estimate = 0.905928, chains estimate = 0.605627, best estimate = 0.604570, Combo's score = 0.604570, optimum = 0.604570, visited 132 nodes
		"USAir97.net", //332 nodes
		"s838.net", //512 nodes
		"netscience.net", //1589 nodes
		"power-grid.net" //4941 nodes
	};
	for(const string& net_name : network_file_names)
	{
		Graph G = ReadGraphFromFile(path + net_name);
		cout << endl << net_name << " size = " << G.Size() << endl;
		clock_t time_start = clock();
		cout << " estimated UB = " << EstimateUB_chains_fast(G)
		 	 << ". Time: " << double(clock() - time_start) / CLOCKS_PER_SEC << endl;
	}
	return 0;
}

int main(int argc, char** argv)
{
	cout.precision(17);
	bool compare_with_ILP = false;
	int num_combo_runs = 2;
	int text_level = 0;
	//run_Miyauchi_nets();
	cout << "Starting random test set 1" << endl;
	run_CPP_rand_test_nets(1, compare_with_ILP, num_combo_runs, text_level);
	cout << "Starting random test set 2" << endl;
	run_CPP_rand_test_nets(2, compare_with_ILP, num_combo_runs, text_level);
	cout << "Starting random test set 3" << endl;
	run_CPP_rand_test_nets(3, compare_with_ILP, num_combo_runs, text_level);
	cout << "Starting GW real world test set" << endl;
	run_CPP_rw_test_nets(1, compare_with_ILP, num_combo_runs, text_level);
	cout << "Starting Oosten real world test set" << endl;
	run_CPP_rw_test_nets(2, compare_with_ILP, num_combo_runs, text_level);
	
	return 0;
}
