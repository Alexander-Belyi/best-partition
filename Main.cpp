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

string TESTS_FOLDER = "../test-networks/";

ComboAlgorithm COMBO(7, 0, 0, 0);

vector<SolutionInfo> run_CPP_test(Graph& G, BnBParameters bnb_params,
	bool compare_with_ILP = false, int num_combo_runs = 2, int text_level = 0)
{
	clock_t time_start = clock();
	cout.precision(6);
	double mod_combo = 0;
	vector<size_t> communities;
	for (int i = 0; i < num_combo_runs; ++i) {
		COMBO.Run(G, nullopt, i & 1);
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
		SolutionInfo ILP_info = BestPartitionILP(G, nullopt, 0);
		ILP_info.run_time = double(clock() - time_start) / CLOCKS_PER_SEC;
		if (abs(ILP_info.optimal_solution - info.optimal_solution) > EPS)
			cerr << "ERROR: optimal_solution by ILP != optimal_solution" << endl;
		if (abs(info.optimal_solution - info.best_estimate) > EPS)
			cerr << "ERROR: optimal_solution != best_estimate" << endl;
		res = {info, ILP_info};
	}
	return res;
}

int run_CPP_rand_test_nets(int test_set, bool compare_with_ILP = false, int num_combo_runs = 2, int text_level = 0)
{
	string path = TESTS_FOLDER+"/Jaehn/cpp_random_graphs/test_set_" + to_string(test_set) + "/";
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
	string path = TESTS_FOLDER+"/Jaehn/cpp_real_world_graphs/Grotschel-Wakabayashi/";
	vector<string> file_names = {"wild_cats", "cars", "workers", "cetacea", "micro", "UNO", "UNO_1a", "UNO_1b", "UNO_2a", "UNO_2b"};
	BnBParameters bnb_params;
	if (test_set == 2) {
		path = TESTS_FOLDER+"/Jaehn/cpp_real_world_graphs/Oosten/";
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

int run_Miyauchi_nets(int text_level)
{
	string path = TESTS_FOLDER+"/Miyauchi/modularity/";
	vector<string> network_file_names = {
		"Zachary Karate.net",
		"Dolphins Social Network.net",
		"Les Miserables_unit.net",
		//"Political Books.net",
		//"American College Football.net",
		//"USAir97.net",
		//"s838.net",
		//"netscience.net",
		//"power-grid.net"
	};
	for(const string& net_name : network_file_names) {
		Graph G = ReadGraphFromFile(path + net_name);
		cout << net_name << " size = " << G.Size() << endl;
		clock_t time_start = clock();
		COMBO.Run(G, nullopt, 1);
		cout << "Combo's result = " << G.Modularity()
			 << " estimated UB chains = " << EstimateUB_chains_simplex(G, text_level)
			 << " estimated UB chains+stars = " << EstimateUB_chains_and_stars(G, text_level)
		 	 << ". Time: " << double(clock() - time_start) / CLOCKS_PER_SEC << endl;
	}
	return 0;
}

int run_reduction_ILP_test(int test_set, int text_level, bool use_ILP_solver = true)
{
	string path;
	vector<string> network_file_names;
	if (test_set == 1) {
		path = TESTS_FOLDER+"/Lorena/";
		network_file_names = {
			//"counter_example.net",
			"lesmis.net",
			"GD00-a_main.net",
			"ca-sandi-auths.net",
			"rt-retweet.net",
			"netscience_main.net",
			"bio-DM-LC.net",
			//"power-494-bus.net",
			"bio-diseasome.net",
			"bio-grid-mouse.net",
			"ca-CSphd.net"
		};
	} else if (test_set == 2) {
		path = TESTS_FOLDER+"/Jaehn/cpp_real_world_graphs/Grotschel-Wakabayashi/";
		network_file_names = {
			"wild_cats.edgelist",
			"cars.edgelist",
			"workers.edgelist",
			"cetacea.edgelist",
			"micro.edgelist",
			"UNO.edgelist",
			"UNO_1a.edgelist",
			"UNO_1b.edgelist",
			"UNO_2a.edgelist",
			"UNO_2b.edgelist"
			};
	}
	for(const string& net_name : network_file_names) {
		Graph G;
		if (test_set == 1)
			G = ReadGraphFromFile(path + net_name);
		else
			G = ReadGraphFromFile(path + net_name, 1.0, true);
		cout << net_name << " size = " << G.Size() << endl;
		clock_t time_start = clock();
		G.MergeIdenticalNodes();
		cout << "Size after merging identical = " << G.Size() << endl;
		G.MergeStronglyConnected();
		cout << "Size after merging strongly connected = " << G.Size() << endl;
		cout << "Preprocessing time: " << double(clock() - time_start) / CLOCKS_PER_SEC << endl;
		if (use_ILP_solver) {
			time_start = clock();
			SolutionInfo ILP_info = BestPartitionILP(G, nullopt, -5);
			ILP_info.run_time = double(clock() - time_start) / CLOCKS_PER_SEC;
			cout << "Max modularity by exact solution of ILP: "
				<< ILP_info.optimal_solution
				<< ", visited " << ILP_info.num_visited_nodes << " nodes, "
				<< "Time elapsed: " << double(clock() - time_start) / CLOCKS_PER_SEC << endl;
		}
		cout << endl;
	}
	return 0;
}

int parse_input(int argc, char** argv) {
	int experiment = 0;
	if (argc == 2 && string(argv[0]) == "1") {
		experiment = 1;
	} else if (argc == 2 && string(argv[0]) == "2") {
		experiment = 1;
	} else {
		while (experiment == 0) {
			cout << "Please, enter \"1\" to run the program reproducing results of \"Subnetwork Constraints ...\" paper,\n"
			     << "or enter \"2\" to run the program reproducing results of \"Network size reduction ...\" paper.\n";
			string input;
			cin >> input;
			if (input == "1")
				experiment = 1;
			else if (input == "2")
				experiment = 2;
		}
	}
	return experiment;
}

int main(int argc, char** argv)
{
	int experiment = parse_input(argc, argv);
	cout.precision(17);
	bool compare_with_ILP = false;
	int num_combo_runs = 2;
	int text_level = 0;
	if (experiment == 1) {
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
		cout << "Starting Miyauchi real world test set optimizing modularity" << endl;
		run_Miyauchi_nets(text_level);
	} else if (experiment == 2) {
		cout << "Starting solving CPP using ILP solver for the GW real world test set" << endl << endl;
		run_reduction_ILP_test(2, text_level);
		cout << "Starting modularity optimization using ILP solver for the Lorena real world test set" << endl << endl;
		run_reduction_ILP_test(1, text_level);
	}
	
	return 0;
}
