import glob
import networkx as nx
import random


def convert_gt_to_net():
	name_size = []
	for file_name in glob.glob("*.gt") + glob.glob("*.GT"):
		with open(file_name) as file:
			#print(file_name)
			max_row = 0
			max_col = 0
			connections = []
			next(file)
			for line in file:
				if len(line.strip()) == 0:
					continue
				connections.append(list(map(int, line.strip().split())))
				max_row = max(max_row, connections[-1][0])
				max_col = max(max_col, max(connections[-1][1:]))
			net = nx.Graph()
			for connection in connections:
				for i in range(1, len(connection)):
					net.add_edge(connection[0], connection[i] + max_row)
			#print(net.size())
			name_size.append((file_name, net.size()))
			net_file_name = file_name[:-2] + "net"
			nx.write_pajek(net, net_file_name)
	for name, size in sorted(name_size):
		print(name)


def main():
	convert_gt_to_net()


if __name__ == '__main__':
	main()