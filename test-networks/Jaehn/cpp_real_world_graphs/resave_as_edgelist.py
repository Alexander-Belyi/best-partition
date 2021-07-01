import networkx as nx


def generate_net_from_features(features, name):
	n = len(features)
	p = len(features[0])
	G = nx.complete_graph(n)
	pos_sum = 0
	for (u,v,w) in G.edges(data=True):
		not_equal = 0
		equal = 0
		for a, b in zip(features[u], features[v]):
			if name != "cetacea":
				a = int(a)
				b = int(b)
			if name == "micro":
				not_equal += int(abs(a - b) > max((a, b, 1)) * 0.3)
			elif a == '*' or b == '*':
				pass
			else:
				if a == b:
					equal += 1
				else:
					not_equal += 1
		if name != "cetacea":
			w['weight'] = p - 2 * not_equal
		else:
			w['weight'] = equal - not_equal
		if w['weight'] > 0:
			pos_sum += w['weight']
	#print(u, v, equal, not_equal, p - 2 * not_equal)
	print(pos_sum)
	return G


def read_features(name):
	features = []
	feature_file_name = "Grotschel-Wakabayashi/" + name + ".csv"
	if name == 'UNO_1b' or name == 'UNO_1a':
		feature_file_name = "Grotschel-Wakabayashi/UNO_1.csv"
	if name == 'UNO_2b' or name == 'UNO_2a':
		feature_file_name = "Grotschel-Wakabayashi/UNO_2.csv"
	with open(feature_file_name) as fin:
		for line in fin:
			row = line.strip().split()
			if name == 'UNO_1b' and '3' in row:
				continue
			if name == 'UNO_2b' and row.count('3') >= 8:
				continue
			features.append(row)
	return features


def set1():
	net_names = ['UNO_2b']#'UNO_2a', 'UNO_1b', 'UNO_1a', 'UNO', 'cars', 'workers', 'cetacea', "wild_cats", "micro"]
	for name in net_names:
		features = read_features(name)
		net = generate_net_from_features(features, name)
		nx.write_edgelist(net, "Grotschel-Wakabayashi/"+name+".edgelist", data=['weight'])


def read_table(file_name):
	table = []
	with open(file_name) as fin:
		row1 = next(fin).split(' ')
		n, m = list(map(int, row1))
		table = [[-1] * m for _ in range(n)]
		for line in fin:
			row = line.split(' ')
			i, j = list(map(int, row))
			table[i-1][j-1] = 1
	return table


def generate_group_tech_net(table):
	n = len(table)
	m = len(table[0])
	G = nx.complete_graph(n + m)
	for (u,v,w) in G.edges(data=True):
		w['weight'] = 0
	for i in range(n):
		for j in range(m):
			G[i][j+n]['weight'] = table[i][j]
	return G


def set2():
	net_names = ["BOC"]#, "MCC", "SEI", "SUL", "KKV"]
	for name in net_names:
		table_file_name = "Oosten/" + name + ".csv"
		table = read_table(table_file_name)
		net = generate_group_tech_net(table)
		nx.write_edgelist(net, "Oosten/"+name+".edgelist", data=['weight'])


def main():
	set2()


if __name__ == '__main__':
	main()