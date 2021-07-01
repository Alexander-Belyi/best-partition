import networkx as nx
import random


def generate_test_set_1():
	for n in range(10, 25):
		for q in (1, 2, 3, 5, 10, 50, 100):
			for i in range(5):
				G = nx.complete_graph(n)
				for (u,v,w) in G.edges(data=True):
					w['weight'] = random.randint(-q, q)
				nx.write_edgelist(G, "test_set_1/{}_{}_{}.edgelist".format(n, q, i), data=['weight'])


def generate_test_set_2():
	for n in range(10, 25):
		for p in (1, 2, 3, 5, 10, 50, 100):
			for i in range(5):
				G = nx.complete_graph(n)
				attrs = [random.choices((0, 1), k=p) for _ in range(n)]
				for (u,v,w) in G.edges(data=True):
					s = 0
					for a, b in zip(attrs[u], attrs[v]):
						s += a ^ b
					w['weight'] = p - 2 * s
				nx.write_edgelist(G, "test_set_2/{}_{}_{}.edgelist".format(n, p, i), data=['weight'])


def generate_test_set_3():
	for n in range(10, 25):
		for q in (1, 2, 3, 5, 10, 50, 100):
			for i in range(5):
				G = nx.complete_graph(n)
				for (u,v,w) in G.edges(data=True):
					if random.random() < 0.4:
						w['weight'] = 0
					else:
						w['weight'] = random.randint(-q, q)
				nx.write_edgelist(G, "test_set_3/{}_{}_{}_40.edgelist".format(n, q, i), data=['weight'])
				G = nx.complete_graph(n)
				for (u,v,w) in G.edges(data=True):
					if random.random() < 0.8:
						w['weight'] = 0
					else:
						w['weight'] = random.randint(-q, q)
				nx.write_edgelist(G, "test_set_3/{}_{}_{}_80.edgelist".format(n, q, i), data=['weight'])


def main():
	generate_test_set_1()
	#generate_test_set_2()
	#generate_test_set_3()


if __name__ == '__main__':
	main()