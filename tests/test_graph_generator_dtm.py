import numpy as np

from graph_generator import Graph_generator


def make_line_graph_generator():
    graph_generator = Graph_generator(map_size=(10, 10), k_size=3, sensor_range=80)
    graph_generator.node_coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [1.0, 1.0],
    ])
    for node in range(4):
        graph_generator.graph.add_node(str(node))

    graph_generator.graph.add_edge('0', '1', 1.0)
    graph_generator.graph.add_edge('1', '2', 1.0)
    graph_generator.graph.add_edge('1', '3', 1.0)
    return graph_generator


def test_shortest_path_tree_returns_first_hops_on_undirected_graph():
    graph_generator = make_line_graph_generator()

    distances, first_hops, reachable = graph_generator.get_shortest_path_tree(0)

    np.testing.assert_allclose(distances, np.array([0.0, 1.0, 2.0, 2.0]))
    np.testing.assert_array_equal(first_hops, np.array([0, 1, 1, 1]))
    np.testing.assert_array_equal(reachable, np.array([True, True, True, True]))


def test_decayed_route_memory_uses_max_decay_semantics():
    graph_generator = make_line_graph_generator()
    graph_generator.route_node = [
        graph_generator.node_coords[0],
        graph_generator.node_coords[1],
    ]

    memory = graph_generator.get_decayed_route_memory(gamma=0.5, sigma=1.0, window=2).reshape(-1)

    assert memory[1] == 1.0
    np.testing.assert_allclose(memory[0], 0.5)
    np.testing.assert_allclose(memory[2], np.exp(-1.0))
    np.testing.assert_allclose(memory[3], np.exp(-1.0))
