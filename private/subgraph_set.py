from .molecule_graph import MolKey

class SubGraphSet():
    def __init__(self, init_subgraphs, subgraphs_idx, inputs):
        self.subgraphs = init_subgraphs
        self.subgraphs_idx = subgraphs_idx
        self.inputs = inputs
        self.map_to_input = self.get_map_to_input()

    def get_map_to_input(self):
        '''
        Input:
        init_subgraphs: a list with length equal to # input graphs, elements are Chem.Mol
        subgraphs_idx: a three-level list
                the first level: the same length with # input graphs, each element corresponds to a input graph
                the second level: the same length with # sub_graphs of the input graph, each element corresponds to a sub_graph
                the third level: the same length with # atoms of the sub_graph, each element corresponds to an atom index of input graph
        Output:
        map_to_input: a dict, [key_of_subgraphs][0][key_of_inputs][the_matched_subgraph][atom_idx_of_input]
                              [key_of_subgraphs][1]
        '''
        map_to_input = dict()
        for i, input_i in enumerate(self.inputs):
            key_input = MolKey(input_i)
            subgraphs_idx_i = self.subgraphs_idx[i]
            for j, subgraph_idx in enumerate(subgraphs_idx_i):
                key_subgraph = MolKey(self.subgraphs[i][j])
                subg = input_i.subgraphs[input_i.subgraphs_idx.index(subgraph_idx)]
                if key_subgraph not in map_to_input.keys():
                    map_to_input[key_subgraph] = [dict(), 1]
                else:
                    map_to_input[key_subgraph][1] += 1
                if key_input not in map_to_input[key_subgraph][0].keys():
                    map_to_input[key_subgraph][0][key_input] = list()
                map_to_input[key_subgraph][0][key_input].append((subgraph_idx, subg))
        return map_to_input
    
    def update(self, input_graphs):
        new_subgraphs = []
        new_subgraphs_idx = []
        new_inputs = []
        # Could merge with get_map_to_input()
        for i, input_i in enumerate(input_graphs):
            new_subgraphs.append(input_i.subgraphs)
            new_subgraphs_idx.append(input_i.subgraphs_idx)
            new_inputs.append(input_i)
        self.subgraphs = new_subgraphs
        self.subgraphs_idx = new_subgraphs_idx
        self.inputs = new_inputs
        self.map_to_input = self.get_map_to_input()

