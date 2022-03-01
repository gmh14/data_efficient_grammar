import torch
import torch.multiprocessing as mp
import numpy as np
import fcntl
import argparse
import setproctitle
from retro_star.api import RSPlanner
from rdkit import Chem


def lock(f):
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        return False
    return True


class Synthesisability():
    def __init__(self):
        self.planner = RSPlanner(
                gpu=-1,
                starting_molecules='./retro_star/dataset/origin_dict.csv',
                use_value_fn=True,
                iterations=50,
                expansion_topk=50)

    def get_syn_rate(self, mol_list):
        assert type(mol_list) == list
        syn_flag = []
        for i, mol_sml in enumerate(mol_list):
            result = self.planner.plan(Chem.MolToSmiles(mol_sml))
            if result:
                syn_flag.append(result['succ'])
            else:
                syn_flag.append(False)
        return np.mean(syn_flag)


def main(proc_id, filename, output_filename):
    syn = Synthesisability()
    while(True):
        selected_mol = None
        with open(filename, 'r') as f:
            editable = lock(f)
            if editable:
                lines = f.readlines()
                num_samples = len(lines)
                new_lines = []
                for idx, line in enumerate(lines):
                    splitted_line = line.strip().split()
                    if len(splitted_line) == 1 and (selected_mol is None):
                        selected_mol = (idx, splitted_line[0])
                        new_line = "{} {}\n".format(splitted_line[0], "working")
                    else:
                        new_line = "{}\n".format(" ".join(splitted_line))
                    new_lines.append(new_line)
                with open(filename, 'w') as fw:
                    for _new_line in new_lines:
                        fw.write(_new_line)
                fcntl.flock(f, fcntl.LOCK_UN)
        if selected_mol is None:
            continue

        print("====Working for sample {}/{}====".format(selected_mol[0], num_samples))
        try:
            result = syn.planner.plan(selected_mol[1])
        except:
            result = None

        while(True):
            with open(output_filename, 'a') as f:
                editable = lock(f)
                if editable:
                    f.write("{} {} {}\n".format(selected_mol[0], selected_mol[1], "False" if result is None else "True"))
                    fcntl.flock(f, fcntl.LOCK_UN)
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='retro* listener')
    parser.add_argument('--proc_id', type=int, default=1, help="process id")
    parser.add_argument('--filename', type=str, default="generated_samples.txt", help="file name to lister")
    parser.add_argument('--output_filename', type=str, default="output_syn.txt", help="file name to output")
    args = parser.parse_args()
    setproctitle.setproctitle("retro_star_listener")
    main(args.proc_id, args.filename, args.output_filename)
