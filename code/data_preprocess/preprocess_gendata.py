'''
Code adapted from the ParaRel repo.
'''
import argparse
import glob
import random
import os
import csv
import sys
import io
sys.path.append("YOUR_PROJECT_PATH")
sys.path.append("YOUR_PROJECT_PATHpararel/data/pattern_data/graphs/")
from pararel.pararel.consistency import utils
import torch
import pickle
from tqdm import tqdm
import json

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        # print(module)
        if module == "pararel.patterns.graph_types":
            renamed_module = "pararel.pararel.patterns.graph_types"

        return super(RenameUnpickler, self).find_class(renamed_module, name)

def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()

def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)

def generate_data(num_relations, num_tuples, relations_given, LAMA_path):

    graph_path = "YOUR_PROJECT_PATHpararel/data/pattern_data/graphs/"
    relations_path = glob.glob(graph_path + "*.graph")
    output_path = "YOUR_PROJECT_PATHoutput/data"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    random.shuffle(relations_path)
    relation_path_keep = []
    metadata = "_"
    if relations_given != "":
        relations_given = sorted(relations_given.split(","))
        for relation_path in relations_path:
            relation = relation_path.split("/")[-1].split(".")[0]
            if relation in relations_given:
                relation_path_keep.append(relation_path)
        metadata += "_".join(relations_given)
        metadata += "-"
    if len(relation_path_keep) < num_relations:
        for relation_path in relations_path:
            if relation_path not in relation_path_keep:
                relation = relation_path.split("/")[-1].split(".")[0]
                relation_path_keep.append(relation_path)
                metadata += relation
                metadata += "-"
                if len(relation_path_keep) == num_relations:
                    break
    metadata = metadata.strip("-")
    output_path = output_path + str(num_tuples) + "_" + str(
        num_relations) + metadata + "/"

    if not os.path.exists(output_path):
        print("Saving data to: ", output_path)
        os.mkdir(output_path)

    output_path_true = output_path + "train_"
    output_path_mlm = output_path + "train_mlm.txt"
    f_mlm = open(output_path_mlm, "w")

    for relation_path in relation_path_keep:
        with open(relation_path, "rb") as f:
            graph = renamed_load(f)
        relation = relation_path.split("/")[-1].split(".")[0]
        data = utils.read_jsonl_file(LAMA_path + relation + ".jsonl")
        random.shuffle(data)
        
        cur_r_sub = list(set(d['sub_label'] for d in data))

        for i, d in enumerate(data):
            random.shuffle(data)
            for node in graph.nodes():
                pattern = node.lm_pattern
                pattern = pattern.replace("[X]", d["sub_label"])
                pattern = pattern.replace("[Y]", "[MASK]")
                pattern_mlm = pattern.replace("[MASK]", d["obj_label"])

                f_true.write(pattern)
                f_true.write("\n")
                f_mlm.write(pattern_mlm)
                f_mlm.write("\n")

            f_true.write("\n")

            if i >= num_tuples:
                break

        f_true.close()

    f_mlm.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_relations',
                        '-nr',
                        type=int,
                        default=3,
                        help='number of relations')
    parser.add_argument('--num_tuples',
                        '-nt',
                        type=int,
                        default=100,
                        help='number of tuples')
    parser.add_argument('--relations_given',
                        '-r',
                        type=str,
                        default="P138,P449,P37",
                        help='which relations')
    parser.add_argument('--LAMA_path',
                        '-lama',
                        type=str,
                        default="./pararel/data/trex_lms_vocab/",
                        help='number of tuples')
    args = parser.parse_args()
    generate_data(args.num_relations, args.num_tuples, args.relations_given,
                  args.LAMA_path)

if __name__ == "__main__":
    main()