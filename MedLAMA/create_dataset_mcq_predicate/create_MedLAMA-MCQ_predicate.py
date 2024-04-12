import argparse
import ast
import random

import pandas as pd
import pyarrow as pa


def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Process input and output files.")

    # Add arguments
    parser.add_argument("--input", help="Input file path", required=True)
    parser.add_argument("--output", help="Output file path", required=True)

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the input and output file paths
    input_file = args.input
    output_file = args.output

    data_df = pd.read_json(path_or_buf=input_file, lines=True)

    # ----------------------------------------------------------
    # Convert to MC dataset by adding fields:
    #   choice1, choice2, choice3, choice4, answer
    # ----------------------------------------------------------
    # group instances by relation type
    dict_instances_same_relation = {}
    for idx_row in range(0, data_df.shape[0]):
        relation = data_df['rel'][idx_row]
        if relation not in dict_instances_same_relation:
            dict_instances_same_relation[relation] = []
        dict_instances_same_relation[relation].append(idx_row)

    # Add fields: choice1, choice2, choice3, choice4, answer, answer_idx
    choice1 = [None] * data_df.shape[0]
    choice2 = [None] * data_df.shape[0]
    choice3 = [None] * data_df.shape[0]
    choice4 = [None] * data_df.shape[0]
    answer = [None] * data_df.shape[0]
    answer_idx = [None] * data_df.shape[0]
    for idx_row in range(0, data_df.shape[0]):
        # relation type of the current instance
        relation = data_df['rel'][idx_row]
        # [Random Sampling] add field: answer by randomly sampling form the tail_names_list
        obj_list = ast.literal_eval(data_df['tail_names_list'][idx_row])
        answer[idx_row] = random.choices(obj_list, k=1)[0]
        # [Random Sampling] add fields: choice{1,2,3,4} with random position of answer
        exclude_list = obj_list
        choices = [None] * NUM_OPTIONS
        counter = 0
        while True:
            sampled_idx_r = random.choices(dict_instances_same_relation[relation])[0]
            sampled_neg_op = random.choices(ast.literal_eval(data_df['tail_names_list'][sampled_idx_r]))[0]
            if sampled_neg_op not in exclude_list:
                choices[counter] = sampled_neg_op
                exclude_list.extend(sampled_neg_op)
                counter += 1
            if counter == NUM_OPTIONS - 1:
                break
        # shuffle choices
        choices[NUM_OPTIONS - 1] = answer[idx_row]
        random.shuffle(choices)
        choice1[idx_row] = choices[0]
        choice2[idx_row] = choices[1]
        choice3[idx_row] = choices[2]
        choice4[idx_row] = choices[3]
        answer_idx[idx_row] = str(choices.index(answer[idx_row]))
    tb_c1 = pa.array(choice1)
    tb_c2 = pa.array(choice2)
    tb_c3 = pa.array(choice3)
    tb_c4 = pa.array(choice4)
    tb_as = pa.array(answer)
    tb_asi = pa.array(answer_idx)

    data_df['opa'] = tb_c1
    data_df['opb'] = tb_c2
    data_df['opc'] = tb_c3
    data_df['opd'] = tb_c4
    data_df['cop'] = tb_asi
    data_df['answer'] = tb_as

    with open(output_file, "w") as f:
        f.write(data_df.to_json(orient='records', lines=True, force_ascii=False))


if __name__ == "__main__":
    NUM_OPTIONS = 4
    main()
