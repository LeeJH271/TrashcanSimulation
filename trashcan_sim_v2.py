import numpy as np
import matplotlib.pyplot as plt

import itertools

'''
# space parameter
sqrt_box_count = 700

# population parameter
population_count = 15888
trash_per_person = 0.5
trash_gen_prob = 2.31E-4
goto_next_trashcan_prob = 0.1

# trashcan parameter
trashcan_count = 500
'''

# space parameter
sqrt_box_count = 100

# population parameter
population_count = 100
trash_gen_prob = 2.31E-4
goto_next_trashcan_prob = 0.1

# trashcan parameter
trashcan_count = 50

# global variables
# population_matrix size : (person_index, 4)
population_matrix = np.array([])
# trashcan_matrix size : (sqrt_box_count, sqrt_box_count)
trashcan_matrix = np.array([])


# initializer
def initializer():
    global population_matrix, trashcan_matrix

    possible_coordinates = list(np.ndindex(sqrt_box_count, sqrt_box_count))
    population_matrix = np.concatenate(
        (np.array([possible_coordinates[i]
                   for i in np.random.choice(sqrt_box_count ** 2, size=population_count, replace=False)]),
         np.full((population_count, 2), -1)),
        axis=1
    )

    trashcan_matrix = np.full((sqrt_box_count, sqrt_box_count), -1, dtype=float)
    for row in range(15, sqrt_box_count, 31):
        for col in range(15, sqrt_box_count, 31):
            trashcan_matrix[row, col] = 0


# trashcan_finder : find the closest trashcan
def trashcan_finder(loc, find_current_loc):
    global trashcan_matrix

    row, col = loc[0], loc[1]

    if find_current_loc and trashcan_matrix[row, col] != -1:
        return loc
    else:
        i = 1
        is_end = False
        while not is_end:
            is_end = True
            next_loc_list = list(itertools.product(range(row - i, row + i + 1), [col - i, col + i])) + list(
                itertools.product([row - i, row + i], range(col - i + 1, col + i - 1 + 1)))

            for _row, _col in next_loc_list:
                if 0 <= _row < sqrt_box_count and 0 <= _col < sqrt_box_count:
                    is_end = False
                    if trashcan_matrix[_row, _col] != -1:
                        return [_row, _col]

            i += 1

        return [-1, -1]


def updator():
    global population_matrix, trashcan_matrix

    for person in population_matrix:
        row, col, obj_row, obj_col = person

        if obj_row != -1:
            if row == obj_row and col == obj_col:
                if (trashcan_matrix[row, col] == 10
                        and np.random.choice(2, p=[1 - goto_next_trashcan_prob, goto_next_trashcan_prob])):
                    person[2], person[3] = trashcan_finder([row, col], 0)
                else:
                    trashcan_matrix[row, col] += 0.5
                    person[2], person[3] = -1, -1

                    if trashcan_matrix[row, col] > 10:
                        return 1
            else:
                if row < obj_row:
                    person[0], person[1] = row + 1, col
                elif row > obj_row:
                    person[0], person[1] = row - 1, col
                elif col < obj_col:
                    person[0], person[1] = row, col + 1
                elif col > obj_col:
                    person[0], person[1] = row, col - 1

        elif np.random.choice(2, p=[1 - trash_gen_prob, trash_gen_prob]):
            person[2], person[3] = trashcan_finder([row, col], 1)

        else:
            next_loc_list = [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
            can_go_loc_index = [(row, col)]
            for loc in next_loc_list:
                if 0 <= loc[0] < sqrt_box_count and 0 <= loc[1] < sqrt_box_count:
                    can_go_loc_index.append(loc)
            next_loc = can_go_loc_index[np.random.randint(len(can_go_loc_index))]
            person[0], person[1] = next_loc[0], next_loc[1]

    return 0


def tester(control_prob):
    global goto_next_trashcan_prob
    goto_next_trashcan_prob = control_prob

    result = []
    for count in range(50):
        print("current step :", count)

        initializer()

        time = 0
        while True:
            # if there is full trashcan
            if updator():
                break
            else:
                time += 1

        result.append(time)

    return result


def main():
    cont_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(len(cont_probs)):
        result = tester(cont_probs[i])
        print(cont_probs[i])
        print(np.mean(np.array(result)))
        print(np.std(np.array(result)))


if __name__ == '__main__':
    main()