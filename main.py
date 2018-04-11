from decision_tree import DecisionTree


def main():
    training_data = [
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 1]
    ]
    design_matrix = [row[:-1] for row in training_data]
    target_values = [row[-1] for row in training_data]
    decision_tree = DecisionTree()
    decision_tree.fit(design_matrix, target_values)
    predictions = decision_tree.predict(design_matrix)
    decision_tree.print()


if __name__ == '__main__':
    main()
