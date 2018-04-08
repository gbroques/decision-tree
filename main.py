from decision_tree import DecisionTree


def main():
    training_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 3, 'Apple'],
        ['Red', 1, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon']
    ]
    decision_tree = DecisionTree()
    tree = decision_tree.build_tree(training_data)
    print(tree)


if __name__ == '__main__':
    main()
