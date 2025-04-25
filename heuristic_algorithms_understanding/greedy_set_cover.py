def greedy_set_cover(universe, subsets):
    uncovered = set(universe)
    cover = []
# Example usage
if __name__ == "__main__":
    U = {1, 2, 3, 4, 5}
    S = [{1, 2}, {2, 3, 4}, {4, 5}]

    result = greedy_set_cover(U, S)

    print("Selected subsets for cover:")
    for s in result:
        print(s)
