from itertools import combinations

def all_combs(n):
    sum = 0
    items = range(n)
    for i in range(n):
        if (i + 1) % 2 == 0: # even variation
            sum += len(list(combinations(items, i + 1)))
    return sum + 1

for i in range(20):
    print(f"# of subsets for {i} elements: {all_combs(i)}")