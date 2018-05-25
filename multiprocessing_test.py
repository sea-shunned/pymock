import random
import multiprocessing
from functools import partial

def main(seed_num):
    random.seed(seed_num)
    return random.randint(0,10), random.random()

def main2(arg1, arg2, seed_num):
    random.seed(seed_num)
    return random.randint(0,10), random.random()

if __name__ == '__main__':
    seed_list = [(1,),(2,),(3,),(4,),(5,)]

    with multiprocessing.Pool() as pool:
        results = pool.starmap(main, seed_list)
    print(results)

    non_mp_res = []
    for i in seed_list:
        # print(i[0])
        random.seed(i[0])
        non_mp_res.append(random.randint(0,10))
    print(non_mp_res)

    func = partial(main2, arg1=1, arg2=4)
    print(func)

    func = partial(main2, 1, 4)
    print(func)
    with multiprocessing.Pool() as pool:
        results = pool.starmap(func, seed_list)
    print(results)

    print("\n")
    func = partial(main2, *[1, 4])
    print(func)
    with multiprocessing.Pool() as pool:
        results = pool.starmap(func, seed_list)
    print(results)


    a_dict = {
        "arg1": 1,
        "arg2": 4
    }
    # print(list(a_dict.values()))

