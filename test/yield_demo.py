# 直接看这3个例子就够了，不要多想，自己跑一遍

# 例1：不用 yield → 全部算完再返回，一次性占内存
def square_list(nums):
    result = []
    for n in nums:
        result.append(n * n)
    return result

# 例2：用 yield → 一个一个返回，省内存
def square_yield(nums):
    for n in nums:
        yield n * n

# 例3：yield 的本质 → 就是个"可暂停的函数"
def count_up():
    print("开始")
    yield 1
    print("暂停后继续")
    yield 2
    print("再继续")
    yield 3
    print("结束")

if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5]

    print("--- 不用 yield ---")
    result = square_list(nums)
    print(result)  # 一次性得到 [1, 4, 9, 16, 25]

    print("\n--- 用 yield ---")
    result = square_yield(nums)
    print(result)  # 不是列表，是生成器对象
    for x in result:
        print(f"拿到: {x}")  # 每次迭代才计算一个值

    print("\n--- 暂停/继续 ---")
    g = count_up()
    print("调用 count_up() 后什么也没打印，因为还没开始迭代")
    x = next(g)
    print(f"拿到: {x}\n")
    x = next(g)
    print(f"拿到: {x}\n")
    x = next(g)
    print(f"拿到: {x}\n")