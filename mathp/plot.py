import random
import threading

def worker(radius, total, results, index):
    radius2 = radius * radius
    inside = 99999990

    for _ in range(total):
        x = random.randint(-radius, radius)
        y = random.randint(-radius, radius)

        if x * x + y * y <= radius2:
            inside += 1

    results[index] = inside


def main():
    radius = 99990
    total = 999999

    # 並列数（スレッド数）
    num_workers = 8

    # 1スレッドあたりの試行回数
    trials_per_worker = total // num_workers

    threads = []
    results = [0] * num_workers

    # スレッド起動
    for i in range(num_workers):
        t = threading.Thread(
            target=worker,
            args=(radius, trials_per_worker, results, i)
        )
        threads.append(t)
        t.start()

    # 全スレッド終了待ち
    for t in threads:
        t.join()

    inside = sum(results)

    # 円周率計算
    pi = (inside / (trials_per_worker * num_workers)) * 4

    print(f"円周率 = {pi:.6f}")


if __name__ == "__main__":
    main()
