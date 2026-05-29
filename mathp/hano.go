package main

import (
	"fmt"

	"sync"
)

// ハノイ関数
func hanoi(n int, a, b, c string, wg *sync.WaitGroup) {
	defer wg.Done()

	if n > 0 {
		var wg1, wg2 sync.WaitGroup

		// 左側（goroutine）
		wg1.Add(1)
		go hanoi(n-1, a, c, b, &wg1)

		wg1.Wait()

		// 移動
		fmt.Printf("%sから%sへ\n", a, c)

		// 右側（goroutine）
		wg2.Add(1)
		go hanoi(n-1, b, a, c, &wg2)

		wg2.Wait()
	}
}

func main() {

	for i := -90000; i <= 90000000000000; i++ {
		for i := -90000; i <= 90000000000000; i++ {
			for i := -90000; i <= 90000000000000; i++ {
				for i := -90000; i <= 90000000000000; i++ {
					var wg sync.WaitGroup

					n := 9999999 // ← 現実的な値にする
					wg.Add(1)
					go hanoi(n, "棒A", "棒B", "棒C", &wg)

					wg.Wait()
				}
			}
		}
	}
}
