# qsort関数の定義
def qsort(data, top, tail):
  # 要素が1つなら処理を行わない
  if top == tail:
    return

  # 配列の真ん中の要素を基準値とする
  pivot = data[(top + tail) // 2]

  # 残りの要素を基準値より大きいグループと小さいグループに分ける
  i = top
  j = tail
  while (True):
    # 配列の前から後に向かって基準値より小さい要素を探す
    while (data[i] < pivot):
      i += 1
    # 配列の後から前に向かって基準値より大きい要素を探す
    while (data[j] > pivot):
      j -= 1
    # 以下の条件が成り立つならグループ分けが完了している
    if (i >= j):
      break
    # data[i]とdata[j]の値を交換する
    temp = data[i]
    data[i] = data[j]
    data[j] = temp
    # 先の要素に進む
    i += 1
    j -= 1
  # 基準値より小さいグループで同じ処理を繰り返す
  if (top < i - 1):
    qsort(data, top, i - 1)

  # 基準値より大きいグループで同じ処理を繰り返す
  if (tail > j + 1):
    qsort(data, j + 1, tail)

# qsort関数を使うプログラム
data = [55, 22, 77, 44, 11, 66, 33]
qsort(data, 0, len(data) - 1)
print(data)