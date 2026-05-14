#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <math.h>

/**
 * biot_savart_precision.c
 * 
 * ビオ・サバールの法則を用いた高精度磁場計算プログラム (long double版)
 * 参照画像（qrgosa02.png 等）の「誤差 (gosa)」を考慮した数値計算デモです。
 * 
 * 公式: dH = (I / 4π) * (dl × r) / |r|^3
 */

typedef struct {
    long double x, y, z;
} Vector3L;

// ベクトルの外積 (Cross Product)
Vector3L cross_product(Vector3L a, Vector3L b) {
    Vector3L res;
    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;
    return res;
}

// ベクトルのノルム（長さ）
long double vector_norm(Vector3L a) {
    return sqrtl(a.x * a.x + a.y * a.y + a.z * a.z);
}

int main() {
    // 数学定数 PI (long double)
    const long double PI_L = acosl(-1.0L);
    
    // 物理定数・設定
    long double current = 1.0L; // 電流 I [A]
    
    // 観測点 P (Z軸上の点)
    Vector3L p = {0.0L, 0.0L, 0.1L}; 

    // 電流素片の位置 source と方向ベクトル dl
    // (X軸付近にある微小電流要素)
    Vector3L source = {0.05L, 0.0L, 0.0L}; 
    Vector3L dl     = {0.0L, 0.001L, 0.0L}; 

    // 距離ベクトル r = P - source
    Vector3L r = {p.x - source.x, p.y - source.y, p.z - source.z};
    long double r_mag = vector_norm(r);

    if (r_mag < 1e-20L) {
        printf("エラー: 観測点が電流素片に近すぎます。\n");
        return 1;
    }

    // ビオ・サバールの法則の計算
    // dH = (I / (4 * PI * r^3)) * (dl cross r)
    Vector3L dl_xr = cross_product(dl, r);
    long double factor = current / (4.0L * PI_L * powl(r_mag, 3.0L));

    Vector3L dh;
    dh.x = dl_xr.x * factor;
    dh.y = dl_xr.y * factor;
    dh.z = dl_xr.z * factor;

    // 結果の表示
    printf("=== ビオ・サバール高精度計算 (long double) ===\n");
    printf("電流 (I):         %.5Lf A\n", current);
    printf("観測点 P:         (%.5Lf, %.5Lf, %.5Lf)\n", p.x, p.y, p.z);
    printf("素片位置 Source:  (%.5Lf, %.5Lf, %.5Lf)\n", source.x, source.y, source.z);
    printf("距離 |r|:         %.15Lf\n", r_mag);
    printf("\n");
    printf("磁場成分 dH:\n");
    printf("  Hx: %+.25Le\n", dh.x);
    printf("  Hy: %+.25Le\n", dh.y);
    printf("  Hz: %+.25Le\n", dh.z);

    // double型との比較
    double d_r_mag = (double)r_mag;
    double d_factor = (double)current / (4.0 * M_PI * pow(d_r_mag, 3.0));
    printf("\n--- 標準 double 型との精度比較 ---\n");
    printf("  Hz (double): %+.25e\n", (double)dl_xr.z * d_factor);
    printf("  Difference:  %+.25Le (long double の有効性)\n", dh.z - (long double)((double)dl_xr.z * d_factor));

    return 0;
}
