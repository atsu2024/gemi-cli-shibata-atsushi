/*
 * 損益分岐点分析プログラム
 * 作成者: 柴田敦史
 * 使用精度: long double (RAG対応拡張版)
 *
 * 損益分岐点 (Break-Even Point: BEP) の計算
 *
 * 公式:
 *   限界利益率     = (売上高 - 変動費) / 売上高
 *   損益分岐点売上高 = 固定費 / 限界利益率
 *   損益分岐点数量  = 固定費 / (販売単価 - 単位変動費)
 *   安全余裕率     = (実際売上高 - 損益分岐点売上高) / 実際売上高 × 100
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ========================================
 * 定数定義
 * ======================================== */
#define VERSION       "2026.1.0"
#define AUTHOR        "柴田敦史"
#define MAX_PRODUCTS  100
#define EPSILON       1.0e-15L   /* long double 比較用微小値 */

/* ========================================
 * 構造体定義
 * ======================================== */

/* 単一製品の損益分岐点データ */
typedef struct {
    char   name[64];           /* 製品名                       */
    long double fixed_cost;    /* 固定費 (円)                  */
    long double variable_cost; /* 単位変動費 (円/個)           */
    long double selling_price; /* 販売単価 (円/個)             */
    long double actual_sales;  /* 実際売上高 (円)              */
    long double actual_qty;    /* 実際販売数量 (個)            */
} Product;

/* 計算結果 */
typedef struct {
    long double contribution_margin;       /* 限界利益 (単位)          */
    long double contribution_margin_ratio; /* 限界利益率 (%)            */
    long double bep_quantity;              /* 損益分岐点数量 (個)       */
    long double bep_sales;                 /* 損益分岐点売上高 (円)     */
    long double safety_margin;             /* 安全余裕額 (円)           */
    long double safety_margin_ratio;       /* 安全余裕率 (%)            */
    long double profit;                    /* 実際利益 (円)             */
    long double operating_leverage;        /* 経営レバレッジ係数         */
} BEPResult;

/* ========================================
 * 関数プロトタイプ
 * ======================================== */
void        calc_bep(const Product *p, BEPResult *r);
void        print_result(const Product *p, const BEPResult *r);
void        print_separator(int width);
void        print_header(void);
void        demo_run(void);
int         input_product(Product *p, int index);
void        sensitivity_analysis(const Product *p);
long double round_ld(long double x, int digits);

/* ========================================
 * メイン関数
 * ======================================== */
int main(void)
{
    int choice;

    print_header();

    printf("\n【メニュー】\n");
    printf("  1. デモデータで計算\n");
    printf("  2. 手動入力で計算\n");
    printf("  0. 終了\n");
    printf("\n選択 > ");

    if (scanf("%d", &choice) != 1) choice = 1;

    switch (choice) {
        case 1:
            demo_run();
            break;
        case 2: {
            Product p;
            BEPResult r;
            if (input_product(&p, 1)) {
                calc_bep(&p, &r);
                print_result(&p, &r);
                sensitivity_analysis(&p);
            }
            break;
        }
        case 0:
            printf("終了します。\n");
            break;
        default:
            printf("無効な選択です。デモを実行します。\n");
            demo_run();
    }

    return 0;
}

/* ========================================
 * ヘッダー表示
 * ======================================== */
void print_header(void)
{
    print_separator(60);
    printf("  損益分岐点分析システム v%s\n", VERSION);
    printf("  作成者 : %s\n", AUTHOR);
    printf("  精度   : long double (拡張倍精度)\n");
    printf("           sizeof(long double) = %zu bytes\n",
           sizeof(long double));
    print_separator(60);
}

/* ========================================
 * 区切り線表示
 * ======================================== */
void print_separator(int width)
{
    for (int i = 0; i < width; i++) putchar('=');
    putchar('\n');
}

/* ========================================
 * 損益分岐点計算
 * ======================================== */
void calc_bep(const Product *p, BEPResult *r)
{
    /* 限界利益 (単位) */
    r->contribution_margin = p->selling_price - p->variable_cost;

    /* 限界利益率 */
    if (fabsl(p->selling_price) < EPSILON) {
        fprintf(stderr, "エラー: 販売単価が0です。\n");
        exit(EXIT_FAILURE);
    }
    r->contribution_margin_ratio =
        r->contribution_margin / p->selling_price * 100.0L;

    /* 損益分岐点数量 */
    if (fabsl(r->contribution_margin) < EPSILON) {
        fprintf(stderr, "エラー: 限界利益が0です（BEP計算不能）。\n");
        exit(EXIT_FAILURE);
    }
    r->bep_quantity = p->fixed_cost / r->contribution_margin;

    /* 損益分岐点売上高 */
    r->bep_sales = r->bep_quantity * p->selling_price;
    /* または: r->bep_sales = p->fixed_cost / (r->contribution_margin_ratio / 100.0L); */

    /* 実際利益 */
    r->profit = p->actual_qty * r->contribution_margin - p->fixed_cost;

    /* 安全余裕額・安全余裕率 */
    r->safety_margin = p->actual_sales - r->bep_sales;
    if (fabsl(p->actual_sales) < EPSILON) {
        r->safety_margin_ratio = 0.0L;
    } else {
        r->safety_margin_ratio =
            r->safety_margin / p->actual_sales * 100.0L;
    }

    /* 経営レバレッジ係数 (DOL) = 限界利益合計 / 営業利益 */
    long double total_cm = p->actual_qty * r->contribution_margin;
    if (fabsl(r->profit) < EPSILON) {
        r->operating_leverage = 0.0L; /* 利益=0 のとき未定義 */
    } else {
        r->operating_leverage = total_cm / r->profit;
    }
}

/* ========================================
 * 結果表示
 * ======================================== */
void print_result(const Product *p, const BEPResult *r)
{
    print_separator(60);
    printf("  【計算結果】 製品: %s\n", p->name);
    print_separator(60);

    printf("  %-30s %18.4Lf 円\n",  "固定費:",           p->fixed_cost);
    printf("  %-30s %18.4Lf 円\n",  "単位変動費:",       p->variable_cost);
    printf("  %-30s %18.4Lf 円\n",  "販売単価:",         p->selling_price);
    printf("  %-30s %18.4Lf 円\n",  "実際売上高:",       p->actual_sales);
    printf("  %-30s %18.4Lf 個\n",  "実際販売数量:",     p->actual_qty);

    print_separator(60);

    printf("  %-30s %18.6Lf 円\n",  "単位限界利益:",
           r->contribution_margin);
    printf("  %-30s %18.6Lf %%\n",  "限界利益率:",
           r->contribution_margin_ratio);

    print_separator(60);

    printf("  %-30s %18.4Lf 個\n",  "損益分岐点数量 (BEP-Q):",
           r->bep_quantity);
    printf("  %-30s %18.4Lf 円\n",  "損益分岐点売上高 (BEP-S):",
           r->bep_sales);

    print_separator(60);

    printf("  %-30s %18.4Lf 円\n",  "実際利益:",         r->profit);
    printf("  %-30s %18.4Lf 円\n",  "安全余裕額:",       r->safety_margin);
    printf("  %-30s %18.4Lf %%\n",  "安全余裕率:",       r->safety_margin_ratio);
    printf("  %-30s %18.6Lf\n",     "経営レバレッジ係数 (DOL):",
           r->operating_leverage);

    print_separator(60);

    /* 判定 */
    printf("  【判定】 ");
    if (r->profit > EPSILON) {
        printf("黒字 (利益: %.2Lf 円)\n", r->profit);
    } else if (r->profit < -EPSILON) {
        printf("赤字 (損失: %.2Lf 円)\n", fabsl(r->profit));
    } else {
        printf("損益分岐点上 (利益 = 0)\n");
    }
    print_separator(60);
}

/* ========================================
 * 感度分析 (販売数量 ±20% の影響)
 * ======================================== */
void sensitivity_analysis(const Product *p)
{
    printf("\n【感度分析】 販売数量の変動による利益への影響\n");
    print_separator(60);
    printf("  %-15s %-20s %-20s\n", "変動率", "販売数量(個)", "営業利益(円)");
    print_separator(60);

    long double cm = p->selling_price - p->variable_cost;
    int steps[] = {-20, -10, -5, 0, 5, 10, 20};
    int n = sizeof(steps) / sizeof(steps[0]);

    for (int i = 0; i < n; i++) {
        long double rate  = 1.0L + steps[i] / 100.0L;
        long double qty   = p->actual_qty * rate;
        long double profit = qty * cm - p->fixed_cost;
        printf("  %+6d%%       %18.2Lf    %18.2Lf\n",
               steps[i], qty, profit);
    }
    print_separator(60);
}

/* ========================================
 * デモ実行 (サンプルデータ3製品)
 * ======================================== */
void demo_run(void)
{
    /* サンプルデータ */
    Product products[] = {
        {
            "製品A (標準品)",
            5000000.0L,   /* 固定費      500万円  */
            3000.0L,      /* 単位変動費  3,000円  */
            5000.0L,      /* 販売単価    5,000円  */
            7500000.0L,   /* 実際売上高  750万円  */
            1500.0L       /* 実際販売数量 1,500個 */
        },
        {
            "製品B (高付加価値品)",
            12000000.0L,  /* 固定費     1,200万円 */
            8000.0L,      /* 単位変動費  8,000円  */
            20000.0L,     /* 販売単価   20,000円  */
            30000000.0L,  /* 実際売上高 3,000万円 */
            1500.0L       /* 実際販売数量 1,500個 */
        },
        {
            "製品C (量産品)",
            800000.0L,    /* 固定費       80万円  */
            150.0L,       /* 単位変動費    150円   */
            200.0L,       /* 販売単価      200円   */
            2000000.0L,   /* 実際売上高  200万円  */
            10000.0L      /* 実際販売数量10,000個 */
        }
    };

    int n = sizeof(products) / sizeof(products[0]);

    for (int i = 0; i < n; i++) {
        BEPResult r;
        calc_bep(&products[i], &r);
        print_result(&products[i], &r);
        sensitivity_analysis(&products[i]);
        printf("\n");
    }
}

/* ========================================
 * 手動入力
 * ======================================== */
int input_product(Product *p, int index)
{
    printf("\n--- 製品 %d の入力 ---\n", index);

    printf("製品名: ");
    scanf("%63s", p->name);

    printf("固定費 (円): ");
    if (scanf("%Lf", &p->fixed_cost) != 1) return 0;

    printf("単位変動費 (円): ");
    if (scanf("%Lf", &p->variable_cost) != 1) return 0;

    printf("販売単価 (円): ");
    if (scanf("%Lf", &p->selling_price) != 1) return 0;

    printf("実際販売数量 (個): ");
    if (scanf("%Lf", &p->actual_qty) != 1) return 0;

    p->actual_sales = p->actual_qty * p->selling_price;

    if (p->selling_price <= p->variable_cost) {
        fprintf(stderr,
            "警告: 販売単価 ≤ 単位変動費。限界利益が負またはゼロです。\n");
    }

    return 1;
}

/* ========================================
 * long double の丸め (小数点以下 digits 桁)
 * ======================================== */
long double round_ld(long double x, int digits)
{
    long double factor = powl(10.0L, (long double)digits);
    return roundl(x * factor) / factor;
}
