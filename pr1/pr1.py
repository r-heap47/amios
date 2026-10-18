#!/usr/bin/env python3
"""АМиОС, практическая работа №1.
Моделирование и исследование случайных величин и последовательностей.

Из равномерного датчика U(0,1) методом обратной функции получаются:
  1) равномерная СВ на [0; alpha];
  2) показательная СВ с параметром lambda;
  3) треугольная СВ на [0; a] с модой в 0 (a=0, c=0, b=a — см. README, п. «Противоречия»).

Для каждой СВ считаются оценки МО, дисперсии и СКО на выборках разного объёма,
доверительный интервал для МО, критерий Колмогорова, строятся гистограммы
с наложенной теоретической плотностью. Объём выборки выбирается из заданной
относительной точности оценки МО и СКО (п. 2 порядка выполнения в методичке).

Запуск:
    python3 pr1.py --variant N            # N — номер варианта 1..30 из п. 1.6 методички
    python3 pr1.py --variant N --delta 0.01 --seed 2303 --out results
Результаты: <out>/results.json, <out>/*.png и заполненный отчёт (по умолчанию report.md
рядом со скриптом, ПЕРЕЗАПИСЫВАЕТ черновик; другой путь: --report FILE, отключить: --no-report).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

# Перечень заданий, п. 1.6 «Пособия по практическим занятиям» (Романцев, Филатов, 2016):
# вариант: (alpha, 1/lambda, a)
VARIANTS = {
    1: (70, 150, 90), 2: (110, 20, 170), 3: (130, 130, 170), 4: (200, 190, 120),
    5: (70, 180, 90), 6: (180, 190, 200), 7: (10, 50, 170), 8: (20, 200, 190),
    9: (60, 200, 140), 10: (200, 90, 190), 11: (20, 150, 70), 12: (110, 130, 110),
    13: (80, 100, 110), 14: (130, 50, 80), 15: (90, 50, 160), 16: (190, 130, 80),
    17: (170, 40, 200), 18: (130, 60, 20), 19: (70, 190, 30), 20: (110, 190, 140),
    21: (120, 110, 30), 22: (80, 110, 190), 23: (40, 200, 180), 24: (100, 120, 10),
    25: (60, 170, 10), 26: (100, 200, 160), 27: (80, 40, 10), 28: (20, 160, 110),
    29: (160, 60, 130), 30: (200, 110, 20),
}

Z = 1.959964  # квантиль N(0,1) уровня 0.975 (доверительная вероятность 0.95)


def make_laws(alpha, inv_lam, a):
    """Описание трёх законов: обратная функция, плотность, теоретические моменты."""
    lam = 1.0 / inv_lam
    return {
        "uniform": dict(
            title=f"Равномерный закон на [0; {alpha}]",
            inverse=lambda u: alpha * u,                       # F(y) = y/alpha
            inverse_tex=f"y = {alpha}·u",
            pdf=lambda y: np.where((y >= 0) & (y <= alpha), 1.0 / alpha, 0.0),
            mean=alpha / 2, var=alpha ** 2 / 12,
            kurt=1.8,                                          # β2 = μ4/σ⁴
            scipy=stats.uniform(loc=0, scale=alpha),
            xmax=alpha,
        ),
        "exponential": dict(
            title=f"Показательный закон, λ = 1/{inv_lam}",
            inverse=lambda u: -inv_lam * np.log1p(-u),         # F(y) = 1 - exp(-λy)
            inverse_tex=f"y = -{inv_lam}·ln(1 - u)",
            pdf=lambda y: np.where(y >= 0, lam * np.exp(-lam * np.clip(y, 0, None)), 0.0),
            mean=inv_lam, var=inv_lam ** 2,
            kurt=9.0,
            scipy=stats.expon(scale=inv_lam),
            xmax=None,
        ),
        "triangular": dict(
            title=f"Треугольный закон на [0; {a}], мода в 0",
            inverse=lambda u: a * (1 - np.sqrt(1 - u)),        # F(y) = 1 - (a-y)²/a²
            inverse_tex=f"y = {a}·(1 - √(1 - u))",
            pdf=lambda y: np.where((y >= 0) & (y <= a), 2 * (a - y) / a ** 2, 0.0),
            mean=a / 3, var=a ** 2 / 18,
            kurt=2.4,
            scipy=stats.triang(c=0, loc=0, scale=a),
            xmax=a,
        ),
    }


def required_n(law, delta):
    """Объём выборки для относительной точности delta оценок МО и СКО при P = 0.95.
    МО:  n >= (z·σ / (δ·M))²
    СКО: Var(s) ≈ σ²(β2 - 1)/(4n)  =>  n >= z²(β2 - 1) / (4δ²)
    """
    cv = math.sqrt(law["var"]) / law["mean"]
    n_mean = math.ceil((Z * cv / delta) ** 2)
    n_std = math.ceil(Z ** 2 * (law["kurt"] - 1) / (4 * delta ** 2))
    return n_mean, n_std


def estimate(sample, law):
    n = len(sample)
    m = float(sample.mean())
    d = float(sample.var(ddof=1))
    s = math.sqrt(d)
    half = Z * s / math.sqrt(n)
    ks = stats.kstest(sample, law["scipy"].cdf)
    return dict(n=n, mean=m, var=d, std=s, ci_low=m - half, ci_high=m + half,
                mean_in_ci=(m - half) <= law["mean"] <= (m + half),
                rel_err_mean=abs(m - law["mean"]) / law["mean"],
                rel_err_std=abs(s - math.sqrt(law["var"])) / math.sqrt(law["var"]),
                ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))


def plot_hist(sample, law, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(sample)
    k = int(min(60, max(8, round(1 + 3.322 * math.log10(n)) * 2)))  # ~2× правило Стёрджеса
    hi = law["xmax"] if law["xmax"] is not None else float(np.quantile(sample, 0.995))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=130)
    ax.hist(sample, bins=k, range=(0, hi), density=True, color="#2a78d6",
            edgecolor="#fcfcfb", linewidth=1.0, label="гистограмма (оценка плотности)")
    xs = np.linspace(0, hi, 400)
    ax.plot(xs, law["pdf"](xs), color="#0b0b0b", linewidth=2, label="теоретическая плотность")
    ax.set_title(f"{law['title']}, n = {n}", fontsize=10, color="#0b0b0b")
    ax.set_xlabel("y", color="#52514e")
    ax.set_ylabel("f(y)", color="#52514e")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def build_report(variant, params, delta, seed, results, out_rel):
    """Отчёт в Markdown по разделам шаблона ЛЭТИ; в .docx собирается ../lab_docx.py."""
    alpha, inv_lam, a = params
    lines = []
    L = lines.append
    L("---")
    L("work: 1")
    L("topic: Моделирование и исследование случайных величин и последовательностей")
    L(f"variant: {variant}")
    L("listing: pr1.py")
    L("---")
    L("")
    L("## Цель работы")
    L("")
    L("Вспомнить свойства и способ построения случайной величины и освоить её моделирование. Задачи:")
    L("")
    L("1. Рассмотреть способ построения функции над заданной случайной величиной для получения "
      "случайной величины с заданным законом распределения.")
    L("2. Смоделировать этот процесс.")
    L("3. Оценить результаты.")
    L("")
    L("## Основные теоретические положения")
    L("")
    L("### Задание")
    L("")
    L(f"С помощью датчика равномерно распределённых случайных чисел смоделировать (вариант №{variant}):")
    L("")
    L(f"1. СВ, распределённую равномерно на [0; α], α = {alpha}.")
    L(f"2. СВ, распределённую по показательному закону, λ = 1/{inv_lam}.")
    L(f"3. СВ, распределённую по треугольному закону на [0; a] с модой в нуле, a = {a}.")
    L("")
    L("Для каждой СВ построить гистограммы, рассчитать МО, дисперсию и СКО; выбрать объём выборки "
      "по заданной точности оценки МО и СКО.")
    L("")
    L("### Метод обратной функции")
    L("")
    L("Если u — значение СВ U, равномерной на (0; 1), а F — непрерывная строго возрастающая функция "
      "распределения, то Y = F⁻¹(U) имеет функцию распределения F, так как P(Y < y) = P(U < F(y)) = F(y). "
      "Формула преобразования получается решением уравнения F(y) = u относительно y.")
    L("")
    L("Таблица: Законы распределения, функции преобразования и теоретические характеристики")
    L("")
    L("| Закон | F(y) | y = F⁻¹(u) | M | D | σ |")
    L("|---|---|---|---:|---:|---:|")
    cdf = {"uniform": f"y/{alpha}", "exponential": f"1 − exp(−y/{inv_lam})", "triangular": f"1 − ({a} − y)²/{a}²"}
    for key, r in results.items():
        law = r["law"]
        L(f"| {law['title']} | {cdf[key]} | {law['inverse_tex']} | {fmt(law['mean'])} | {fmt(law['var'], 1)} | "
          f"{fmt(math.sqrt(law['var']))} |")
    L("")
    L("### Объём выборки и проверка закона")
    L("")
    L(f"Объём выборки выбирается по относительной точности δ = {delta:.0%} при доверительной вероятности 0,95 "
      "(z = 1,96):")
    L("")
    L("- для МО: n ≥ (z·σ / (δ·M))²;")
    L("- для СКО: дисперсия оценки s приближённо равна σ²(β₂ − 1)/(4n), где β₂ — коэффициент эксцесса "
      "(1,8 у равномерного, 9 у показательного, 2,4 у треугольного закона), отсюда n ≥ z²(β₂ − 1)/(4δ²).")
    L("")
    L("Доверительный интервал для МО: M* ± z·s/√n. Соответствие выборки заданному закону проверяется "
      "критерием Колмогорова на уровне значимости 0,05.")
    L("")
    L("Средство моделирования: Python 3 (numpy, scipy, matplotlib). Датчик: `numpy.random.default_rng` (PCG64), "
      f"зерно {seed}.")
    L("")
    L("## Экспериментальные результаты")
    L("")
    for key, r in results.items():
        law = r["law"]
        L(f"Таблица: Оценки характеристик, {law['title'][0].lower() + law['title'][1:]}")
        L("")
        L("| n | M* | D* | σ* | 95% ДИ для M | M в ДИ | Колмогоров D | p-value |")
        L("|---:|---:|---:|---:|---|:-:|---:|---:|")
        for e in r["estimates"]:
            L(f"| {e['n']} | {fmt(e['mean'])} | {fmt(e['var'], 1)} | {fmt(e['std'])} | "
              f"[{fmt(e['ci_low'], 2)}; {fmt(e['ci_high'], 2)}] | {'да' if e['mean_in_ci'] else 'нет'} | "
              f"{fmt(e['ks_stat'], 4)} | {fmt(e['ks_p'], 3)} |")
        L(f"| теория | {fmt(law['mean'])} | {fmt(law['var'], 1)} | {fmt(math.sqrt(law['var']))} | | | | |")
        L("")
    L("## Обработка результатов эксперимента")
    L("")
    L("Таблица: Требуемый объём выборки и относительные ошибки оценок при этом объёме")
    L("")
    L(f"| Закон | n по МО | n по СКО | принят n | ошибка M | ошибка σ |")
    L("|---|---:|---:|---:|---:|---:|")
    for key, r in results.items():
        e = r["estimates"][[x["n"] for x in r["estimates"]].index(max(r["n_req"]))]
        L(f"| {r['law']['title']} | {r['n_req'][0]} | {r['n_req'][1]} | {max(r['n_req'])} | "
          f"{e['rel_err_mean']:.2%} | {e['rel_err_std']:.2%} |")
    L("")
    for key, r in results.items():
        law = r["law"]
        L("Таблица: Относительные ошибки оценок, " + law["title"][0].lower() + law["title"][1:])
        L("")
        L("| n | ошибка M | ошибка σ |")
        L("|---:|---:|---:|")
        for e in r["estimates"]:
            L(f"| {e['n']} | {e['rel_err_mean']:.2%} | {e['rel_err_std']:.2%} |")
        L("")
        for e in r["estimates"]:
            L(f"![{law['title']}, n = {e['n']}]({out_rel}/{key}_{e['n']}.png)")
            L("")
    L("## Выводы")
    L("")
    L("<!-- Проверьте выводы по своим числам после запуска. -->")
    L("1. Метод обратной функции позволяет получить из равномерного датчика СВ с равномерным, показательным "
      "и треугольным законами распределения; формулы преобразования выведены из уравнения F(y) = u.")
    L("2. С ростом объёма выборки оценки МО и СКО сходятся к теоретическим значениям, а погрешность "
      "убывает примерно как 1/√n; гистограммы приближаются к теоретической плотности.")
    L("3. При одинаковой относительной точности показательному закону нужна самая большая выборка, "
      "потому что у него наибольшие коэффициент вариации (1) и эксцесс (9).")
    L("4. Критерий Колмогорова на уровне 0,05 не отвергает гипотезу о заданном законе "
      "(сверьте значения p-value в таблицах).")
    L("")
    L("## Приложение А. Исходный код программы")
    L("")
    L(f"Запуск: `python3 pr1.py --variant {variant} --delta {delta} --seed {seed}`.")
    L("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", type=int, required=True, choices=sorted(VARIANTS))
    ap.add_argument("--delta", type=float, default=0.01, help="относительная точность оценки МО и СКО")
    ap.add_argument("--seed", type=int, default=2303)
    ap.add_argument("--sizes", default="100,1000,10000", help="объёмы выборок для сравнения")
    ap.add_argument("--out", default=None, help="папка результатов (по умолчанию results_vNN рядом со скриптом)")
    ap.add_argument("--report", default=None, help="куда записать отчёт (по умолчанию report.md рядом со скриптом)")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    out = Path(args.out) if args.out else here / f"results_v{args.variant:02d}"
    out.mkdir(parents=True, exist_ok=True)

    params = VARIANTS[args.variant]
    laws = make_laws(*params)
    rng = np.random.default_rng(args.seed)
    base_sizes = [int(s) for s in args.sizes.split(",")]

    results, dump = {}, {"variant": args.variant, "alpha": params[0], "inv_lambda": params[1],
                         "a": params[2], "delta": args.delta, "seed": args.seed, "laws": {}}
    for key, law in laws.items():
        n_req = required_n(law, args.delta)
        sizes = sorted(set(base_sizes + [max(n_req)]))
        ests = []
        for n in sizes:
            u = rng.random(n)
            y = law["inverse"](u)
            e = estimate(y, law)
            ests.append(e)
            plot_hist(y, law, out / f"{key}_{n}.png", key)
        results[key] = dict(law=law, n_req=n_req, estimates=ests)
        dump["laws"][key] = dict(title=law["title"], mean=law["mean"], var=law["var"],
                                 std=math.sqrt(law["var"]), n_required_mean=n_req[0],
                                 n_required_std=n_req[1], estimates=ests)

    (out / "results.json").write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")

    for key, r in results.items():
        print(f"== {r['law']['title']}: теория M={r['law']['mean']:.3f} σ={math.sqrt(r['law']['var']):.3f}; "
              f"n_req(M)={r['n_req'][0]}, n_req(σ)={r['n_req'][1]}")
        for e in r["estimates"]:
            print(f"   n={e['n']:>7}  M*={e['mean']:9.3f}  σ*={e['std']:9.3f}  "
                  f"ДИ=[{e['ci_low']:.2f}; {e['ci_high']:.2f}]  KS p={e['ks_p']:.3f}")

    if not args.no_report:
        report = Path(args.report) if args.report else here / "report.md"
        try:
            out_rel = out.resolve().relative_to(report.resolve().parent).as_posix()
        except ValueError:
            out_rel = out.resolve().as_posix()
        report.write_text(build_report(args.variant, params, args.delta, args.seed, results, out_rel),
                          encoding="utf-8")
        print(f"Отчёт: {report}")
    print(f"Результаты: {out}")


if __name__ == "__main__":
    main()
