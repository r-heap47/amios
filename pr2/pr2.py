#!/usr/bin/env python3
"""АМиОС, практическая работа №2. Моделирование центра массового обслуживания (ЦМО)
с неограниченной очередью: один прибор, дисциплина FIFO, пуассоновский входной поток
(интервалы ~ Exp со средним 1/λ) и три закона времени обслуживания со средним t_об = ρ/λ:
  uniform     — равномерный на [0; 2·t_об]           (ν² = 1/3)
  exponential — показательный со средним t_об         (ν² = 1)
  triangular  — треугольный на [0; 3·t_об], мода в 0  (ν² = 1/2)
(такие же законы, как в примере ПР2 из папки «Примеры практических работ».)

Модель — дискретно-событийная, через рекурсию Линдли: W_i = max(0, W_{i-1} + S_{i-1} - A_i).
Характеристики: QM, QA, QZ, QT, QX (очередь), FR, FT (прибор).
Теория (формула Поллачека — Хинчина, п. 2.2 методички):
  r̄ = ρ²(1+ν²) / (2(1-ρ)),   t̄_ож = r̄ / λ,   t̄_об = 1/μ = ρ/λ.

Запуск:
    python3 pr2.py --variant N                # N — номер варианта 1..10 из п. 2.6 методички
    python3 pr2.py --variant N --mean-arrival 50 --seed 2303
Результаты: results_vNN/ (results.json, гистограммы времени ожидания) и report.md рядом со скриптом
(перезаписывает черновик; --report FILE — другой путь, --no-report — без отчёта).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

# Перечень заданий, п. 2.6 методички: вариант -> ((ρ1, ρ2), (N1, N2))
VARIANTS = {
    1: ((0.50, 0.70), (1000, 50000)),
    2: ((0.55, 0.90), (1500, 40000)),
    3: ((0.45, 0.80), (2000, 55000)),
    4: ((0.40, 0.75), (1500, 45000)),
    5: ((0.45, 0.85), (1750, 47500)),
    6: ((0.40, 0.70), (1000, 55000)),
    7: ((0.50, 0.65), (2000, 50000)),
    8: ((0.60, 0.80), (1000, 55000)),
    9: ((0.60, 0.85), (1500, 47500)),
    10: ((0.55, 0.75), (1000, 47500)),
}

LAWS = {
    "uniform": dict(title="равномерный", nu2=1 / 3),
    "exponential": dict(title="показательный", nu2=1.0),
    "triangular": dict(title="треугольный", nu2=0.5),
}


def service_sampler(law, t_ob, rng, n):
    u = rng.random(n)
    if law == "uniform":
        return 2 * t_ob * u
    if law == "exponential":
        return -t_ob * np.log1p(-u)
    if law == "triangular":
        return 3 * t_ob * (1 - np.sqrt(1 - u))
    raise ValueError(law)


def theory(rho, lam, nu2):
    r = rho ** 2 * (1 + nu2) / (2 * (1 - rho))
    return dict(QA=r, QT=r / lam, FT=rho / lam, FR=rho)


def simulate(n, rho, law, mean_arrival, rng):
    """Прогон до выхода n заявок из прибора."""
    lam = 1.0 / mean_arrival
    t_ob = rho / lam
    a = rng.exponential(mean_arrival, n)          # интервалы между заявками
    s = service_sampler(law, t_ob, rng, n)        # времена обслуживания
    arr = np.cumsum(a)
    w = np.empty(n)
    w[0] = 0.0
    wi = 0.0
    for i in range(1, n):                         # рекурсия Линдли
        wi = wi + s[i - 1] - a[i]
        if wi < 0:
            wi = 0.0
        w[i] = wi
    start = arr + w
    dep = start + s
    T = float(dep[-1])                            # модельное время окончания прогона
    # длина очереди во времени: +1 при входе в очередь с ожиданием, -1 при выходе на прибор
    waiting = w > 0
    ev_t = np.concatenate([arr[waiting], start[waiting]])
    ev_d = np.concatenate([np.ones(waiting.sum()), -np.ones(waiting.sum())])
    order = np.lexsort((ev_d, ev_t))              # при равенстве времени сначала выход
    qm = int(np.max(np.cumsum(ev_d[order]))) if waiting.any() else 0
    qa = float(w.sum() / T)                       # средняя по времени длина очереди
    qz = int((~waiting).sum())
    qt = float(w.mean())
    qx = float(w[waiting].mean()) if waiting.any() else 0.0
    fr = float(s.sum() / T)
    ft = float(s.mean())
    return dict(N=n, QM=qm, QA=qa, QZ=qz, QT=qt, QX=qx, FR=fr, FT=ft), w


def wait_table(w, width, nbins=10):
    """Таблица «число заявок в зависимости от времени пребывания в очереди».
    Первая строка — заявки без ожидания, далее интервалы (0; width), [width; 2·width), ..., [.., ∞)."""
    n = len(w)
    bins = [("0 (без ожидания)", w <= 0), (f"(0; {width:g})", (w > 0) & (w < width))]
    for k in range(1, nbins - 1):
        bins.append((f"[{k * width:g}; {(k + 1) * width:g})", (w >= k * width) & (w < (k + 1) * width)))
    bins.append((f"≥ {(nbins - 1) * width:g}", w >= (nbins - 1) * width))
    rows, cum = [], 0
    for label, mask in bins:
        c = int(mask.sum())
        cum += c
        rows.append(dict(label=label, count=c, pct=100 * c / n, cum_pct=100 * cum / n))
    return rows


def plot_wait(w, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=130)
    w = w[w > 0]                                  # без нулевых ожиданий: их доля — в таблице
    hi = float(np.quantile(w, 0.99)) if len(w) else 1.0
    ax.hist(w, bins=40, range=(0, hi), color="#2a78d6", edgecolor="#fcfcfb", linewidth=1.0)
    ax.set_title(title, fontsize=10, color="#0b0b0b")
    ax.set_xlabel("время ожидания в очереди", color="#52514e")
    ax.set_ylabel("число заявок", color="#52514e")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_transient(trans, th_qt, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=130)
    ns = [t["N"] for t in trans]
    ax.plot(ns, [t["QT"] for t in trans], color="#2a78d6", linewidth=2, marker="o", markersize=4,
            label="QT, моделирование")
    ax.axhline(th_qt, color="#0b0b0b", linewidth=1.2, linestyle="--", label="QT, теория")
    ax.set_xscale("log")
    ax.set_title(title, fontsize=10, color="#0b0b0b")
    ax.set_xlabel("N (число заявок, лог. шкала)", color="#52514e")
    ax.set_ylabel("среднее время ожидания", color="#52514e")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def f(x, nd=3):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", type=int, required=True, choices=sorted(VARIANTS))
    ap.add_argument("--mean-arrival", type=float, default=50.0,
                    help="средний интервал между заявками 1/λ (в методичке не задан; в примере 50)")
    ap.add_argument("--seed", type=int, default=2303)
    ap.add_argument("--transient", default="1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000",
                    help="значения N для оценки переходного процесса")
    ap.add_argument("--reps", type=int, default=10, help="число повторных экспериментов (п. 6)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    out = Path(args.out) if args.out else here / f"results_v{args.variant:02d}"
    out.mkdir(parents=True, exist_ok=True)
    (rho1, rho2), (n1, n2) = VARIANTS[args.variant]
    lam = 1.0 / args.mean_arrival
    rng = np.random.default_rng(args.seed)

    main_runs = []  # п. 3–4: 3 закона × 2 ρ × 2 N
    for law, ld in LAWS.items():
        for rho in (rho1, rho2):
            th = theory(rho, lam, ld["nu2"])
            for n in (n1, n2):
                res, w = simulate(n, rho, law, args.mean_arrival, rng)
                main_runs.append(dict(law=law, rho=rho, theory=th, **res))
                if n == n2:
                    plot_wait(w, f"Exp/{ld['title']}, ρ = {rho}, N = {n}, ожидавшие заявки", out / f"wait_{law}_{rho}.png")
                    main_runs[-1]["wait_table"] = wait_table(w, width=args.mean_arrival)

    # п. 5: переходный процесс — QT при росте N
    trans_ns = [int(x) for x in args.transient.split(",")]
    transient = {}
    for law, ld in LAWS.items():
        for rho in (rho1, rho2):
            th = theory(rho, lam, ld["nu2"])
            rows = []
            for n in trans_ns:
                res, _ = simulate(n, rho, law, args.mean_arrival, rng)
                rows.append(res)
            transient[f"{law}_{rho}"] = dict(law=law, rho=rho, theory=th, rows=rows)
            plot_transient(rows, th["QT"], f"Exp/{ld['title']}, ρ = {rho}: QT(N)",
                           out / f"transient_{law}_{rho}.png")

    # п. 6–7: 10 независимых прогонов Exp/Exp при ρ2, N2
    reps = []
    for _ in range(args.reps):
        res, _ = simulate(n2, rho2, "exponential", args.mean_arrival, rng)
        reps.append(res)
    th_rep = theory(rho2, lam, 1.0)
    ci = {}
    tcrit = float(stats.t.ppf(0.975, args.reps - 1))
    for key in ("QT", "QA", "FR", "FT"):
        x = np.array([r[key] for r in reps])
        m, s = float(x.mean()), float(x.std(ddof=1))
        h = tcrit * s / math.sqrt(len(x))
        ci[key] = dict(mean=m, std=s, low=m - h, high=m + h, theory=th_rep[key],
                       inside=(m - h) <= th_rep[key] <= (m + h))

    dump = dict(variant=args.variant, rho=[rho1, rho2], N=[n1, n2], mean_arrival=args.mean_arrival,
                seed=args.seed, main=main_runs, transient=transient, reps=reps, ci=ci, t_crit=tcrit)
    (out / "results.json").write_text(json.dumps(dump, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")

    print(f"Вариант {args.variant}: ρ = {rho1}, {rho2}; N = {n1}, {n2}; 1/λ = {args.mean_arrival}")
    print(f"{'закон':12} {'ρ':>5} {'N':>7} {'QM':>4} {'QA':>7} {'QZ':>7} {'QT':>9} {'QX':>9} {'FR':>6} {'FT':>7} | теория QA, QT, FT")
    for r in main_runs:
        th = r["theory"]
        print(f"{r['law']:12} {r['rho']:5.2f} {r['N']:7d} {r['QM']:4d} {r['QA']:7.3f} {r['QZ']:7d} {r['QT']:9.3f} "
              f"{r['QX']:9.3f} {r['FR']:6.3f} {r['FT']:7.3f} | {th['QA']:.3f}, {th['QT']:.3f}, {th['FT']:.3f}")
    print("10 прогонов Exp/Exp:", {k: (round(v['low'], 3), round(v['high'], 3), round(v['theory'], 3)) for k, v in ci.items()})

    if not args.no_report:
        report = Path(args.report) if args.report else here / "report.md"
        try:
            rel = out.resolve().relative_to(report.resolve().parent).as_posix()
        except ValueError:
            rel = out.resolve().as_posix()
        report.write_text(build_report(args, (rho1, rho2), (n1, n2), lam, main_runs, transient, reps, ci,
                                       tcrit, rel), encoding="utf-8")
        print(f"Отчёт: {report}")
    print(f"Результаты: {out}")


def build_report(args, rhos, ns, lam, main_runs, transient, reps, ci, tcrit, rel):
    """Отчёт в Markdown по разделам шаблона ЛЭТИ; в .docx собирается ../lab_docx.py."""
    L = []
    A = L.append
    A("---")
    A("work: 2")
    A("topic: Моделирование центра массового обслуживания")
    A(f"variant: {args.variant}")
    A("listing: pr2.py")
    A("---")
    A("")
    A("## Цель работы")
    A("")
    A("Изучить модель обслуживания заявок с неограниченной очередью. Задачи:")
    A("")
    A("1. Изучить модель со схожими и с различными законами распределения потоков заявок и обслуживания.")
    A("2. Запрограммировать модель.")
    A("3. Сравнить результаты моделирования с теоретическими.")
    A("")
    A("## Основные теоретические положения")
    A("")
    A("### Задание")
    A("")
    A("Смоделировать одноканальную СМО с неограниченной очередью, пуассоновским потоком заявок и тремя законами "
      "времени обслуживания (равномерным, показательным, треугольным). Для каждой пары законов провести "
      "эксперимент при двух значениях приведённой интенсивности ρ и двух значениях числа заявок N, "
      "рассчитать теоретические значения, оценить переходный процесс, провести 10 экспериментов для "
      "показательных законов и построить доверительные интервалы.")
    A("")
    A(f"Вариант №{args.variant}: ρ₁ = {rhos[0]}, ρ₂ = {rhos[1]}, N₁ = {ns[0]}, N₂ = {ns[1]}.")
    A("")
    A("### Модель")
    A("")
    A(f"- Интервалы между заявками распределены по показательному закону, 1/λ = {args.mean_arrival:g} "
      f"(λ = {lam:.4f}). Методичка значение λ не задаёт, поэтому оно выбрано так же, как в примере.")
    A("- Среднее время обслуживания t_об = 1/μ = ρ/λ. Законы обслуживания с этим средним: равномерный "
      "на [0; 2·t_об] (ν² = 1/3), показательный (ν² = 1), треугольный на [0; 3·t_об] с модой в 0 (ν² = 1/2); "
      "ν — коэффициент вариации времени обслуживания.")
    A("- Дисциплина FIFO, один прибор. Прогон заканчивается, когда систему покинут N заявок.")
    A("- Средство моделирования: Python 3 (numpy, scipy, matplotlib), дискретно-событийная модель на рекурсии "
      f"Линдли W_i = max(0, W_(i−1) + S_(i−1) − A_i); зерно датчика {args.seed}.")
    A("")
    A("### Теоретические формулы")
    A("")
    A("Для СМО M/G/1 по формуле Поллачека — Хинчина:")
    A("")
    A("- средняя длина очереди r̄ = ρ²(1 + ν²) / (2(1 − ρ));")
    A("- среднее время ожидания t̄_ож = r̄ / λ (формула Литтла);")
    A("- среднее время обслуживания t̄_об = ρ / λ; коэффициент загрузки прибора равен ρ.")
    A("")
    A("Обозначения: QM — максимальная длина очереди, QA — средняя длина очереди, QZ — число заявок, "
      "обслуженных без ожидания, QT — среднее время ожидания по всем заявкам, QX — среднее время ожидания "
      "заявок, которые ждали, FR — коэффициент загрузки прибора, FT — среднее время обслуживания.")
    A("")
    A("Таблица: Теоретические значения характеристик")
    A("")
    A("| Закон обслуживания | ρ | μ | ν² | QA (r̄) | QT (t̄_ож) | FR | FT (t̄_об) |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for law, ld in LAWS.items():
        for rho in rhos:
            th = theory(rho, lam, ld["nu2"])
            A(f"| {ld['title']} | {rho} | {lam / rho:.5f} | {ld['nu2']:.3f} | {f(th['QA'])} | {f(th['QT'], 2)} | "
              f"{rho} | {f(th['FT'], 2)} |")
    A("")
    A("## Экспериментальные результаты")
    A("")
    for law, ld in LAWS.items():
        A(f"Таблица: Результаты моделирования, обслуживание по закону: {ld['title']}")
        A("")
        A("| ρ | N | QM | QA | QZ | QT | QX | FR | FT |")
        A("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for rho in rhos:
            rows = [r for r in main_runs if r["law"] == law and r["rho"] == rho]
            th = rows[0]["theory"]
            for r in rows:
                A(f"| {rho} | {r['N']} | {r['QM']} | {f(r['QA'])} | {r['QZ']} | {f(r['QT'])} | {f(r['QX'])} | "
                  f"{f(r['FR'])} | {f(r['FT'])} |")
            A(f"| {rho} | теория | | {f(th['QA'])} | | {f(th['QT'])} | | {f(th['FR'])} | {f(th['FT'])} |")
        A("")
    with_wt = [r for r in main_runs if r.get("wait_table")]
    A(f"Таблица: Распределение времени ожидания при N = {ns[1]}, % заявок")
    A("")
    A("| Интервал | " + " | ".join(f"{LAWS[r['law']]['title'][:5]}., ρ = {r['rho']}" for r in with_wt) + " |")
    A("|---|" + "---:|" * len(with_wt))
    for i, t in enumerate(with_wt[0]["wait_table"]):
        A(f"| {t['label']} | " + " | ".join(f"{r['wait_table'][i]['pct']:.1f}" for r in with_wt) + " |")
    A("")
    A(f"Таблица: Результаты 10 экспериментов, показательные законы, ρ = {rhos[1]}, N = {ns[1]}")
    A("")
    A("| № | QT | QA | FR | FT |")
    A("|---:|---:|---:|---:|---:|")
    for i, r in enumerate(reps, start=1):
        A(f"| {i} | {f(r['QT'], 2)} | {f(r['QA'])} | {f(r['FR'])} | {f(r['FT'], 2)} |")
    A("")
    A("## Обработка результатов эксперимента")
    A("")
    A("### Сравнение с теорией")
    A("")
    A(f"Таблица: Относительное отклонение результатов моделирования от теории при N = {ns[1]}")
    A("")
    A("| Закон | ρ | QA | QT | FR | FT |")
    A("|---|---:|---:|---:|---:|---:|")
    for r in with_wt:
        th = r["theory"]
        d = lambda k: f"{(r[k] - th[k]) / th[k]:+.1%}"
        A(f"| {LAWS[r['law']]['title']} | {r['rho']} | {d('QA')} | {d('QT')} | {d('FR')} | {d('FT')} |")
    A("")
    for r in with_wt:
        A(f"![Распределение времени ожидания ожидавших заявок, обслуживание: {LAWS[r['law']]['title']}, ρ = {r['rho']}, "
          f"N = {r['N']}]({rel}/wait_{r['law']}_{r['rho']}.png)")
        A("")
    A("### Переходный процесс")
    A("")
    A("Для оценки переходного процесса проведены отдельные прогоны при увеличении N.")
    A("")
    keys = list(transient)
    A("Таблица: Среднее время ожидания QT при разном числе заявок N")
    A("")
    A("| N | " + " | ".join(f"{LAWS[transient[k]['law']]['title'][:5]}., ρ = {transient[k]['rho']}"
                           for k in keys) + " |")
    A("|---:|" + "---:|" * len(keys))
    for i in range(len(transient[keys[0]]["rows"])):
        A(f"| {transient[keys[0]]['rows'][i]['N']} | " +
          " | ".join(f(transient[k]["rows"][i]["QT"], 2) for k in keys) + " |")
    A("| теория | " + " | ".join(f(transient[k]["theory"]["QT"], 2) for k in keys) + " |")
    A("")
    for k in keys:
        A(f"![Зависимость QT от N, обслуживание: {LAWS[transient[k]['law']]['title']}, "
          f"ρ = {transient[k]['rho']}]({rel}/transient_{k}.png)")
        A("")
    A("<!-- Впишите оценку длительности переходного процесса по таблице выше. -->")
    A("")
    A("### Доверительные интервалы")
    A("")
    A(f"Доверительный интервал на уровне 0,95: x̄ ± t·s/√n, t(0,975; {len(reps) - 1}) = {tcrit:.3f} "
      "(распределение Стьюдента, так как n = 10 мало и σ неизвестна).")
    A("")
    A("Таблица: Доверительные интервалы по 10 экспериментам")
    A("")
    A("| Характеристика | x̄ | s | 95% ДИ | теория | теория в ДИ |")
    A("|---|---:|---:|---|---:|:-:|")
    for key, c in ci.items():
        A(f"| {key} | {f(c['mean'])} | {f(c['std'])} | [{f(c['low'])}; {f(c['high'])}] | {f(c['theory'])} | "
          f"{'да' if c['inside'] else 'нет'} |")
    A("")
    A("## Выводы")
    A("")
    A("<!-- Проверьте формулировки по своим числам. -->")
    A("1. FT и FR совпадают с теоретическими t_об и ρ уже при малых N для всех законов обслуживания.")
    A("2. QA и QT растут с увеличением коэффициента вариации времени обслуживания: равномерный (ν² = 1/3) "
      "< треугольный (ν² = 1/2) < показательный (ν² = 1), как и предсказывает формула Поллачека — Хинчина.")
    A("3. При большом ρ очередь и время ожидания резко растут (множитель 1/(1 − ρ)), а оценки QT при малом N "
      "заметно отличаются от теории: переходный процесс длинный, дисперсия оценки большая.")
    A("4. Попадание теоретических значений в доверительные интервалы проверено в таблице выше.")
    A("")
    A("## Приложение А. Исходный код программы")
    A("")
    A(f"Запуск: `python3 pr2.py --variant {args.variant} --mean-arrival {args.mean_arrival:g} --seed {args.seed}`.")
    A("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
