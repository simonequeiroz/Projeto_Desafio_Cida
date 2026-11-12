# ============================================================
# Projeto CIDA P1 - Levantamento experimental de molas
# zigue-zague (nosag) fabricadas por impressão 3D em PLA
#
# Fluxo do programa (etapas do enunciado):
#   1. Leitura das planilhas (Excel) -> DataFrame -> dicionário
#   2. Rigidez F/δ de cada medição
#   3. Identificação e eliminação de pontos aberrantes (outliers)
#   4. K e D de cada mola, com os dados já limpos
#   5. Estatística descritiva
#   6. Relações entre a geometria e a rigidez (correlações e modelo)
#   7. Gráficos: histogramas, boxplots e dispersões
#   8. Projeto de uma nova mola para imprimir (usa o modelo)
#
# Definições:
#   Rigidez K     = média de F/δ das medições válidas da mola (kgf/mm)
#   Ductilidade D = δ/C na 5ª carga (a mais alta), em %
#
# Para instalar as bibliotecas: pip install -r requirements.txt
# ============================================================

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats

PASTA = Path(__file__).resolve().parent

# Arquivos de entrada
ARQ_PROFESSOR = PASTA / "experimento.xlsx"
ARQ_GRUPO = PASTA / "Planilha_Python_CIDA_2_corrigida.xlsx"

# Arquivos de saída
ARQ_RESULTADOS = PASTA / "resultados_CIDA_P1.xlsx"
PASTA_GRAFICOS = PASTA / "graficos"

# Nível de significância dos testes estatísticos
ALFA = 0.05

# Rigidez desejada para a mola que será impressa (kgf/mm).
# Troque este valor pelo que o grupo escolher.
K_ALVO = 0.05

# Variáveis geométricas medidas em cada mola
GEOMETRIA = ["C", "h", "t", "L", "n", "Ltot", "m"]
NOMES = {
    "C": "Comprimento C (mm)",
    "h": "Espessura do fio h (mm)",
    "t": "Altura t (mm)",
    "L": "Passo L (mm)",
    "n": "Número de elos n",
    "Ltot": "Largura total Ltot (mm)",
    "m": "Massa m (g)",
    "K": "Rigidez K (kgf/mm)",
    "D": "Ductilidade D (%)",
}

# Observações de laboratório sobre medições específicas
OBSERVACOES = {
    (58, 5): "ruptura relatada na 5ª carga",
}


# ------------------------------------------------------------
# 1. Leitura das planilhas
# ------------------------------------------------------------
def ler_planilhas():
    """Lê as planilhas do grupo e do professor e devolve duas tabelas:
    molas (uma linha por mola) e medicoes (uma linha por carga)."""
    # Professor: tabela1 = características; tabela2 = ensaios
    molas_prof = pd.read_excel(ARQ_PROFESSOR, sheet_name="tabela1", header=1, usecols="B:K")
    med_prof = pd.read_excel(ARQ_PROFESSOR, sheet_name="tabela2", header=1, usecols="B:D")

    # Grupo: Página1 = características; Página2 = ensaios
    molas_grupo = pd.read_excel(ARQ_GRUPO, sheet_name="Página1")
    med_grupo = pd.read_excel(ARQ_GRUPO, sheet_name="Página2")

    # Na Página2 o ID só aparece na 3ª linha de cada bloco de 5 cargas
    for inicio in range(0, len(med_grupo), 5):
        bloco = med_grupo.iloc[inicio:inicio + 5]
        ids = bloco["ID"].dropna().unique()
        if len(ids) != 1 or len(bloco) != 5:
            raise ValueError(f"Bloco inválido na linha {inicio + 2} da Página2: IDs {ids}")
        med_grupo.loc[bloco.index, "ID"] = ids[0]

    med_grupo = med_grupo.rename(columns={
        "Carga Aplicada (kgf)": "carga",
        "Alongamento (mm)": "alongamento",
    })

    molas_grupo["origem"] = "Grupo"
    molas_prof["origem"] = "Professor"
    med_grupo["origem"] = "Grupo"
    med_prof["origem"] = "Professor"

    molas = pd.concat([molas_grupo[["ID", "origem"] + GEOMETRIA],
                       molas_prof[["ID", "origem"] + GEOMETRIA]], ignore_index=True)
    medicoes = pd.concat([med_grupo[["ID", "origem", "carga", "alongamento"]],
                          med_prof[["ID", "origem", "carga", "alongamento"]]], ignore_index=True)

    for tabela in (molas, medicoes):
        tabela["ID"] = pd.to_numeric(tabela["ID"]).astype(int)
    for coluna in GEOMETRIA:
        molas[coluna] = pd.to_numeric(molas[coluna], errors="coerce")
    for coluna in ["carga", "alongamento"]:
        medicoes[coluna] = pd.to_numeric(medicoes[coluna], errors="coerce")

    if molas[GEOMETRIA].isna().any().any() or medicoes[["carga", "alongamento"]].isna().any().any():
        raise ValueError("Há células vazias ou não numéricas nas planilhas.")

    # Número da carga (1 a 5) dentro de cada mola, da menor para a maior
    medicoes = medicoes.sort_values(["origem", "ID", "carga"], kind="stable").reset_index(drop=True)
    medicoes["carga_n"] = medicoes.groupby(["origem", "ID"]).cumcount() + 1

    contagem = medicoes.groupby(["origem", "ID"]).size()
    if (contagem != 5).any():
        raise ValueError(f"Molas sem 5 cargas:\n{contagem[contagem != 5]}")

    return molas, medicoes


# ------------------------------------------------------------
# 3. Outliers: teste de Grubbs dentro de cada mola
# ------------------------------------------------------------
def grubbs_critico(n, alfa=ALFA):
    """Valor crítico do teste de Grubbs bilateral para n valores."""
    t = stats.t.ppf(1 - alfa / (2 * n), n - 2)
    return (n - 1) / math.sqrt(n) * math.sqrt(t ** 2 / (n - 2 + t ** 2))


def marcar_outliers(medicoes):
    """Aplica o teste de Grubbs às 5 razões F/δ de cada mola.
    O valor mais afastado da média é aberrante quando G > G crítico.
    Compara cada medição só com as da mesma mola, porque molas
    diferentes têm rigidez diferente por causa da geometria."""
    medicoes["G"] = np.nan
    medicoes["outlier"] = False
    g_crit = grubbs_critico(5)

    for _, dados in medicoes.groupby(["origem", "ID"]):
        x = dados["rigidez"].to_numpy()
        j = int(np.argmax(np.abs(x - x.mean())))
        g = abs(x[j] - x.mean()) / x.std(ddof=1)
        indice = dados.index[j]
        medicoes.loc[indice, "G"] = g
        medicoes.loc[indice, "outlier"] = g > g_crit

    medicoes["observacao"] = [
        OBSERVACOES.get((i, c), "") for i, c in zip(medicoes["ID"], medicoes["carga_n"])
    ]
    return g_crit


# ------------------------------------------------------------
# 6. Modelo da rigidez em função da geometria
# ------------------------------------------------------------
VARS_MODELO = ["h", "t", "L", "n"]


def matriz_modelo(tabela):
    """Colunas do modelo de potência em escala log: 1, ln h, ln t, ln L, ln n."""
    return np.column_stack([np.ones(len(tabela))] + [np.log(tabela[v]) for v in VARS_MODELO])


def ajustar_modelo(molas):
    """Ajusta K = A · h^a · t^b · L^c · n^d por mínimos quadrados em escala log
    e valida tirando uma mola de cada vez (validação cruzada leave-one-out)."""
    X = matriz_modelo(molas)
    y = np.log(molas["K"].to_numpy())
    n_obs, n_par = X.shape

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuos = y - X @ coef
    gl = n_obs - n_par
    s2 = (residuos ** 2).sum() / gl
    cov = s2 * np.linalg.inv(X.T @ X)
    erro_padrao = np.sqrt(np.diag(cov))
    t_crit = stats.t.ppf(1 - ALFA / 2, gl)
    r2 = 1 - (residuos ** 2).sum() / ((y - y.mean()) ** 2).sum()

    tabela_coef = pd.DataFrame({
        "termo": ["ln A"] + [f"expoente de {v}" for v in VARS_MODELO],
        "coeficiente": coef,
        "erro_padrao": erro_padrao,
        "IC95_inf": coef - t_crit * erro_padrao,
        "IC95_sup": coef + t_crit * erro_padrao,
        "p_valor": 2 * stats.t.sf(np.abs(coef / erro_padrao), gl),
    })

    # Validação: prevê cada mola com o modelo ajustado nas outras 16
    previsto_loo = []
    for i in range(n_obs):
        resto = np.arange(n_obs) != i
        c, *_ = np.linalg.lstsq(X[resto], y[resto], rcond=None)
        previsto_loo.append(math.exp(X[i] @ c))

    validacao = molas[["ID", "origem"] + VARS_MODELO + ["K"]].copy()
    validacao["K_previsto"] = np.exp(X @ coef)
    validacao["erro_pct"] = (validacao["K_previsto"] / validacao["K"] - 1) * 100
    validacao["K_previsto_LOO"] = previsto_loo
    validacao["erro_LOO_pct"] = (validacao["K_previsto_LOO"] / validacao["K"] - 1) * 100

    return {
        "coef": coef, "cov": cov, "s2": s2, "gl": gl, "t_crit": t_crit, "r2": r2,
        "XtX_inv": np.linalg.inv(X.T @ X), "tabela": tabela_coef, "validacao": validacao,
        "faixa": {v: (molas[v].min(), molas[v].max()) for v in VARS_MODELO},
    }


def prever_k(modelo, h, t, L, n):
    """Rigidez prevista (kgf/mm) e intervalo de predição de 95% para uma mola nova."""
    x0 = np.array([1, math.log(h), math.log(t), math.log(L), math.log(n)])
    ln_k = x0 @ modelo["coef"]
    se = math.sqrt(modelo["s2"] * (1 + x0 @ modelo["XtX_inv"] @ x0))
    margem = modelo["t_crit"] * se
    return math.exp(ln_k), math.exp(ln_k - margem), math.exp(ln_k + margem)


def projetar_mola(modelo, k_alvo):
    """Procura combinações de h, t e n e calcula o passo L que dá a rigidez k_alvo.
    Só aceita geometrias dentro da faixa ensaiada (o modelo não vale fora dela)."""
    a0, ah, at, aL, an = modelo["coef"]
    faixa = modelo["faixa"]
    opcoes = []
    for h in np.arange(1.5, 3.01, 0.5):
        for t in np.arange(4.0, 10.01, 2.0):
            for n in (3, 4, 5):
                ln_L = (math.log(k_alvo) - a0 - ah * math.log(h) - at * math.log(t) - an * math.log(n)) / aL
                L = math.exp(ln_L)
                dentro = all(faixa[v][0] <= valor <= faixa[v][1]
                             for v, valor in zip(VARS_MODELO, (h, t, L, n)))
                if dentro:
                    k, k_inf, k_sup = prever_k(modelo, h, t, L, n)
                    opcoes.append({"h": h, "t": t, "L": round(L, 1), "n": n, "K_previsto": k,
                                   "PI95_inf": k_inf, "PI95_sup": k_sup,
                                   "incerteza_pct": (k_sup - k_inf) / 2 / k * 100})
    # As primeiras opções são as de previsão mais confiável (geometria mais
    # parecida com as molas ensaiadas, logo intervalo de predição mais estreito)
    if not opcoes:
        return pd.DataFrame()
    return pd.DataFrame(opcoes).sort_values("incerteza_pct").reset_index(drop=True)


# ------------------------------------------------------------
# 7. Gráficos
# ------------------------------------------------------------
CORES = {"Grupo": "#2a78d6", "Professor": "#eb6834"}
MARCADORES = {"Grupo": "o", "Professor": "s"}
CINZA = "#52514e"
VERMELHO = "#c0392b"

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titlelocation": "left", "savefig.dpi": 150, "savefig.bbox": "tight",
})
virgula = FuncFormatter(lambda valor, _: f"{valor:g}".replace(".", ","))


def fmt(valor, casas=2):
    """Número com vírgula decimal, para títulos e textos dos gráficos."""
    return f"{valor:.{casas}f}".replace(".", ",")


def fmt_p(p):
    """p-valor com vírgula; valores muito pequenos aparecem como < 0,001."""
    return "< 0,001" if p < 0.001 else "= " + fmt(p, 3)


def legenda_origens(eixo, **kwargs):
    for origem in CORES:
        eixo.scatter([], [], color=CORES[origem], marker=MARCADORES[origem], label=origem)
    eixo.legend(frameon=False, **kwargs)


def gerar_graficos(molas, medicoes, modelo, correlacoes):
    PASTA_GRAFICOS.mkdir(exist_ok=True)
    ordem = molas.sort_values("K", ascending=False)
    aberrantes = medicoes[medicoes["outlier"]]

    # 01 - Dispersão carga × alongamento, com os pontos eliminados
    fig, eixo = plt.subplots(figsize=(9, 5.5))
    for _, mola in molas.iterrows():
        dados = medicoes[(medicoes["ID"] == mola["ID"]) & (medicoes["origem"] == mola["origem"])]
        validos = dados[~dados["outlier"]]
        cor = CORES[mola["origem"]]
        eixo.plot(validos["alongamento"], validos["carga"], color=cor, alpha=0.35, linewidth=1)
        eixo.scatter(validos["alongamento"], validos["carga"], color=cor, s=18,
                     marker=MARCADORES[mola["origem"]], zorder=3)
    eixo.scatter(aberrantes["alongamento"], aberrantes["carga"], marker="x", s=80, color=VERMELHO,
                 linewidth=2, zorder=4, label="Ponto aberrante eliminado")
    for _, p in aberrantes.iterrows():
        eixo.annotate(f"ID {p['ID']}", (p["alongamento"], p["carga"]), xytext=(6, 4),
                      textcoords="offset points", fontsize=8, color=VERMELHO)
    legenda_origens(eixo, loc="upper right")
    eixo.set_title("Carga × alongamento das 17 molas (85 medições)")
    eixo.set_xlabel("Alongamento δ (mm)")
    eixo.set_ylabel("Carga F (kgf)")
    eixo.xaxis.set_major_formatter(virgula)
    eixo.yaxis.set_major_formatter(virgula)
    fig.savefig(PASTA_GRAFICOS / "01_dispersao_carga_alongamento.png")
    plt.close(fig)

    # 02 - Boxplot de F/δ por mola, com os pontos eliminados
    fig, eixo = plt.subplots(figsize=(11, 4.8))
    caixas = []
    for _, mola in ordem.iterrows():
        dados = medicoes[(medicoes["ID"] == mola["ID"]) & (medicoes["origem"] == mola["origem"])]
        caixas.append(dados["rigidez"])
    bp = eixo.boxplot(caixas, widths=0.55, patch_artist=True, showfliers=False,
                      medianprops={"color": "black"})
    for caixa, origem in zip(bp["boxes"], ordem["origem"]):
        caixa.set_facecolor(CORES[origem])
        caixa.set_alpha(0.6)
    for pos, (_, mola) in enumerate(ordem.iterrows(), start=1):
        dados = medicoes[(medicoes["ID"] == mola["ID"]) & (medicoes["origem"] == mola["origem"])]
        eixo.scatter([pos] * (~dados["outlier"]).sum(), dados.loc[~dados["outlier"], "rigidez"],
                     s=12, color="black", zorder=3)
        eixo.scatter([pos] * dados["outlier"].sum(), dados.loc[dados["outlier"], "rigidez"],
                     marker="x", s=70, color=VERMELHO, linewidth=2, zorder=4)
    eixo.scatter([], [], marker="x", color=VERMELHO, label="Eliminado (Grubbs, α = 0,05)")
    eixo.set_yscale("log")
    eixo.yaxis.set_major_formatter(virgula)
    eixo.set_xticks(range(1, len(ordem) + 1), [str(i) for i in ordem["ID"]])
    for rotulo, origem in zip(eixo.get_xticklabels(), ordem["origem"]):
        rotulo.set_color(CORES[origem])
        rotulo.set_fontweight("bold")
    eixo.grid(axis="x", visible=False)
    eixo.set_title("Rigidez F/δ das 5 cargas de cada mola")
    eixo.set_xlabel("ID da mola (da mais rígida para a menos rígida; cor = origem)")
    eixo.set_ylabel("F/δ (kgf/mm, escala log)")
    eixo.legend(frameon=False, loc="upper right")
    fig.savefig(PASTA_GRAFICOS / "02_boxplot_rigidez_por_mola.png")
    plt.close(fig)

    # 03 - Histogramas de K e D
    fig, paineis = plt.subplots(1, 2, figsize=(10, 4.2))
    bins_k = np.logspace(np.log10(molas["K"].min() * 0.9), np.log10(molas["K"].max() * 1.1), 9)
    paineis[0].hist([molas.loc[molas["origem"] == o, "K"] for o in CORES], bins=bins_k, stacked=True,
                    color=list(CORES.values()), edgecolor="white", label=list(CORES))
    paineis[0].set_xscale("log")
    paineis[0].xaxis.set_major_formatter(virgula)
    paineis[0].set_xlabel("Rigidez K (kgf/mm, escala log)")
    paineis[0].set_title("Histograma da rigidez K")
    paineis[1].hist([molas.loc[molas["origem"] == o, "D"] for o in CORES], bins=8, stacked=True,
                    color=list(CORES.values()), edgecolor="white", label=list(CORES))
    paineis[1].set_xlabel("Ductilidade D (%)")
    paineis[1].set_title("Histograma da ductilidade D")
    for eixo in paineis:
        eixo.set_ylabel("Número de molas")
        eixo.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PASTA_GRAFICOS / "03_histogramas_K_D.png")
    plt.close(fig)

    # 04 - Boxplots de K e D por origem
    fig, paineis = plt.subplots(1, 2, figsize=(9, 4.2))
    for eixo, var in zip(paineis, ["K", "D"]):
        grupos = [molas.loc[molas["origem"] == o, var] for o in CORES]
        eixo.boxplot(grupos, widths=0.45, showfliers=False, medianprops={"color": "black"})
        for pos, (origem, valores) in enumerate(zip(CORES, grupos), start=1):
            jitter = np.linspace(-0.1, 0.1, len(valores))
            eixo.scatter(pos + jitter, valores, color=CORES[origem], marker=MARCADORES[origem], zorder=3, s=25)
        _, p = stats.mannwhitneyu(*grupos)
        eixo.set_xticks([1, 2], list(CORES))
        eixo.set_ylabel(NOMES[var])
        eixo.yaxis.set_major_formatter(virgula)
        eixo.set_title(f"{var} por origem (Mann-Whitney p {fmt_p(p)})")
        if var == "K":
            eixo.set_yscale("log")
    fig.tight_layout()
    fig.savefig(PASTA_GRAFICOS / "04_boxplot_K_D_por_origem.png")
    plt.close(fig)

    # 05 - Dispersões da rigidez com cada variável geométrica
    variaveis = ["h", "t", "L", "n", "Ltot", "m"]
    fig, paineis = plt.subplots(2, 3, figsize=(12, 7))
    for eixo, var in zip(paineis.flat, variaveis):
        for origem in CORES:
            dados = molas[molas["origem"] == origem]
            eixo.scatter(dados[var], dados["K"], color=CORES[origem], marker=MARCADORES[origem],
                         s=35, edgecolor="white")
        linha = correlacoes[(correlacoes["alvo"] == "K") & (correlacoes["variavel"] == var)].iloc[0]
        destaque = "black" if linha["spearman_p"] < ALFA else CINZA
        eixo.set_title(f"ρ = {fmt(linha['spearman_rho'])}  (p {fmt_p(linha['spearman_p'])})",
                       color=destaque, fontsize=10)
        eixo.set_xlabel(NOMES[var])
        eixo.set_yscale("log")
        eixo.yaxis.set_major_formatter(virgula)
        eixo.xaxis.set_major_formatter(virgula)
    for eixo in paineis[:, 0]:
        eixo.set_ylabel("Rigidez K (kgf/mm, log)")
    legenda_origens(paineis[0, 0], loc="upper left")
    fig.suptitle("Rigidez × geometria (correlação de Spearman; título preto = significativa a 5%)",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(PASTA_GRAFICOS / "05_dispersao_K_geometria.png")
    plt.close(fig)

    # 06 - Matriz de correlação de Spearman
    variaveis = GEOMETRIA + ["K", "D"]
    matriz = molas[variaveis].corr(method="spearman")
    fig, eixo = plt.subplots(figsize=(7.5, 6.3))
    imagem = eixo.imshow(matriz, cmap="RdBu_r", vmin=-1, vmax=1)
    eixo.set_xticks(range(len(variaveis)), variaveis)
    eixo.set_yticks(range(len(variaveis)), variaveis)
    eixo.grid(False)
    for i in range(len(variaveis)):
        for j in range(len(variaveis)):
            valor = matriz.iloc[i, j]
            eixo.text(j, i, fmt(valor), ha="center", va="center", fontsize=8,
                      color="white" if abs(valor) > 0.6 else "black")
    fig.colorbar(imagem, shrink=0.8, format=virgula)
    eixo.set_title("Matriz de correlação de Spearman (17 molas)")
    fig.savefig(PASTA_GRAFICOS / "06_matriz_correlacao.png")
    plt.close(fig)

    # 07 - Modelo: K previsto × K medido
    val = modelo["validacao"]
    fig, eixo = plt.subplots(figsize=(6.5, 6))
    limites = [val["K"].min() * 0.7, val["K"].max() * 1.4]
    eixo.plot(limites, limites, color="black", linewidth=1, label="Previsto = medido")
    eixo.fill_between(limites, [v * 0.9 for v in limites], [v * 1.1 for v in limites],
                      color="#bbbbbb", alpha=0.35, label="±10%")
    for origem in CORES:
        dados = val[val["origem"] == origem]
        eixo.scatter(dados["K"], dados["K_previsto_LOO"], color=CORES[origem], marker=MARCADORES[origem],
                     s=40, edgecolor="white", label=origem, zorder=3)
    for _, mola in val.iterrows():
        eixo.annotate(str(mola["ID"]), (mola["K"], mola["K_previsto_LOO"]), xytext=(4, -9),
                      textcoords="offset points", fontsize=7, color=CINZA)
    eixo.set_xscale("log")
    eixo.set_yscale("log")
    eixo.set_xlim(limites)
    eixo.set_ylim(limites)
    eixo.xaxis.set_major_formatter(virgula)
    eixo.yaxis.set_major_formatter(virgula)
    erro = val["erro_LOO_pct"].abs()
    eixo.set_title(f"Modelo de rigidez: erro médio {fmt(erro.mean(), 1)}% (validação cruzada)")
    eixo.set_xlabel("K medido (kgf/mm)")
    eixo.set_ylabel("K previsto sem a própria mola (kgf/mm)")
    eixo.legend(frameon=False, loc="upper left")
    fig.savefig(PASTA_GRAFICOS / "07_modelo_previsto_medido.png")
    plt.close(fig)


# ------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------
def main():
    # 1. Leitura -> DataFrame -> dicionário
    molas, medicoes = ler_planilhas()
    molas_dict = molas.to_dict("list")
    medicoes_dict = medicoes.to_dict("list")
    print(f"Lidas {len(molas_dict['ID'])} molas e {len(medicoes_dict['ID'])} medições.")

    # 2. Rigidez de cada medição
    medicoes["rigidez"] = medicoes["carga"] / medicoes["alongamento"]

    # 3. Pontos aberrantes
    g_crit = marcar_outliers(medicoes)
    aberrantes = medicoes[medicoes["outlier"]]
    print(f"\n=== PONTOS ABERRANTES (Grubbs dentro de cada mola, α = {ALFA}, G crítico = {g_crit:.3f}) ===")
    for _, p in aberrantes.iterrows():
        k_resto = medicoes[(medicoes["ID"] == p["ID"]) & (medicoes["origem"] == p["origem"])
                           & ~medicoes["outlier"]]["rigidez"].mean()
        print(f"  Mola {p['ID']:>3} ({p['origem']}), carga {p['carga_n']}: F/δ = {p['rigidez']:.4f} "
              f"({(p['rigidez'] / k_resto - 1) * 100:+.1f}% em relação às demais), G = {p['G']:.3f}"
              + (f" - {p['observacao']}" if p["observacao"] else ""))
    print(f"  Eliminados {len(aberrantes)} de {len(medicoes)} pontos.")

    # 4. K e D de cada mola com os dados limpos
    validos = medicoes[~medicoes["outlier"]]
    agregado = validos.groupby(["origem", "ID"])["rigidez"].agg(K="mean", K_dp="std", n_validas="count")
    agregado = agregado.reset_index()
    molas = molas.merge(agregado, on=["origem", "ID"])
    t_crit = stats.t.ppf(1 - ALFA / 2, molas["n_validas"] - 1)
    margem = t_crit * molas["K_dp"] / np.sqrt(molas["n_validas"])
    molas["K_IC95_inf"] = molas["K"] - margem
    molas["K_IC95_sup"] = molas["K"] + margem
    quinta = medicoes[medicoes["carga_n"] == 5][["origem", "ID", "carga", "alongamento"]]
    quinta = quinta.rename(columns={"carga": "F_5", "alongamento": "delta_5"})
    molas = molas.merge(quinta, on=["origem", "ID"])
    molas["D"] = molas["delta_5"] / molas["C"] * 100

    print("\n=== RIGIDEZ E DUCTILIDADE POR MOLA ===")
    print(molas[["ID", "origem", "K", "K_IC95_inf", "K_IC95_sup", "n_validas", "D"]]
          .sort_values("K", ascending=False).round(4).to_string(index=False))

    # 5. Estatística descritiva
    linhas = []
    grupos = {"Todas": molas, "Grupo": molas[molas["origem"] == "Grupo"],
              "Professor": molas[molas["origem"] == "Professor"]}
    for nome, dados in grupos.items():
        for var in ["K", "D"] + GEOMETRIA:
            x = dados[var]
            ic = stats.t.interval(1 - ALFA, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
            linhas.append({"grupo": nome, "variavel": var, "n": len(x), "media": x.mean(),
                           "desvio_padrao": x.std(), "CV_pct": x.std() / x.mean() * 100,
                           "IC95_inf": ic[0], "IC95_sup": ic[1], "minimo": x.min(),
                           "Q1": x.quantile(0.25), "mediana": x.median(), "Q3": x.quantile(0.75),
                           "maximo": x.max(), "shapiro_p": stats.shapiro(x).pvalue})
    descritiva = pd.DataFrame(linhas)
    print("\n=== ESTATÍSTICA DESCRITIVA (17 molas) ===")
    print(descritiva[(descritiva["grupo"] == "Todas") & descritiva["variavel"].isin(["K", "D"])]
          .round(4).to_string(index=False))
    _, p_mw = stats.mannwhitneyu(grupos["Grupo"]["K"], grupos["Professor"]["K"])
    print(f"  K do grupo × K do professor (Mann-Whitney): p = {p_mw:.3f}")

    # 6a. Correlações entre geometria, K e D
    linhas = []
    for alvo in ["K", "D"]:
        for var in GEOMETRIA:
            r, p = stats.pearsonr(np.log(molas[var]), np.log(molas[alvo]))
            rho, p_s = stats.spearmanr(molas[var], molas[alvo])
            linhas.append({"alvo": alvo, "variavel": var, "spearman_rho": rho, "spearman_p": p_s,
                           "pearson_log_r": r, "pearson_log_p": p})
    correlacoes = pd.DataFrame(linhas)
    print("\n=== CORRELAÇÕES COM A RIGIDEZ K ===")
    print(correlacoes[correlacoes["alvo"] == "K"].round(4).to_string(index=False))

    # 6b. Modelo da rigidez
    modelo = ajustar_modelo(molas)
    a = modelo["coef"]
    erro_loo = modelo["validacao"]["erro_LOO_pct"].abs()
    print("\n=== MODELO DA RIGIDEZ ===")
    print(f"  K = {math.exp(a[0]):.4f} · h^{a[1]:.2f} · t^{a[2]:.2f} · L^{a[3]:.2f} · n^{a[4]:.2f}"
          "   (K em kgf/mm; h, t, L em mm)")
    print(f"  R² (escala log) = {modelo['r2']:.4f}")
    print(f"  Validação cruzada: erro médio {erro_loo.mean():.1f}%, máximo {erro_loo.max():.1f}%")
    print(modelo["tabela"].round(4).to_string(index=False))

    # Outlier entre molas: mola que foge do modelo (resíduo de validação cruzada)
    residuo = np.log(modelo["validacao"]["K"] / modelo["validacao"]["K_previsto_LOO"])
    g = np.max(np.abs(residuo - residuo.mean())) / residuo.std(ddof=1)
    mola_g = modelo["validacao"]["ID"].iloc[int(np.argmax(np.abs(residuo - residuo.mean())))]
    print(f"  Grubbs nos resíduos das molas: maior G = {g:.3f} (mola {mola_g}); "
          f"crítico = {grubbs_critico(len(residuo)):.3f} -> "
          + ("mola aberrante" if g > grubbs_critico(len(residuo)) else "nenhuma mola aberrante"))

    # 7. Gráficos
    gerar_graficos(molas, medicoes, modelo, correlacoes)

    # 8. Projeto da mola para imprimir
    opcoes = projetar_mola(modelo, K_ALVO)
    print(f"\n=== GEOMETRIAS PARA K = {K_ALVO} kgf/mm (dentro da faixa ensaiada) ===")
    if opcoes.empty:
        print("  Nenhuma geometria dentro da faixa ensaiada atinge essa rigidez.")
    else:
        print(f"  {len(opcoes)} opções; as 8 de previsão mais confiável "
              "(todas estão na aba projeto_nova_mola):")
        print(opcoes.head(8).round(4).to_string(index=False))

    # Salva os resultados
    with pd.ExcelWriter(ARQ_RESULTADOS) as arquivo:
        molas.to_excel(arquivo, sheet_name="molas", index=False)
        medicoes.to_excel(arquivo, sheet_name="medicoes", index=False)
        aberrantes.to_excel(arquivo, sheet_name="pontos_eliminados", index=False)
        descritiva.to_excel(arquivo, sheet_name="estatistica_descritiva", index=False)
        correlacoes.to_excel(arquivo, sheet_name="correlacoes", index=False)
        modelo["tabela"].to_excel(arquivo, sheet_name="modelo", index=False)
        modelo["validacao"].to_excel(arquivo, sheet_name="validacao_modelo", index=False)
        opcoes.to_excel(arquivo, sheet_name="projeto_nova_mola", index=False)

    print(f"\nResultados salvos em {ARQ_RESULTADOS.name}; gráficos na pasta {PASTA_GRAFICOS.name}/")


if __name__ == "__main__":
    main()
