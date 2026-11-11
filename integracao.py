
# PROJETO CIDA
# Teste de leitura das planilhas

import pandas as pd

# Arquivos Excel
arquivo_grupo = "Planilha_Python_Cida_2_corrigida.xlsx"
arquivo_professor = "experimento.xlsx"

# Leitura da planilha do grupo
grupo = pd.read_excel(
    arquivo_grupo,
    sheet_name="Página1"
)

# Leitura dos dados experimentais do professor
professor = pd.read_excel(
    arquivo_professor,
    sheet_name="tabela1",
    header=1,
    usecols="B:K"
)

# Apresentação dos resultados
print("===== DADOS DO GRUPO =====")
print(grupo.head())

print("\n===== DADOS DO PROFESSOR =====")
print(professor.head())

# Quantidade de registros
print("\n===== QUANTIDADE DE AMOSTRAS =====")
print("Grupo:", len(grupo))
print("Professor:", len(professor))

print("\nLeitura das planilhas concluída!")


# Verificação dos identificadores das amostras

grupo_ensaios = pd.read_excel(
    arquivo_grupo,
    sheet_name="Página2"
)

professor_ensaios = pd.read_excel(
    arquivo_professor,
    sheet_name="tabela2",
    header=1,
    usecols="B:F"
)

# ==========================================
# CORREÇÃO E CONFERÊNCIA DOS IDs DOS ENSAIOS
# ==========================================

# Cada mola possui cinco medições.
# O ID aparece na terceira linha de cada bloco.

for inicio in range(0, len(grupo_ensaios), 5):

    bloco = grupo_ensaios.iloc[inicio:inicio + 5]

    identificadores = bloco["ID"].dropna().unique()

    if len(identificadores) != 1:
        print("ERRO: identificação incorreta no bloco", inicio)
        continue

    grupo_ensaios.loc[bloco.index, "ID"] = identificadores[0]


# ==========================================
# CONFERÊNCIA DA DISTRIBUIÇÃO
# ==========================================

print("\n===== DISTRIBUIÇÃO DOS ENSAIOS =====")

print("\nGrupo:")
print(grupo_ensaios["ID"].value_counts().sort_index())

print("\nProfessor:")
print(professor_ensaios["ID"].value_counts().sort_index())

print("\n===== COLUNAS DOS ENSAIOS =====")

print("\nEnsaios do grupo:")
print(grupo_ensaios.columns.tolist())
print(grupo_ensaios.head())

print("\nEnsaios do professor:")
print(professor_ensaios.columns.tolist())
print(professor_ensaios.head())

# ==========================================
# PADRONIZAÇÃO DOS ENSAIOS DO GRUPO
# ==========================================

grupo_padronizado = grupo_ensaios.rename(columns={
    "Carga Aplicada (kgf)": "carga",
    "Alongamento (mm)": "alongamento",
    "Comprimento (mm)": "comprimento"
})

# Converter os valores para formato numérico
grupo_padronizado["carga"] = pd.to_numeric(
    grupo_padronizado["carga"], errors="coerce"
)

grupo_padronizado["alongamento"] = pd.to_numeric(
    grupo_padronizado["alongamento"], errors="coerce"
)

grupo_padronizado["comprimento"] = pd.to_numeric(
    grupo_padronizado["comprimento"], errors="coerce"
)

# Calcular rigidez e ductilidade
grupo_padronizado["rigidez"] = (
    grupo_padronizado["carga"] /
    grupo_padronizado["alongamento"]
)

grupo_padronizado["dutilidade"] = (
    grupo_padronizado["alongamento"] /
    grupo_padronizado["comprimento"]
)

grupo_padronizado["origem"] = "Grupo"


# ==========================================
# PADRONIZAÇÃO DOS ENSAIOS DO PROFESSOR
# ==========================================

professor_padronizado = professor_ensaios.copy()

professor_padronizado["origem"] = "Professor"


# ==========================================
# INTEGRAÇÃO DAS DUAS BASES
# ==========================================

colunas = [
    "ID",
    "carga",
    "alongamento",
    "rigidez",
    "dutilidade",
    "origem"
]

base_integrada = pd.concat(
    [
        grupo_padronizado[colunas],
        professor_padronizado[colunas]
    ],
    ignore_index=True
)


# ==========================================
# CONFERÊNCIA E EXPORTAÇÃO
# ==========================================

print("\n===== BASE INTEGRADA =====")
print(base_integrada.head(10))

print("\nTotal de ensaios:", len(base_integrada))

print("\nEnsaios por origem:")
print(base_integrada["origem"].value_counts())

print("\nEnsaios por mola:")
print(base_integrada.groupby(
    ["origem", "ID"]
).size())

base_integrada.to_excel(
    "base_integrada.xlsx",
    index=False
)

print("\nArquivo base_integrada.xlsx criado com sucesso!")

# ==========================================
# VERIFICACAO DA QUALIDADE DOS DADOS
# ==========================================

print("\n===== QUALIDADE DA BASE INTEGRADA =====")

print("\nValores ausentes:")
print(base_integrada.isnull().sum())

print("\nValores iguais a zero:")
print((base_integrada == 0).sum())

print("\nEstatísticas descritivas:")
print(base_integrada[
    ["carga", "alongamento", "rigidez", "dutilidade"]
].describe())

print("\nConferência concluída!")

# ==========================================
# ESTATISTICAS AGRUPADAS POR MOLA
# ==========================================

resumo_molas = base_integrada.groupby(
    ["origem", "ID"]
).agg(
    media_carga=("carga", "mean"),
    media_alongamento=("alongamento", "mean"),
    media_rigidez=("rigidez", "mean"),
    media_dutilidade=("dutilidade", "mean"),
    quantidade_ensaios=("ID", "count")
).reset_index()

print("\n===== RESUMO POR MOLA =====")
print(resumo_molas.to_string(index=False))


# ==========================================
# ESTATISTICAS POR ORIGEM
# ==========================================

resumo_origem = base_integrada.groupby("origem")[
    ["carga", "alongamento", "rigidez", "dutilidade"]
].agg(["mean", "std", "min", "max"])

print("\n===== RESUMO POR ORIGEM =====")
print(resumo_origem)


# ==========================================
# EXPORTAR RESUMOS PARA EXCEL
# ==========================================

with pd.ExcelWriter("resultados_estatisticos.xlsx") as arquivo:

    base_integrada.to_excel(
        arquivo,
        sheet_name="Base integrada",
        index=False
    )

    resumo_molas.to_excel(
        arquivo,
        sheet_name="Resumo por mola",
        index=False
    )

    resumo_origem.to_excel(
        arquivo,
        sheet_name="Resumo por origem"
    )


# ==========================================
# GERAR GRAFICOS
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("graficos", exist_ok=True)

sns.set_theme(style="whitegrid")


# Gráfico 1: distribuição da rigidez
plt.figure(figsize=(9, 5))

sns.histplot(
    data=base_integrada,
    x="rigidez",
    hue="origem",
    kde=True,
    element="step"
)

plt.title("Distribuição da rigidez das molas")
plt.xlabel("Rigidez (kgf/mm)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig("graficos/distribuicao_rigidez.png", dpi=300)
plt.close()


# Gráfico 2: comparação da rigidez por origem
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=base_integrada,
    x="origem",
    y="rigidez"
)

plt.title("Comparação da rigidez por origem")
plt.xlabel("Origem dos dados")
plt.ylabel("Rigidez (kgf/mm)")
plt.tight_layout()
plt.savefig("graficos/comparacao_rigidez.png", dpi=300)
plt.close()


# Gráfico 3: relação entre carga e alongamento
plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=base_integrada,
    x="alongamento",
    y="carga",
    hue="origem",
    s=70
)

plt.title("Relação entre carga e alongamento")
plt.xlabel("Alongamento (mm)")
plt.ylabel("Carga (kgf)")
plt.tight_layout()
plt.savefig("graficos/carga_alongamento.png", dpi=300)
plt.close()


print("\n===== FINALIZADO =====")
print("Planilha: resultados_estatisticos.xlsx")
print("Gráficos salvos na pasta: graficos")

print("\n===== CONFERÊNCIA DA MOLA ID 03 =====")

print(
    base_integrada[
        (base_integrada["origem"] == "Grupo") &
        (base_integrada["ID"] == 3)
    ][["ID", "carga", "alongamento", "rigidez", "dutilidade"]]
)

# ==========================================
# IDENTIFICACAO DE POSSIVEIS OUTLIERS
# Metodo: Intervalo Interquartil (IQR)
# ==========================================

variaveis = [
    "carga",
    "alongamento",
    "rigidez",
    "dutilidade"
]

print("\n===== ANALISE DE VALORES ATIPICOS =====")

for origem, dados in base_integrada.groupby("origem"):

    print(f"\nORIGEM: {origem}")

    for variavel in variaveis:

        q1 = dados[variavel].quantile(0.25)
        q3 = dados[variavel].quantile(0.75)

        iqr = q3 - q1

        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        outliers = dados[
            (dados[variavel] < limite_inferior) |
            (dados[variavel] > limite_superior)
        ]

        print(f"\nVariável: {variavel}")
        print("Limite inferior:", round(limite_inferior, 4))
        print("Limite superior:", round(limite_superior, 4))
        print("Quantidade de possíveis outliers:", len(outliers))

        if not outliers.empty:
            print(outliers[["ID", variavel]])

# ==========================================
# MATRIZES DE CORRELACAO
# ==========================================

print("\n===== CORRELACAO DE PEARSON =====")

for origem, dados in base_integrada.groupby("origem"):

    print(f"\nORIGEM: {origem}")

    correlacao = dados[
        ["carga", "alongamento", "rigidez", "dutilidade"]
    ].corr(method="pearson")

    print(correlacao.round(3))

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlacao,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1
    )

    plt.title(f"Matriz de correlação - {origem}")
    plt.tight_layout()

    nome = origem.lower()
    plt.savefig(
        f"graficos/correlacao_{nome}.png",
        dpi=300
    )

    plt.close()

print("\nMatrizes de correlação geradas!")