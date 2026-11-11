import math
import pandas as pd
import numpy as np

# Arquivos de entrada: mantenha estes dois arquivos na mesma pasta deste script.
ARQ_PROF = "experimento.xlsx"
ARQ_GRUPO = "Planilha_Python_CIDA_2_corrigida.xlsx"

# Arquivos de saída.
SAIDA_TABELA = "Tabela1_CIDA_atualizada.xlsx"
SAIDA_ENSAIOS = "Ensaios_integrados_CIDA.xlsx"


def main():
    # 1. Leitura das tabelas
    prof_char = pd.read_excel(
        ARQ_PROF, sheet_name="tabela1", header=1, usecols="B:K"
    )
    prof_ens = pd.read_excel(
        ARQ_PROF, sheet_name="tabela2", header=1, usecols="B:F"
    )
    grupo_char = pd.read_excel(
        ARQ_GRUPO, sheet_name="Página1"
    )
    grupo_ens = pd.read_excel(
        ARQ_GRUPO, sheet_name="Página2"
    )

    # 2. Completa o ID nas cinco linhas de cada mola da planilha do grupo.
    for inicio in range(0, len(grupo_ens), 5):
        bloco = grupo_ens.iloc[inicio:inicio + 5]
        ids = bloco["ID"].dropna().unique()

        if len(ids) != 1 or len(bloco) != 5:
            raise ValueError(
                f"Bloco inválido na linha {inicio + 1}: IDs encontrados {ids}"
            )

        grupo_ens.loc[bloco.index, "ID"] = ids[0]

    # 3. Padroniza os nomes das colunas dos ensaios do grupo.
    grupo_ens = grupo_ens.rename(columns={
        "Carga Aplicada (kgf)": "carga",
        "Alongamento (mm)": "alongamento"
    })

    # Confere se as colunas necessárias existem antes de continuar.
    colunas_grupo = ["ID", "carga", "alongamento"]
    faltantes_grupo = [c for c in colunas_grupo if c not in grupo_ens.columns]
    if faltantes_grupo:
        raise KeyError(
            f"Colunas ausentes na Página2 da planilha do grupo: {faltantes_grupo}"
        )

    colunas_prof = ["ID", "carga", "alongamento", "rigidez"]
    faltantes_prof = [c for c in colunas_prof if c not in prof_ens.columns]
    if faltantes_prof:
        raise KeyError(
            f"Colunas ausentes na tabela2 da planilha do professor: {faltantes_prof}"
        )

    # 4. Calcula a rigidez dos ensaios do grupo e identifica a origem.
    grupo_ens["rigidez"] = (
        pd.to_numeric(grupo_ens["carga"], errors="coerce") /
        pd.to_numeric(grupo_ens["alongamento"], errors="coerce")
    )
    grupo_ens["origem"] = "Grupo"
    prof_ens["origem"] = "Professor"

    # Padroniza os campos numéricos.
    for coluna in ["ID", "carga", "alongamento", "rigidez"]:
        grupo_ens[coluna] = pd.to_numeric(grupo_ens[coluna], errors="coerce")
        prof_ens[coluna] = pd.to_numeric(prof_ens[coluna], errors="coerce")

    ensaios = pd.concat([
        grupo_ens[["ID", "carga", "alongamento", "rigidez", "origem"]],
        prof_ens[["ID", "carga", "alongamento", "rigidez", "origem"]]
    ], ignore_index=True)

    # 5. Prepara as características das molas.
    colunas_caracteristicas = ["ID", "C", "h", "t", "L", "n", "Ltot", "m"]

    for nome, tabela in [("grupo", grupo_char), ("professor", prof_char)]:
        faltantes = [c for c in colunas_caracteristicas if c not in tabela.columns]
        if faltantes:
            raise KeyError(
                f"Colunas ausentes na tabela de características ({nome}): {faltantes}"
            )

    grupo_char = grupo_char[colunas_caracteristicas].copy()
    grupo_char["origem"] = "Grupo"

    prof_char = prof_char[colunas_caracteristicas].copy()
    prof_char["origem"] = "Professor"

    caracteristicas = pd.concat(
        [grupo_char, prof_char], ignore_index=True
    )

    for coluna in colunas_caracteristicas:
        caracteristicas[coluna] = pd.to_numeric(
            caracteristicas[coluna], errors="coerce"
        )

    # 6. Calcula um resultado por mola.
    # K média: média das cinco medições.
    # IC 95%: intervalo t de Student, com 4 graus de liberdade.
    # Ductilidade adotada no projeto: alongamento da 5ª medição / C x 100.
    resultados = []

    t_critico_95_gl4 = 2.776445

    for _, mola in caracteristicas.iterrows():
        dados = ensaios[
            (ensaios["ID"] == mola["ID"]) &
            (ensaios["origem"] == mola["origem"])
        ].reset_index(drop=True)

        if len(dados) != 5:
            raise ValueError(
                f"Mola {mola['ID']} ({mola['origem']}) tem "
                f"{len(dados)} ensaios; eram esperados 5."
            )

        valores_k = dados["rigidez"].astype(float).to_numpy()

        if not all(math.isfinite(valor) for valor in valores_k):
            raise ValueError(
                f"Há valor de rigidez ausente ou inválido na mola "
                f"{mola['ID']} ({mola['origem']})."
            )

        n_medicoes = len(valores_k)
        media = valores_k.mean()
        dp = valores_k.std(ddof=1)
        margem = t_critico_95_gl4 * dp / math.sqrt(n_medicoes)

        quinto = dados.iloc[4]

        if pd.isna(mola["C"]) or mola["C"] == 0:
            raise ValueError(
                f"Comprimento C inválido para a mola {mola['ID']} "
                f"({mola['origem']})."
            )

        ductilidade = float(quinto["alongamento"]) / float(mola["C"]) * 100

        resultados.append({
            **mola.to_dict(),
            "K_media_kgf_mm": media,
            "K_desvio_padrao": dp,
            "IC95_inferior": media - margem,
            "IC95_superior": media + margem,
            "carga_5_kgf": quinto["carga"],
            "alongamento_5_mm": quinto["alongamento"],
            "ductilidade_5_pct": ductilidade
        })

    resumo = pd.DataFrame(resultados)

    # 7. Salva os arquivos de saída na mesma pasta do script.
    resumo.to_excel(SAIDA_TABELA, index=False)
    ensaios.to_excel(SAIDA_ENSAIOS, index=False)

    print("\nResultados calculados por mola:")
    print(resumo[
        ["ID", "origem", "K_media_kgf_mm",
         "IC95_inferior", "IC95_superior", "ductilidade_5_pct"]
    ].to_string(index=False))

    print("\nArquivos gerados com sucesso:")
    print(f"- {SAIDA_TABELA}")
    print(f"- {SAIDA_ENSAIOS}")


if __name__ == "__main__":
    main()
