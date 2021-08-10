# CERN-ATLAS-Qualify/analysis/e28_quad

Esta análise foi feita junto com o Micael (micael.verissimo@lps.ufrj.br) para mostrar um exemplo quadrantes utilizados no Trigger

## Para rodar o processamento dos dados

```console
source run_job_quad.sh
```

Depois de rodar este comando, uma pasta chamada  `egam1_test` com um arquivo `egam1_test.root` dentro. Este arquivo deve ser processado para gerar os gráficos de análise formatado da maneira que o grupo utiliza.

## Para rodar o processamento dos gráficos

```console
source run_plot.sh
```

Este comando vai rodar somente na imagem do Juan. Fique atento a isso! E vai gerar os plots em um formato visualização mais simples. Uma vez que este processo termine de rodar, você terá acesso a todos os gráficos plotados sem nenhum problema.

## Principais modificações nos arquivos 

### job_quad.py

```python
triggerList = [
                Chain( "EMU_e28_lhtight_nod0_ringer_v8" ,  "L1_EM3", "HLT_e28_lhtight_nod0_ringer_v8"  ),
                Chain( "EMU_e28_lhtight_nod0_ringer_v11",  "L1_EM3", "HLT_e28_lhtight_nod0_ringer_v11" ),
              ]
```
Alterações a partir da linha 73. Aqui foram criadas as cadeias de processamento. Note que as cadeias que foram que cadeias de HLT, `e28` (elétrons com pelo menos 28 GeV de energia), para likelyhood tight (`lhtight`). A comparação entre as duas cadeias de ringer foi feita. As versões do ringer foram `V8` e `V11`.

```python
q_alg.add_quadrant( "HLT_e28_lhtight_nod0_ringer_v8"  , "EMU_e28_lhtight_nod0_ringer_v8", # Ringer v8
                    'HLT_e28_lhtight_nod0_ringer_v11' , "EMU_e28_lhtight_nod0_ringer_v11" # Ringer v11
                  ) 
```
Alterações a partir da linha 89. Aqui foram adicionadas as chains a análise de quadrante. 

### plot_quad.py
```python
alg.add_quadrant( "HLT_e28_lhtight_nod0_ringer_v8"  , "EMU_e28_lhtight_nod0_ringer_v8", # Ringer v8
                  'HLT_e28_lhtight_nod0_ringer_v11' , "EMU_e28_lhtight_nod0_ringer_v11" # Ringer v11
                ) 
```

Alterações a partir da linha 37. Aqui foram criadas as cadeias de processamento. Note que as cadeias que foram que cadeias de HLT, `e28` (elétrons com pelo menos 28 GeV de energia), para likelyhood tight (`lhtight`). A comparação entre as duas cadeias de ringer foi feita. As versões do ringer foram `V8` e `V11`.


