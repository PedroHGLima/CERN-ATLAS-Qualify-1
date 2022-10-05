# CERN-ATLAS-Qualify/analysis/algorithm_quad/all_data

Esta análise foi feita junto com o Micael (micael.verissimo@lps.ufrj.br) para mostrar um exemplo de comparação em Ringer `V8` e Ringer `V11` sem outras etapas de filtragem (muito importante) - saída do modelo efetivamente sem nenhuma seleção (viés) de chain, sem trigger! Similar a um predict dos modelos para cada uma entradas.

## Para rodar o processamento dos dados

```console
source run_local_quadrant.sh
```

## Principais modificações nos arquivos 

### job_local_quadrant.py

```python
acc = EventATLAS( "EventATLASLoop",
                  inputFiles = args.inputFiles, 
                  treePath= '*/HLT/Physval/Egamma/probes',
                  dataframe = DataframeEnum.Electron_v1, 
                  outputFile = args.outputFile,
                  level = LoggingLevel.INFO
                  )
```
Alterações a partir da linha 40. Aqui a árvore (`TTree`) de análise foi modificada. A árvore em questão foi observada dentro dos arquivos de entrada que são utilizadas nesta análise.

```python
evt.setCutValue( SelectionType.SelectionOnlineWithRings )
```

```python
# apply selections
evt.setCutValue( SelectionType.SelectionPID, pidname ) 
evt.setCutValue( EtCutType.L2CaloAbove, 15.)
```

Alterações a partir da linha 50. Os cortes foram aplicados aqui. Neste caso, os cortes foram feitos para elétrons acima de `15 GeV`. O resto foi deixado padrão pelo Mica.


```python
# Add all chains into the emulator
emulator = ToolSvc.retrieve( "Emulator" )
# install selectors
installElectronL2CaloRingerSelector_v8()
installElectronL2CaloRingerSelector_v11()
```
Aqui precisamos adicionar os seletores de cada uma das versões do ringer. Tenho que colocar este caminho para encontrar os seletores!!!

```python
q_alg = QuadrantTool("Quadrant")
q_alg.add_quadrant( 
                # tight
                'ringer_v8_tight', 'T0HLTElectronRingerTight_v8', # Ringer v8
                'ringer_v11_tight', 'T0HLTElectronRingerTight_v11'  # Ringer v11
                )
q_alg.add_quadrant( 
                # medium
                'ringer_v8_medium', 'T0HLTElectronRingerMedium_v8', # Ringer v8
                'ringer_v11_medium', 'T0HLTElectronRingerMedium_v11'  # Ringer v11
                )
q_alg.add_quadrant( 
                # loose
                'ringer_v8_loose', 'T0HLTElectronRingerLoose_v8', # Ringer v8
                'ringer_v11_loose', 'T0HLTElectronRingerLoose_v11'  # Ringer v11
                )
q_alg.add_quadrant( 
                # very loose
                'ringer_v8_vloose', 'T0HLTElectronRingerVeryLoose_v8', # Ringer v8
                'ringer_v11_vloose', 'T0HLTElectronRingerVeryLoose_v11'  # Ringer v11
                )
```

Aqui temos adição e configurações.

### plot_quad.py
```python
alg.add_quadrant( "HLT_e28_lhtight_nod0_ringer_v8"  , "EMU_e28_lhtight_nod0_ringer_v8", # Ringer v8
                  'HLT_e28_lhtight_nod0_ringer_v11' , "EMU_e28_lhtight_nod0_ringer_v11" # Ringer v11
                ) 
```

Alterações a partir da linha 37. Aqui foram criadas as cadeias de processamento. Note que as cadeias que foram que cadeias de HLT, `e28` (elétrons com pelo menos 28 GeV de energia), para likelyhood tight (`lhtight`). A comparação entre as duas cadeias de ringer foi feita. As versões do ringer foram `V8` e `V11`.


