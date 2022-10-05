# CERN-ATLAS-Qualify/docker_images/juan
Esta imagem, aparentemente, tem todas as fontes e aplicativos para rodar tudo.

## Instruções do Juan

Dentro dessa imagem, na raiz, tem dois arquivos: download_packages.sh e setup_enviroment.sh

### download_packages.sh 
o primeiro é para baixar os repositorios (caso nao estejam baixados). Ele recebe dois argumentos: o primeiro é o nome do usuario do repostorio (ex: ringer-atlas) e o segundo o caminho para a pasta que vc quer o repositorio (ex: `/home/juan/`). Ele criará uma pasta chamada git_repos

### setup_enviroment.sh
o segundo publica tudo no ambiente e recebe apenas um argumento de entrada: o diretorio dos repositorios. Nesse exemplo, seria rodar: `source setup_packages.sh home/juan/git_repos/`

```console
source /setup_packages.sh /home/juan/git_repos/
```





