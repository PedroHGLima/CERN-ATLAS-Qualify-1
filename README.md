# CERN-ATLAS-Qualify

## Tarefa a ser realizada
The ringer algorithm has been successfully implemented and deployed in the HLT fast step, improving early rejection of fake electrons and saving CPU time downstream in the HLT precision step. For this qualification task, the ringer algorithm training needs to be updated towards Run-3 conditions, in particular to cope with increased pileup. The training of the ringer selection will also consider boosted topologies to avoid reducing the efficiency of close by electrons as much as possible. Progress will be documented and discussed in TrigEgamma meetings, Jira tickets and MRs.

### Para rodar os códigos do time do ATLAS Ringer

### Instalação docker (ubuntu 21.04)
Primeiro, vamos a instalação do docker

```console
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### Instalação da engine do docker (ubuntu 21.04)

```console
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```
Comando para listar as possíveis enginer disponíveis (eu instalei a **5:20.10.7\~3-0\~ubuntu-hirsute**)

```console
sudo apt-cache madison docker-ce
sudo apt-get install docker-ce=<VERSION_STRING> docker-ce-cli=<VERSION_STRING> containerd.io
```

Para testar tudo

```console
sudo docker run hello-world
```

### Instalação do singularity (ubuntu 21.04)

Como o time do projeto utiliza o **singularity**, precisamos instalar o Go para fazer tudo funcionar. Eu escolhi a versão **1.16.5** do Go e vou instalar a mesma. Caso seja necessário, você pode checar se tem outra versão disponível em [Go WebSite](https://golang.org/dl/)

```console
 wget https://dl.google.com/go/go1.16.5.linux-amd64.tar.gz
```

Extraia o arquivo e mova para **/usr/local** (todos os usuários com acesso)
```console
sudo tar -xvf go1.16.5.linux-amd64.tar.gz
sudo mv go /usr/local
```

#### Setup do environment do Go
É necessário ter 3 variáveis de ambiente para utilizar o Go. Portanto, você deve adicionar as seguintes linhas no arquivo **$HOME/.bashrc**):
```console
export GOROOT=/usr/local/go
export GOPATH=$HOME/go
export PATH=$GOPATH/bin:$GOROOT/bin:$PATH
```

Verifique a instalação
```console
go version
```
#### Instalação singularity

Clone o repositório do Singularity
```console
mkdir -p $GOPATH/src/github.com/sylabs
cd $GOPATH/src/github.com/sylabs
git clone https://github.com/sylabs/singularity.git
cd singularity
```

Instale as dependências do Go:
```console
go get -u -v github.com/golang/dep/cmd/dep
```

Compile o Singularity
```console
cd $GOPATH/src/github.com/sylabs/singularity
./mconfig
make -C builddir
sudo make -C builddir install
```

Verifique a instalação
```console
singularity --version
```

### Clonagem da imagem do projeto
O João Victor (lider do time de desenvolvimento) fez uma versão da image do docker (que será utilizada com o singularity) e esta está em [link](https://hub.docker.com/r/jodafons/ringer)

```console
singularity pull docker://jodafons/ringer:base
```

Em seguida, após o download da imagem, execute o seguinte comando para rodar:
OBS: O argumento `--nv` deve ser utilizado caso o computador tenha uma placa de vídeo para que a GPU seja utilizada no container.

```console
singularity run --nv ringer_base.sif
```

Agora dentro do ambiente, faça o setup das dependências:
```console
source /setup_all_here.sh ringer-atlas
```

OBS: O atributo “ringer-atlas” no final desse comando direciona o setup para fazer o clone das frameworks localizadas no repositório do ringer-atlas ([link](https://github.com/ringer-atlas)). Foi recomendado pelo Micael fazer um fork dos repositórios para o seu próprio usuário no github e na hora de executar o comando direcionar para o seu usuário. Por exemplo, no meu caso:


```console
source /setup_all_here.sh natmourajr
```

### Instalação ROOT (ubuntu 21.04)

O ROOT é um framework de análise escrito e suportado por físicos em [link](https://root.cern/). A instalação sempre é complexa e eu estou instalação a versão v6.16.00 (utilizada pelo grupo do Ringer-ATLAS). Para executar a instalação, rode o código abaixo (é necessária a senha de root do computador e fique atento que este script vai gerar duas pastas meio grandes no local onde o código será rodado.)

```console
cd <path>/CERN-ATLAS-Qualify
source installROOT.sh
```