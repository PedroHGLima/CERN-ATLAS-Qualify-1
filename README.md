# CERN-ATLAS-Qualify

## Tarefa a ser relaizada
The ringer algorithm has been successfully implemented and deployed in the HLT fast step, improving early rejection of fake electrons and saving CPU time downstream in the HLT precision step. For this qualification task, the ringer algorithm training needs to be updated towards Run-3 conditions, in particular to cope with increased pileup. The training of the ringer selection will also consider boosted topologies to avoid reducing the efficiency of close by electrons as much as possible. Progress will be documented and discussed in TrigEgamma meetings, Jira tickets and MRs.

### Para rodar o repositório

### instalação docker (ubuntu 21.04)
Primeiro, vamos a instalação do docker

```console
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### instalação da engine do docker (ubuntu 21.04)

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

### instalação do singularity (ubuntu 21.04)

Como o time do projeto utiliza o **singularity**, precisamos instalar o Go para fazer tudo funcionar. Eu escolhi a versão **1.16.5** do Go e vou instalar a mesma. Caso seja necessário, você pode checar se tem outra versão disponível em [Go WebSite](https://golang.org/dl/)

```console
 wget https://dl.google.com/go/go1.16.5.linux-amd64.tar.gz
```

Extraia o arquivo e mova para **/usr/local** (todos os usuários com acesso)
```console
sudo tar -xvf go1.16.5.linux-amd64.tar.gz
sudo mv go /usr/local
```
