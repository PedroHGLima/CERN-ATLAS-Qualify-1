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

