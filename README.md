# Como utilizar a raspberry 5

Instale o RPI-Imager no terminal, para isso use o comando: 

```bash 
sudo apt install rpi-imager
```

* Conectar na máquina um cartão SD;

* Executar no Terminal:
```bash 
sudo rpi-imager
```
Selecione o S.O próprio da raspberry de 64bits, escolha aonde o cartão SD na aba storage e então clique em "Write";
Após a instalação, retire o cartão SD da máquina e insira-o no slot da raspberry(normalmente localizado na parte inferior da placa).

## Preparatório para ligar a Raspberry pi 5

Tenha em mãos uma fonte de energia com voltagem minima de 5V/3A e máxima de 5V/5A.
Na primeira iniciação da raspberry é preciso que você conecte-a a um monitor, teclado e mouse. para fazer as configurações do SSH inicial.

### Ligando a Raspberry

Conecte a fonte, o cartão SD, o cabo micro-usb/HDMI, um cabo de rede e os periféricos na placa, então conecte o lado hdmi do cabo em um monitor, e assim você estará pronto para iniciar as configurações.

#### Configurando

Agora com tudo funcionando, quando você ligar a raspberry será apresentado para você uma tela de configuração, aonde escolherá o nome de usuário, senha, rede wifi, e idioma. Faça as devidas configurações e você estará pronto para prosseguir.
Na tela inicial pressiona as teclas **CTRL** + **ALT** + **T**, ou utilize o menu suspenso para abrir o terminal.
com o terminal aberto você deve executar os seguintes comandos nessa ordem:
```bash
sudo apt install openssh-server # para instalar o ssh caso esteja instalado já, será para atualizar.
sudo service ssh status # para verificar se o ssh está funcionando na maquina.
sudo systemctl enable ssh # para permitir que o serviço SSH seja iniciado automaticamente na inicialização do sistema. 
sudo systemctl start ssh # para iniciar o serviço SSH imediatamente.
```

### Configurações adicionais 
Agora utilizaremos o menu de configuração do raspberry, utilize o seguinte comando:

```bash
sudo raspi-config
```
irá abrir a seguinte tela:

<img src="/README_images/main.png" alt="Página Principal do raspi-config">

Com a página aberta direcione utilizando as setas do teclado e selecione a opção: **3.Interface Options**, assim deverá abrir a seguinte tela:

<img src="/README_images/2.png" alt="Página 3.Interface Options">

Em seguida acesse a opção 1. SSH, e vá até **"Sim"** e pressione a tecla **ENTER**

<img src="/README_images/ssh.png" alt="Página 3.Interface Options">

Repita esse procedimento para todas as opções **EXCETO** a opção **I7. 1-WIRE**. Assim sua raspberry estará configurada e pronta para rodar seus experimentos!!
