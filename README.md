# programas para rodar no raspberry

Autor: Hemerson Pistori (pistori@ucdb.br)

### gravaFotos.py

Descrição: Capturar imagens a cada X segundos usando uma raspberry PI 3 B+ com uma webcam USB acoplada

Exemplo de uso: 


```
# Bate 50 fotos em intervalos de 10 segundos
python gravaFotos.py 10 50 
```

### ia.py 

Descrição: Roda uma IA pré-treinada para reconhecer gente

Exemplo de uso: 

```
# Processa 1 quadro a cada 30 lidos da webcam
python ia.py 30 
```

### Versões usadas no teste

- Hardware: Raspberry PI 3 B+ V1.2
- Sistema Operacional: Raspberry PI OS (64-BIT) - Debian Bullseye 
- Versão do python: 3.9.2
- Versão do opencv: 4.5.5.64


### Instalação do Sistema Operacional na Raspberry PI 3 B+

- Arrume um laptop ou computador com leitor de microSD e insira o cartão microSD no leitor
- Instale o software instalador da Raspberry baixando o arquivo .deb daqui https://www.raspberrypi.com/software e seguindo as instruções. Tem duas formas básicas (veja qual dá certo para você, para mim foi a primeira):

- Usando o snap com rpi-imager:

```
snap install rpi-imager
```

- Baixando o arquivo .deb e usando o dpkg [Assim não deu certo para mim]

```
sudo dpkg -i imager_1.7.2_amd64.deb
sudo apt-get -f install
```

- Execute o rpi-imager e instale o SO sugerido nas dependências (acima) inserindo o cartão SD no slot
- Altere o arquivo config.txt  dentro do diretório raiz do microSD se der problema com monitor HDMI com "No Signal". Descomente as linhas: 

```
     hdmi_safe=1
     hdmi_force_hotplug=1
```

   
- Tire o cartão da máquina e coloque no slot do raspberry (fica na parte de baixo da placa)
- Conecte monitor HDMI (na porta HDMI) e teclados e mouse nas portas USB
- Use um cabo USB-microUSB para ligar a placa Raspberry em uma fonte de energia (pode ser um carregador de celular de 5V e 2A ou uma saída USB do computador). A raspberry tem um único slot microUSB (aquele pequininho de celular) para ligar na energia.

# Instalação das dependências


```
# Programa de leitura de texto (Texto-To-Speech)
sudo apt-get install espeak
sudo pip install opencv-contrib-python pytorch torchvision 
```

- Se der erro de cv2 tenta instalar novamente o opencv

```
sudo pip install opencv-contrib-python
```

- Se der erro no pytorch, tenta seguir estas orientações aqui (instalar usando WHEEL):
https://qengineering.eu/install-pytorch-on-raspberry-pi-4.html. Também coloquei os comandos no script install_torch.sh

```
./install_torch.sh
```


- Se der erro no numpy com um warning onde aparece um número hexadecimal (E.g.: 0x10), busque neste site o mapa de conversão para saber que versão do numpy deve ser instalada. No meu caso, o erro mostra 0x10 e pelo mapa eu vi que tinha que instalar o numpy 1.23

```
sudo pip install numpy=1.23
```


### Para instalar o seu programa em python na raspberry

- Depois de montar o microSD na sua máquina copie os programas para dentro dele
- O microSD será montado em uma pasta chamada /media/NOME_USUARIO/ (onde NOME_USUARIO depende da sua instalação específica ... geralmente é o seu nome de login)
- Dentro desta pasta haverá uma chamada rootfs. É nesta que fica todo o sistema de arquivos que te interessa, incluindo a pasta "home".
- Copie todo o conteúdo de raspberry_camera para home/NOME_USUARIO_RASPBERRY (o padrão é "pi"). Exemplo aqui na minha máquina (meu usuário é pistori e no raspberry eu havia criado um usuário chamado papi):

```
sudo cp -R raspberry_camera/ /media/pistori/rootfs/home/papi/
```

- Altere o arquivo /home/NOME_USUARIO_RASPBERRY/.bashrc para chamar o seu programa. Coloque os comandos abaixo bem no final do arquivo .bashrc  (troque papi pelo nome correto da pasta)
  
```
cd /home/papi/raspberry_camera/
# Tem que esperar alguns segundos para que a conexão com a rede se estabeleça.
# Caso contrário, não vai ter IP para poder falar. 
sleep 6
# Vai falar o IP da máquina, caso tenha conseguido conectar
# em alguma rede. Com isso, dá para logar no raspberry
# via ssh
./fala_IP.sh
# Roda a IA processamento 1 a cada 30 quadros
python ia.py 30 > saida.txt 2> saida_erro.txt &
cd ~
```

### Dicas adicionais
- Para logar na rede wifi da UCDB, que usa um protocolo de segurança WPA2 diferente do padrão das redes domésticas, usei estas orientações aqui: https://gist.github.com/davidhoness/5ee50e881b63c7944c25b8de33453823
- Para alterar login altere o arquivo (no microSD) /etc/wpa_supplicant/wpa_supplicant.conf
- Para alterar a senha, neste mesmo arquivo, você precisa primeiro gerar uma chave hash,
  por segurança, usando os comandos abaixo (e copiar para o arquivo wpa_supplicant.conf os
  números gerados, coloque depois de "hash:"
```  
  echo -n sua_senha | iconv -t utf16le | openssl md4
  # Para reconfigurar a rede:
  wpa_cli -i wlan0 reconfigure
  # Para limpar sua senha do histórico
  history -c
  rm ~/.bash_history
```
  
- Para testar a webcam USB eu usei estas dicas aqui:
  https://raspberrypi-guide.github.io/electronics/using-usb-webcams
  



 

