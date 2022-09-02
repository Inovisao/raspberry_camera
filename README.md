# programas para rodar no raspberry

Autor: Hemerson Pistori (pistori@ucdb.br)

### gravaFotos.py

Descrição: Capturar imagens a cada X segundos usando uma raspberry PI 3 B+ com uma webcam USB acoplada

Exemplo de uso: python gravaFotos.py 10 50 (bate 50 fotos em intervalos de 10 segundos)

### ia.py [EM CONSTRUÇÃO !!!]

Descrição: Roda uma IA pré-treinada para reconhecer gente

Exemplo de uso: python ia.py 2 (pega fotos a cada 2 segundos)

### Dependências 

- Hardware: Raspberry PI 3 B+ 
- Sistema Operacional: Raspberian 64 bit
- Versão do python: 3.9.2
- Versão do opencv: 4.5.5.64
- Outras dependências a serem instaladas: 

```
pip install opencv-contrib-python  
sudo apt-get install espeak
```

### Dicas para começar a usar uma placa Raspberry PI 3 B+

- Arrume um laptop ou computador com leitor de microSD e insira o cartão microSD no leitor
- Instale o software instalador da Raspberry baixando o arquivo .deb daqui https://www.raspberrypi.com/software e seguindo as instruções. Tem duas formas básicas (veja qual dá certo para você):
- Usando o snap:

```
snap install rpi-imager
```

- Baixando o arquivo .deb e usando o dpkg

```
sudo dpkg -i imager_1.7.2_amd64.deb
sudo apt-get -f install
```

- Execute o rpi-imager e instale o SO Raspberian 64 bit inserindo o cartão SD no slot
- Altere o arquivo config.txt  dentro do diretório raiz do microSD para resolver problema com  monitor HDMI com "No Signal". Descomente as linhas: 

```
     hdmi_safe=1
     hdmi_force_hotplug=1
```

   
- Tire o cartão da máquina e coloque no slot do raspberry (fica na parte de baixo da placa)
- Conecte monitor HDMI (na porta HDMI) e teclados e mouse nas portas USB
- Use um cabo USB-microUSB para ligar a placa Raspberry em uma fonte de energia (pode ser um carregador de celular de 5V e 2A ou uma saída USB do computador). A raspberry tem um único slot microUSB (aquele pequininho de celular) para ligar na energia.

### Para instalar o seu programa em python na raspberry

- Depois de montar o microSD na sua máquina copie os programas para ele
- Coloque o microSD de volta na raspberry. Os arquivos que você copiou ficarão na pasta /boot
- Copie os arquivos que você precisa da pasta boot para a pasta /home/pi/raspberry_camera
- Altere o arquivo /home/pi/.bashrc para chamar o seu programa. Coloque os comandos abaixo bem no final do arquivo .bashrc  
- Troque 10 e 50 pelo intervalo em segundo e total de imagens que quer capturar
- IMPORTANTE: dá também para acessar a pasta /home/pi/ sem precisar colocar o microSD de volta
  na raspberry. Neste caso, procure por /media/ALGUMA_COISA/rootfs/home/pi (ALGUMA COISA é nome 
  que você usou na hora de formatar o microSD, no meu caso, usei "pistori"
  
```
cd /home/pi/raspberry_camera/
python gravaFotosIntervalo.py 10 50  > saida.txt 2> saida_erro.txt &
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
```

  
- Para testar a webcam USB eu usei estas dicas aqui:
  https://raspberrypi-guide.github.io/electronics/using-usb-webcams

