# gravaFotosInvervalo

Descrição: Capturar imagens a cada X segundos usando uma raspberry PI 3 B+ com uma webcam USB acoplada

Autor: Hemerson Pistori (pistori@ucdb.br)

Exemplo de uso: python gravaFotos.py 30 (bate uma foto a cada 30 segundos)

### Dependências 

- Hardware: Raspberry PI 3 B+ 
- Sistema Operacional: Raspberian 64 bit
- Versão do python: 3.9.2
- Versão do opencv: 4.5.5.64
- Versão do playsound: 1.3.0
- Outras dependências a serem instaladas: 

```
pip install opencv-contrib-python playsound 
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
- Copie a pasta /boot/raspberry_camera para /home/pi/

```
cd ~
sudo cp /boot/raspberry_camera . 
```

- Altere o arquivo /home/pi/.bashrc para chamar o seu programa. Coloque estes comandos aqui bem no final do arquivo .bashrc  
  
```
cd /home/pi/raspberry_camera/
python gravaFotosIntervalo.py 3 > saida.txt 2> saida_erro.txt &
cd ~  
```

### Dicas adicionais
- Para logar na rede wifi da UCDB, que usa um protocolo de segurança WPA2 diferente do padrão das redes domésticas, usei estas orientações aqui: https://gist.github.com/davidhoness/5ee50e881b63c7944c25b8de33453823
- Para testar a webcam USB eu usei estas dicas aqui:
  https://raspberrypi-guide.github.io/electronics/using-usb-webcams

