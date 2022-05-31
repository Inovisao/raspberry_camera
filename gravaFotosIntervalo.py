# Tira fotos a cada X segundos (X é um parâmetro)
# Autor: Hemerson Pistori
#
# Exemplo de uso: tirar fotos a cada 3 segundos usando a webcam:
#
# $ python gravaFotosIntervalo.py 3
#
# Exemplo de uso: se não tiver câmera, pode passar como segundo
#   parâmetro o nome de um arquivo
#
# $ python gravaFotosIntervalo.py 3 ./meu_video.mp4
#

import cv2
import sys
import time
from playsound import playsound
from subprocess import call

if len(sys.argv[1:]) == 0:
   print('Faltou passar a quantidade de segundos como parâmetro')
   exit(0)

segundos=int(sys.argv[1]) # Pega a quantidade de segundos entre fotos
nomeArquivoVideo=""

print('Irá capturar imagens a cada ',segundos,' segundos')

if len(sys.argv[1:]) > 1: # Passou um nome de arquivo no segundo parâmetro
   nomeArquivoVideo=sys.argv[2]

if nomeArquivoVideo=="": # Se não tem nome de arquivo usa a webcam
   print('Vai ler os quadros da webcam')
   cam = cv2.VideoCapture(0)
else:
   print('Vai ler o quadros de um arquivo de vídeo')
   cam = cv2.VideoCapture(nomeArquivoVideo)   

# Deu um no comando abaixo quando tentei rodar na raspberry, mas pode
# ser por conta da saída vinculada à porta HDMI (retirando o cabo
# HDMI talvez funcione). Tem que descomentar para testar
#playsound('vai_comecar.mp3')

# Executa o programa espeak que lê um texto
time.sleep(2)
call(['espeak -vpt-br -k 10 "Vai Vai Começar" 2>/dev/null'], shell=True)     

i=1 

# Vai tirar 5 fotos e depois parar (troque por True se
# quiser que não pare nunca)
while i<=10:
   ret, image = cam.read()
   if ret==False:
      print('Não conseguiu ler o arquivo ou abrir a webcam')
      call(['espeak -vpt-br -k 10 "Deu problema na webcam" 2>/dev/null'], shell=True)     
      exit(0)
   nome_arquivo='img_'+f"{i:05}"+'.jpg'
   comando=['espeak -vpt-br "Foto '+str(i)+'" 2>/dev/null']
   call(comando, shell=True)     
   print('Salvando ',nome_arquivo)
   cv2.imwrite(nome_arquivo, image)
   time.sleep(segundos)
   i=i+1
	
	
call(['espeak -vpt-br -k 10 "Ter Terminei" 2>/dev/null'], shell=True)     
	
cam.release()
cv2.destroyAllWindows()
