# Tira fotos a cada X segundos (X é um parâmetro)
# Autor: Hemerson Pistori
# Exemplo de uso (tirar fotos a cada 10 segundos):
# $ python gravaFotosIntervalo.py 10

import cv2
import sys
import time
from playsound import playsound
from subprocess import call

if len(sys.argv[1:]) == 0:
   print('Faltou passar a quantidade de segundos como parâmetro')
   exit(0)

segundos=int(sys.argv[1])

print('Irá capturar imagens a cada ',segundos,' segundos')


cam = cv2.VideoCapture(0)

# Deu um no comando abaixo quando tentei rodar na raspberry, mas pode
# ser por conta da saída vinculada à porta HDMI (retirando o cabo
# HDMI talvez funcione). Tem que descomentar para testar
#playsound('vai_comecar.mp3')

time.sleep(2)
call(['espeak -vpt-br -k 10 "Vai Começar" 2>/dev/null'], shell=True)     

i=1 

while i<=5:
   ret, image = cam.read()
   nome_arquivo='img_'+f"{i:05}"+'.jpg'
   comando=['espeak -vpt-br "Foto '+str(i)+'" 2>/dev/null']
   call(comando, shell=True)     
   print('Salvando ',nome_arquivo)
   cv2.imwrite(nome_arquivo, image)
   time.sleep(segundos)
   i=i+1
	
cam.release()
cv2.destroyAllWindows()
