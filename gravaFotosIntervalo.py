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

playsound('vai_comecar.mp3')

i=10 

while i<=30:
   ret, image = cam.read()
   nome_arquivo='img_'+f"{i:05}"+'.jpg'
   comando=['espeak -vpt-br "'+str(i)+'" 2>/dev/null']
   call(comando, shell=True)     
   print('Salvando ',nome_arquivo)
   cv2.imwrite(nome_arquivo, image)
   time.sleep(segundos)
   i=i+1
	
cam.release()
cv2.destroyAllWindows()
