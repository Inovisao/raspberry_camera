# Tira fotos a cada X segundos (X é um parâmetro)
# Autor: Hemerson Pistori
# Exemplo de uso (tirar fotos a cada 10 segundos):
# $ python gravaFotosIntervalo.py 10

import cv2
import sys
import time
from playsound import playsound


if len(sys.argv[1:]) == 0:
   print('Faltou passar a quantidade de segundos como parâmetro')
   exit(0)

segundos=int(sys.argv[1])

print('Irá capturar imagens a cada ',segundos,' segundos')


cam = cv2.VideoCapture(0)

i=1 

while i<=10:
   ret, image = cam.read()
   nome_arquivo='img_'+f"{i:05}"+'.jpg'
   print('Salvando ',nome_arquivo)
   cv2.imwrite(nome_arquivo, image)
   playsound('gravei.mp3')
   time.sleep(segundos)
   i=i+1
	
cam.release()
cv2.destroyAllWindows()
