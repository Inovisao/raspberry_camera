# Tira fotos a cada X segundos (X é um parâmetro)
# Autor: Hemerson Pistori
# Exemplo de uso (tirar 50 fotos em intervalos de 10 segundos):
# $ python gravaFotosIntervalo.py 10 50

import cv2
import sys
import time
from subprocess import call

if len(sys.argv[1:]) == 0:
   print('Faltou passar a quantidade de segundos como parâmetro e o total de fotos')
   exit(0)

segundos=int(sys.argv[1])
total_de_fotos=int(sys.argv[2])

print('Irá capturar',total_de_fotos,'imagens em intervalos de',segundos,'segundos')


cam = cv2.VideoCapture(0)


time.sleep(2)
comando=['espeak -vpt-br "Eu vou tirar '+str(total_de_fotos)+' fotos em intervalos de '+str(segundos)+'segundos" 2>/dev/null']
call(comando, shell=True)     

i=1 

while i<=total_de_fotos:
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
