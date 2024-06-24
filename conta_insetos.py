"""

Autor: Hemerson Pistori

Funcionalidade: rodar uma IA no raspberry que reconhece que tem gente na frente da câmera. A IA foi pré-treinada usando o exemplo_pytorch_v4 disponível aqui: http://git.inovisao.ucdb.br/inovisao/exemplos_pytorch

Vai processar a cada X quadros (X é um parâmetro)


Exemplo de uso (pegando um frame de cada 30):
$ python ia.py 30

"""

import torch 
from torch import nn  # Módulo para redes neurais (neural networks)
import torchvision.transforms as transforms
from torchvision import models
import os
import cv2
from PIL import Image
import sys
import time
import numpy as np
from subprocess import call

if len(sys.argv[1:]) == 0:
   print('Faltou passar a quantidade de segundos entre cada foto que será tirada')
   exit(0)

# Pega o intervalo em segundo da linha de comando
taxa_de_quadros=int(sys.argv[1])

print('Irá processar 1 a cada ',taxa_de_quadros,' quadros')

# Verifica se tem GPU na máquina, caso contrário, usa a CPU mesmo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando {device}")


#model = models.resnet18(pretrained=True) # Avisa que é uma resnet18 pré-treinada
#model.fc = nn.Linear(512, len(classes)) # Muda a última camada da rede para 2 classes de saída apenas 
#model.to(device)  # Ajusta para o dispositivo (CPU ou GPU)
#model.load_state_dict(torch.load('./modelo_treinado_resnet.pth', map_location=device)) # Carrega os pesos


# Função que detecta os insetos na imagem
def detecta_insetos(imagem):

  conta_marrom=5
  conta_verde=4

  # TEM QUE TROCAR ESTA PARTE DE BAIXO PELA FASTER R-CNN 
  #model.eval() # Avisa que a rede está no modo de "uso" e não de "aprendizagem"
  #transform = transforms.Compose([transforms.Resize((224,224)),  
  #                              transforms.ToTensor()
  #                          ])

  #imagem = transform(imagem).to(device).unsqueeze(0) # Ajusta para o formato que a rede precisa
  #predicao = model(imagem).argmax(dim=1).cpu().tolist()  # Realiza a classificação
  
  return conta_marrom,conta_verde

# Prepara para ler imagens da webcam
cam = cv2.VideoCapture(0)

parametros_fala='-s 200 -p 10 -v brazil'

time.sleep(1)  # Espera um pouco para não dar para no comando que será chamado
comando=['espeak '+parametros_fala+' "Olá, sou uma   IA" 2>/dev/null']
call(comando, shell=True)     

quadro=1 
n_img=1
while True:
   ret, imagem = cam.read()  # Lê um quadro da webcam
   imagemRGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB).astype(np.float32)  # Converte de BGR para RGB
   imagemPIL = Image.fromarray(np.uint8(imagemRGB))  # Converte do formato OpenCV para PIL

   if quadro % taxa_de_quadros == 0:
      print(quadro)
      marrons, verdes = detecta_insetos(imagemPIL)  # Vai classificar a imagem (usa o formato PIL)

      if marrons==0 and verdes==0:
         comando=['espeak '+parametros_fala+'"Não vejo percevejos" 2>/dev/null']
      if marrons>0 and verdes==0:
         comando=['espeak '+parametros_fala+' "Vejo '+str(marrons)+' percevejos marrons" 2>/dev/null']  
      if marrons==0 and verdes>0:
         comando=['espeak '+parametros_fala+' "Vejo '+str(verdes)+'" percevejos verdes" 2>/dev/null']
      if marrons>0 and verdes>0:
         comando=['espeak '+parametros_fala+' "Vejo '+str(marrons)+' percevejos marrons e '+str(verdes)+' verdes" 2>/dev/null']
      
      call(comando, shell=True)     
   quadro+=1
	
cam.release()
cv2.destroyAllWindows()


  
