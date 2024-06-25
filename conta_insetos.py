"""

Autores: Hemerson Pistori, João Porto

Funcionalidade: rodar uma IA no raspberry que reconhece que tem gente na frente da câmera. A IA foi pré-treinada usando o exemplo_pytorch_v4 disponível aqui: http://git.inovisao.ucdb.br/inovisao/exemplos_pytorch

Vai processar a cada X quadros (X é um parâmetro)


Exemplo de uso (pegando um frame de cada 30):
$ python ia.py 30

"""

import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import cv2
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

# Função que detecta os insetos na imagem
CLASSES = [
    '_background_',  # Background class
    'marrom',        # Brown insect class
    'verde'          # Green insect class
]

# Define model and load pre-trained weights
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES))
model_path = 'modelo_treinado_faster.pth'  # Path to your trained model
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# Define thresholds
threshold = 0.75
iou_threshold = 0.5

def detecta_insetos(image):
    # Preprocess the image
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float).to(device)
    image = torch.unsqueeze(image, 0)

    # Get the predictions
    with torch.no_grad():
        results = model(image)

    results = [{k: v.to('cpu') for k, v in t.items()} for t in results]

    # Apply Non-Maximum Suppression
    for result in results:
        boxes = result['boxes']
        scores = result['scores']
        labels = result['labels']

        keep = nms(boxes, scores, iou_threshold)

        result['boxes'] = boxes[keep]
        result['scores'] = scores[keep]
        result['labels'] = labels[keep]

    # Count the number of detections for 'verde' and 'marrom' with score >= threshold
    verdes = 0
    marrons = 0
    for result in results:
        labels = result['labels'].numpy()
        scores = result['scores'].numpy()
        for label, score in zip(labels, scores):
            if score >= threshold:
                class_name = CLASSES[label]
                if class_name == 'verde':
                    verdes += 1
                elif class_name == 'marrom':
                    marrons += 1

    return marrons, verdes

def fala(texto):
   parametros_fala='-s 200 -p 10 -v brazil'
   comando=['espeak '+parametros_fala+' "'+texto+'" 2>/dev/null']
   call(comando, shell=True)

# Prepara para ler imagens da webcam
cam = cv2.VideoCapture(0)

time.sleep(1)  # Espera um pouco para não dar para no comando que será chamado

fala('Olá, eu busco percevejos')


quadro=1 
n_img=1
while True:
   ret, imagem = cam.read()  # Lê um quadro da webcam
   imagemRGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB).astype(np.float32)  # Converte de BGR para RGB

   if quadro % taxa_de_quadros == 0:
      marrons, verdes = detecta_insetos(imagemRGB)  # Vai classificar a imagem (usa o formato PIL)
      print(f'marrons: {marrons}, verdes: {verdes}')

      if marrons==0 and verdes==0:
         fala('Não vejo percevejos')
      if marrons>0 and verdes==0:
         if marrons==1:
            fala('Vejo um percevejo marrom')
         else:
            fala('Vejo '+str(marrons)+' percevejos marrons')
      if marrons==0 and verdes>0:
         if verdes==1:
            fala('Vejo um percevejo verde')
         else:
            fala('Vejo '+str(verdes)+' percevejos verdes')
      if marrons>0 and verdes>0:
         if marrons==1 and verdes==1:
            fala('Vejo um percevejo marrom e um percevejo verde')
         elif marrons==1 and verdes>1:
            fala('Vejo um percevejo marrom e '+str(verdes)+' percevejos verdes')
         elif marrons>1 and verdes==1:
            fala('Vejo '+str(marrons)+' percevejos marrons e um percevejo verde')
         else:
            fala('Vejo '+str(marrons)+' percevejos marrons e '+str(verdes)+' percevejos verdes')

      
   quadro += 1

   # Verifica se a tecla 'q' ou 'Esc' foi pressionada para sair
   key = cv2.waitKey(1) & 0xFF
   if key == ord('q') or key == 27:  # 'q' or 'Esc'
       break
	
cam.release()
cv2.destroyAllWindows()


  
