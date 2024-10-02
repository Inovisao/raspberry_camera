import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import cv2
import sys
import time
import numpy as np
from gtts import gTTS
import os
import tkinter as tk
from PIL import Image, ImageTk

if len(sys.argv[1:]) == 0:
    print('Faltou passar a quantidade de quadros que será processada')
    exit(0)

# Pega o intervalo em quadros da linha de comando
taxa_de_quadros = int(sys.argv[1])

print('Irá processar 1 a cada', taxa_de_quadros, 'quadros')

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
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float).to(device)
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        results = model(image)

    results = [{k: v.to('cpu') for k, v in t.items()} for t in results]

    for result in results:
        boxes = result['boxes']
        scores = result['scores']
        labels = result['labels']
        keep = nms(boxes, scores, iou_threshold)

        result['boxes'] = boxes[keep]
        result['scores'] = scores[keep]
        result['labels'] = labels[keep]

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
    tts = gTTS(texto, lang="pt")
    tts.save("audio.mp3")
    os.system('ffplay -af "atempo=1.5" -autoexit -nodisp audio.mp3')

# Função para atualizar a imagem da câmera no Tkinter
def update_frame():
    global quadro
    ret, imagem = cam.read()
    if not ret:
        print("Erro: Não foi possível ler a imagem da câmera.")
        return

    imagemRGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    if quadro % taxa_de_quadros == 0:
        marrons, verdes = detecta_insetos(imagemRGB)
        print(f'marrons: {marrons}, verdes: {verdes}')
        
        if marrons == 0 and verdes == 0:
            fala('Não vejo percevejos')
        elif marrons > 0 and verdes == 0:
            fala(f'Vejo {marrons} percevejos marrons')
        elif marrons == 0 and verdes > 0:
            fala(f'Vejo {verdes} percevejos verdes')
        elif marrons > 0 and verdes > 0:
            fala(f'Vejo {marrons} percevejos marrons e {verdes} percevejos verdes')

    # Atualiza a imagem na interface Tkinter
    img = Image.fromarray(imagemRGB)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    quadro += 1
    label.after(10, update_frame)

# Prepara a captura de vídeo
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Erro: Câmera não acessível.")
    exit(0)

time.sleep(1)  # Espera um pouco para não dar para no comando que será chamado

fala('Olá, eu busco percevejos')

# Inicializa a interface Tkinter
root = tk.Tk()
root.title("Detecção de Insetos")

label = tk.Label(root)
label.pack()

quadro = 1
update_frame()  # Inicia a atualização do frame
root.mainloop()  # Inicia o loop principal da interface

# Libera a câmera
cam.release()
