import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Função para atualizar a imagem
def update_frame():
    ret, frame = cam.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    label.after(10, update_frame)

# Inicializa a janela
root = tk.Tk()
root.title("Câmera")

# Prepara a captura de vídeo
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Erro: Câmera não acessível.")
    exit(0)

label = tk.Label(root)
label.pack()

update_frame()
root.mainloop()

# Libera a câmera
cam.release()