# Script que pode ser colocado dentro de .bashrc para
# que a placa fale em voz alta qual o IP dela (isso é
# bem útil quando mudamos muito de rede e não temos
# um IP fixo). Com este IP, dá para acessar a raspberry
# via ssh (evitando assim ficar conectando teclados e 
# mouse nela toda hora)

hostname -I | sed 's/\./ponto/g' > ip.txt
espeak -vpt-br "O meu IP é "
espeak -vpt-br -f ip.txt
espeak -vpt-br "Repetindo: "
espeak -vpt-br -f ip.txt
