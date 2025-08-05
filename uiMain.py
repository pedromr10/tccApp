import tkinter as tk


def criarDataset():
    print("teste")

def mostrarEmocao():
    print("teste")

#Tela inicial:
janela = tk.Tk()
janela.title("Identificador de emocoes")
janela.geometry("600x300")

#Titulo:
label = tk.Label(janela, text="TCC", font=("Arial", 14))
label.pack(pady=10)

#BotãoCriarDataset
botaoDataset = tk.Button(janela, text="Criar Dataset", command=criarDataset)
botaoDataset.pack()

#a parte 2 do codigo poderia ser automatica.

#BotãoMostrarEmocoes
botaoEmocao = tk.Button(janela, text="Emocoes", command=mostrarEmocao)
botaoEmocao.pack()

# Inicia o loop da interface
janela.mainloop()
