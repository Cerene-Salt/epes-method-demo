import pandas as pd
import matplotlib.pyplot as plt

Casa_de_Banho = [
  "Copo",
  "Papel Higiênico",
  "Dispensador de creme e sabão das mãos",
  "Sabão",
  "Rede de Banho",
  "Pepsodente",
  "Piaçaba",
  "Toalha de rosto e de banho",
  "Balde de lixoe e sacos de lixo ",
  "Gel de Banho",
  "Champoo e Condicionador",
  "Detergente pato para a sanita",
  "Pastilhas para a sanita",
  "Gel de casa e de banho",
  "Mum",
  "Creme para o corpo",
  "Exfoliante para rosto."
]

Cozinha = [
  "Pano de Limpeza",
  "Esponja de louça",
  "Detergente para louça",
  "Chaleira",
  "Prato e Copos",
  "Talheres",
  "Sal, pimenta, molhos",
  "Liquidificador",
  "Vassoura e pá",
  "Esfregona e Balde",
  "Saco de lixo."
]

while len(Cozinha) != Casa_de_Banho:
  Cozinha.append("")
  if len(Cozinha) == len(Casa_de_Banho):
    print("Done")
    break

 


Quarto = [
  "Lençõis",
  "Endredões",
  "Almofadas",
  "Balde de lixo",
  "Tapete",
  "Cabides",
  "Ficha Tripla",
  "Picos para quadro de anotações",
  "Ambiente."
]




Comida = [
  "Legumes",
  "Fruta",
  "Taparões",
  "Panelas",
  "Frigideira."
]

while len(Comida) != len(Quarto):
  Comida.append("")
  if len(Comida) == len(Quarto):
    print("Done also.")
    print(len(Comida))
    break



Lista_de_compras = {
   "Casa_de_Banho":Casa_de_Banho,
   "Cozinha":Cozinha,
   "Quarto":Quarto,
   "Comida":Comida
}

while len(Quarto) != len(Casa_de_Banho):
  Quarto.append("")
  if len(Quarto) == len(Casa_de_Banho):
    print("Done finally.")
    break

#checking
print(len(Quarto))
print(len(Cozinha))
#If one is different
while len(Comida) != len(Casa_de_Banho):
  Comida.append("")
  if len(Comida) == len(Casa_de_Banho):
    print("Completed")

#checking
print(len(Quarto))
print(len(Cozinha))
print(len(Comida))

df = pd.DataFrame(Lista_de_compras)

plot.)
