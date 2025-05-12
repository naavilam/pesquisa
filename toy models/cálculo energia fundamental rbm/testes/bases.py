from pyscf import gto, scf, dft
import matplotlib.pyplot as plt
import numpy as np

# Define a molécula de água com base minimal
mol = gto.Mole()
mol.atom = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'
mol.basis = '6-31G'
mol.build()

# Hartree-Fock
mf_hf = scf.RHF(mol)
mf_hf.kernel()
orbitals_hf = mf_hf.mo_coeff

# DFT com B3LYP
mf_dft = dft.RKS(mol)
mf_dft.xc = 'b3lyp'
mf_dft.kernel()
orbitals_dft = mf_dft.mo_coeff

# Orbitais HOMO (último ocupado)
homo_index = mol.nelectron // 2 - 1
homo_hf = orbitals_hf[:, homo_index]
homo_dft = orbitals_dft[:, homo_index]

# Gráfico comparativo
labels = [f"χ{i+1}" for i in range(len(homo_hf))]
x = np.arange(len(homo_hf))

plt.figure(figsize=(10, 5))
bar_width = 0.35
plt.bar(x - bar_width/2, homo_hf, bar_width, label='HF')
plt.bar(x + bar_width/2, homo_dft, bar_width, label='DFT (B3LYP)')

plt.xlabel('Funções de base χ')
plt.ylabel('Coeficientes no orbital HOMO')
plt.title('Comparação dos Orbitais HOMO: Hartree-Fock vs DFT (B3LYP)')
plt.xticks(x, labels)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()