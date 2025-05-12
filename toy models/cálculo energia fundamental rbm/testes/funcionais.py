from pyscf import gto, dft
import pandas as pd

# Define a molécula de água com base STO-3G
mol = gto.Mole()
mol.atom = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'
mol.basis = 'sto-3g'
mol.build()

# Lista de funcionais para comparar
functionals = ['lda', 'pbe', 'b3lyp']
energies = {}

# Roda os cálculos para cada funcional
for xc in functionals:
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.kernel()
    energies[xc] = mf.e_tot

# Mostra os resultados
df = pd.DataFrame.from_dict(energies, orient='index', columns=['Energia Total (Hartree)'])
df.index.name = 'Funcional XC'
print(df.sort_values(by='Energia Total (Hartree)'))