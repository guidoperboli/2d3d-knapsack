#!/bin/bash
#SBATCH --job-name=gasp_test        # Nome del tuo job di test
#SBATCH --output=run_%j.out         # File dove verrà salvato l'output (%j = ID del Job)
#SBATCH --error=run_%j.err          # File dove verranno salvati gli eventuali errori
#SBATCH --nodes=1                   # FONDAMENTALE: richiede 1 solo nodo (per la memoria condivisa)
#SBATCH --ntasks=1                  # 1 task (il tuo script python principale)
#SBATCH --cpus-per-task=2           # Alloca 2 core per il test
#SBATCH --time=00:15:00             # Tempo massimo (impostato a 15 minuti)
#SBATCH --mem=4G                    # Memoria RAM per il test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guido.perboli@polito.it

# ==========================================
# Configurazione dell'ambiente Python (HPC)
# ==========================================
# 1. Se il tuo HPC usa i moduli per gestire il software:
# module load python/3.10

# 2. Se usi un Virtual Environment sul cluster, decommenta e adattalo:
# source /path/al/tuo/venv/bin/activate
# (Se hai copiato la cartella .venv che abbiamo appena creato anche sul cluster)
# source .venv/bin/activate

# ==========================================
# Avvio script
# ==========================================
echo "Inizio esecuzione job SLURM: $(date)"

# ==========================================
# Configurazione di Java 17 (locale o modulo)
# ==========================================
# Se hai installato Java localmente nella tua Home, scommenta queste righe:
export JAVA_HOME="$HOME/jdk-17.0.10+7-jre"
export PATH="$JAVA_HOME/bin:$PATH"

# Alternativamente, se il cluster mette a disposizione un modulo Java 17:
# module load java/17

# Eseguiamo il run di test veloce (run_demo.py) per verificare l'ambiente Python e Java

source .venv/bin/activate
.venv/bin/python examples/run_demo.py

echo "Fine esecuzione job SLURM: $(date)"
