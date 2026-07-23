#!/bin/bash
#SBATCH --job-name=run_overnight    # Nome del tuo job
#SBATCH --output=run_%j.out         # File dove verrà salvato l'output (%j = ID del Job)
#SBATCH --error=run_%j.err          # File dove verranno salvati gli eventuali errori
#SBATCH --nodes=1                   # FONDAMENTALE: richiede 1 solo nodo (per la memoria condivisa)
#SBATCH --ntasks=1                  # 1 task (il tuo script python principale)
#SBATCH --cpus-per-task=32          # Alloca 32 core per i tuoi 32 workers
#SBATCH --time=24:00:00             # Tempo massimo (qui impostato a 24 ore)
#SBATCH --mem=32G                   # Memoria RAM totale (adattala se il job esplode in RAM)
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

# Eseguiamo il run della "notte di calcolo" con 32 workers
# Assicurati di rinominare o cancellare la cartella "results/" prima
# dell'invio se vuoi ricalcolare tutto da capo!

source .venv/bin/activate
.venv/bin/python examples/run_overnight.py --workers 32 --time 60

echo "Fine esecuzione job SLURM: $(date)"
