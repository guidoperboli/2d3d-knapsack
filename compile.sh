#!/bin/bash

# Spostati nella directory del progetto Java
cd gasp_java || { echo "Errore: cartella gasp_java non trovata"; exit 1; }

# Cerca Maven nel PATH
if command -v mvn &> /dev/null; then
    echo "Uso Maven di sistema..."
    mvn clean package
# Altrimenti usa il Maven scaricato localmente (se presente)
elif [ -f "./tmp_maven/apache-maven-3.9.8/bin/mvn" ]; then
    echo "Uso Maven locale (tmp_maven)..."
    ./tmp_maven/apache-maven-3.9.8/bin/mvn clean package
else
    echo "Errore: Maven non trovato né nel PATH né in tmp_maven."
    echo "Installa Maven sul server oppure scaricalo in gasp_java/tmp_maven."
    exit 1
fi
