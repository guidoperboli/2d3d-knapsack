#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"

# Compile and package the Java project using Maven
echo "Compiling the Java project with Maven..."
mvn clean package

echo "Compilation finished. Check the target/ directory for the output."
