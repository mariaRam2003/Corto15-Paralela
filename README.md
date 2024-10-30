# Corto #15
# Maria Marta Ramirez Gil

## Instrucciones
**Note** Se recomienda usar WSL

1. Instalar CUDA toolkit en la terminal
```bash
apt install nvidia-cuda-toolkit
```
2. Compilar el codigo
```bash
nvcc -o reduction reduction.cu
```
3. Ejecutar el código
```bash
./reduction
```