#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

// Tamaño del array
#define N 20100  // probar con varios

// Kernel de reducción en paralelo
__global__ void reductionKernel(int *d_input, int *d_output, int n) {
    extern __shared__ int sdata[];  // memoria compartida dinámica

    int tid = threadIdx.x;           // índice del hilo
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Cargar en memoria compartida
    if (index < n) {
        sdata[tid] = d_input[index];
    } else {
        sdata[tid] = 0;  // si el índice está fuera del rango, se pone en 0
    }
    __syncthreads();

    // Reducción en paralelo: dividir el bloque a la mitad en cada paso
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // Sincronizar hilos en cada paso
    }

    // El resultado final queda en sdata[0] de cada bloque
    if (tid == 0) {
        d_output[blockIdx.x] = sdata[0];
    }
}

// Función de reducción en la CPU para verificar el resultado
int reductionCPU(int *array, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += array[i];
    }
    return sum;
}

int main() {
    int *h_input, *d_input, *d_output;
    int *h_output;  // Array de salida en la CPU

    int numBytes = N * sizeof(int);
    int blockSize = 256;  // Tamaño de bloque
    int gridSize = (N + blockSize - 1) / blockSize;  // Número de bloques

    // Asignar memoria en la CPU
    h_input = (int *)malloc(numBytes);
    h_output = (int *)malloc(gridSize * sizeof(int));

    // Asignar memoria en la GPU
    cudaMalloc((void **)&d_input, numBytes);
    cudaMalloc((void **)&d_output, gridSize * sizeof(int));

    // Inicializar el array con valores aleatorios
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() % 100;
    }

    // Copiar array a la GPU
    cudaMemcpy(d_input, h_input, numBytes, cudaMemcpyHostToDevice);

    // Medir tiempo de ejecución
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);  // Comenzar a medir el tiempo

    // Ejecutar el kernel
    reductionKernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, N);

    // Esperar a que el kernel termine
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);   // Detener la medición de tiempo
    cudaEventSynchronize(stop);

    // Copiar el resultado de vuelta a la CPU
    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Reducción final en la CPU si se necesita (si es que hay maas de un bloque)
    int gpu_sum = 0;
    for (int i = 0; i < gridSize; i++) {
        gpu_sum += h_output[i];
    }

    // Validacion con la suma secuencial
    int cpu_sum = reductionCPU(h_input, N);
    printf("Suma en CPU: %d\n", cpu_sum);
    printf("Suma en GPU: %d\n", gpu_sum);

    // Verificar la validez del resultado
    if (cpu_sum == gpu_sum) {
        printf("Resultado correcto\n");
    } else {
        printf("Resultado incorrecto\n");
    }

    // Medir el tiempo transcurrido
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Tiempo de ejecución del kernel: %.4f ms\n", elapsedTime);

    // Liberar memoria
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
