
#include "fluid_solver.h"
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <cuda.h>
#include <chrono>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Add sources (density or velocity)
__global__ void add_source_kernel(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += dt * s[idx];
    }
}

__global__ void set_bnd_kernel(int M, int N, int O, int b, float *x) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int idy = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int idz = threadIdx.z + blockIdx.z * blockDim.z + 1;

    if (idx == 0 || idx == M + 1 || idy == 0 || idy == N + 1 || idz == 0 || idz == O + 1) {
        int boundary_idx = (idx == 0) ? 1 : (idx == M + 1) ? M : idx;
        int boundary_idy = (idy == 0) ? 1 : (idy == N + 1) ? N : idy;
        int boundary_idz = (idz == 0) ? 1 : (idz == O + 1) ? O : idz;

        if (idx == 0 || idx == M + 1) {
            x[IX(idx, idy, idz)] = (b == 1) ? -x[IX(boundary_idx, idy, idz)] : x[IX(boundary_idx, idy, idz)];
        }
        if (idy == 0 || idy == N + 1) {
            x[IX(idx, idy, idz)] = (b == 2) ? -x[IX(idx, boundary_idy, idz)] : x[IX(idx, boundary_idy, idz)];
        }
        if (idz == 0 || idz == O + 1) {
            x[IX(idx, idy, idz)] = (b == 3) ? -x[IX(idx, idy, boundary_idz)] : x[IX(idx, idy, boundary_idz)];
        }

        if ((idx == 0 || idx == M + 1) && (idy == 0 || idy == N + 1) && idz == 0) {
            x[IX(idx, idy, idz)] = 0.33f * (x[IX(boundary_idx, boundary_idy, boundary_idz)] +
                                            x[IX(boundary_idx + 1, boundary_idy, boundary_idz)] +
                                            x[IX(boundary_idx, boundary_idy + 1, boundary_idz)]);
        }
    }
}

__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*) address;
    int old = __float_as_int(*address);
    int new_val = __float_as_int(val);

    while (old < new_val) {
        int expected = old;
        old = atomicCAS(address_as_int, expected, new_val);
    }
}

// red-black solver with convergence check
__global__ void lin_solve_kernel_3d(int M, int N, int O, float *x, const float *x0, float a, float c, int parity, float *max_change) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int totalElements = M * N * O;
    if (idx >= totalElements) return;

    int k = idx / (M * N) + 1;
    int j = (idx % (M * N)) / M + 1;
    int i = idx % M + 1;

    if ((i + j + k) % 2 != parity) return;

    float old_x = x[IX(i, j, k)];
    x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                      a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                           x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                           x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;

    float change = fabs(x[IX(i, j, k)] - old_x);
    atomicMaxFloat(max_change, change);
}

void lin_solve_gpu(int M, int N, int O, int b, float *d_x, float *d_x0, float a, float c) {
    const float tol = 1e-7;
    const int max_iter = 20;
    int l = 0;
    float max_c = 0.0f;
    float *d_max_change;

    cudaMalloc(&d_max_change, sizeof(float));

    int blockSize = 256;
    int totalElements = M * N * O;
    int gridSize = (totalElements + blockSize - 1) / blockSize;

    do {
        max_c = 0.0f;
        cudaMemcpy(d_max_change, &max_c, sizeof(float), cudaMemcpyHostToDevice);

        lin_solve_kernel_3d<<<gridSize, blockSize>>>(M, N, O, d_x, d_x0, a, c, 0, d_max_change);
        cudaDeviceSynchronize();

        lin_solve_kernel_3d<<<gridSize, blockSize>>>(M, N, O, d_x, d_x0, a, c, 1, d_max_change);
        cudaDeviceSynchronize();

        cudaMemcpy(&max_c, d_max_change, sizeof(float), cudaMemcpyDeviceToHost);

        set_bnd_kernel<<<gridSize, blockSize>>>(M, N, O, b, d_x);
        cudaDeviceSynchronize();

    } while (++l < max_iter && max_c > tol);

    cudaFree(d_max_change);
}


// Diffusion step (uses implicit methodd
void diffuse(int M, int N, int O, int b, float *d_x, float *d_x0, float diff,
             float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve_gpu(M, N, O, b, d_x, d_x0, a, 1 + 6 * a);
}

__global__ void advect_kernel(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    
    int i = idx % O + 1; 
    int j = (idx / O) % N + 1;  
    int k = idx / (O * N) + 1;

    if (i <= O && j <= N && k <= M) {
        float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

        int xCenter = IX(i, j, k);
        float x = i - dtX * u[xCenter];
        float y = j - dtY * v[xCenter];
        float z = k - dtZ * w[xCenter];

        x = fmaxf(0.5f, fminf(x, M + 0.5f));
        y = fmaxf(0.5f, fminf(y, N + 0.5f));
        z = fmaxf(0.5f, fminf(z, O + 0.5f));

        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1;
        int k0 = (int)z, k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1;
        float u1 = z - k0, u0 = 1 - u1;

        d[xCenter] =
            s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                  t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
            s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                  t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
    }
}

void advect(int M, int N, int O, int b, float *d_d, float *d_d0, float *d_u, float *d_v, float *d_w, float dt) {
    int blockSize = 256; 
    int size = (M + 2) * (N + 2) * (O + 2);  
    int gridSize = (size + blockSize - 1) / blockSize; 
    
    advect_kernel<<<gridSize, blockSize>>>(M, N, O, b, d_d, d_d0, d_u, d_v, d_w, dt);
    cudaDeviceSynchronize();

    set_bnd_kernel<<<gridSize, blockSize>>>(M, N, O, b, d_d);
    cudaDeviceSynchronize();
}


__global__ void project2_kernel(int M, int N, int O, float *u, float *v, float *w, float *p, float *div, float max) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int totalElements = M * N * O;

    if (idx < totalElements) {
        int k = idx / (M * N) + 1;
        int j = (idx % (M * N)) / M + 1;
        int i = idx % M + 1;

        int xCenter = IX(i, j, k);
        
        div[xCenter] = (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
                        v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
                        w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) * max;
        p[xCenter] = 0;
    }
}

__global__ void project3_kernel(int M, int N, int O, float *u, float *v, float *w, float *p) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int totalElements = M * N * O;

    if (idx < totalElements) {
        int k = idx / (M * N) + 1;
        int j = (idx % (M * N)) / M + 1;
        int i = idx % M + 1;

        int xCenter = IX(i, j, k);
        
        u[xCenter] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[xCenter] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[xCenter] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
    }
}

void project(int M, int N, int O, float *d_u, float *d_v, float *d_w, float *d_p, float *d_div) {
    float max = (1.f / MAX(M, MAX(N, O))) * -0.5f;
    int blockSize = 256; 
    int size = (M + 2) * (N + 2) * (O + 2);  
    int gridSize = (size + blockSize - 1) / blockSize; 

    project2_kernel<<<gridSize, blockSize>>>(M, N, O, d_u, d_v, d_w, d_p, d_div, max);
    cudaDeviceSynchronize();

    set_bnd_kernel<<<gridSize, blockSize>>>(M, N, O, 0, d_div);
    set_bnd_kernel<<<gridSize, blockSize>>>(M, N, O, 0, d_p);
    cudaDeviceSynchronize();

    lin_solve_gpu(M, N, O, 0, d_p, d_div, 1, 6);

    project3_kernel<<<gridSize, blockSize>>>(M, N, O, d_u, d_v, d_w, d_p);
    cudaDeviceSynchronize();
    
    set_bnd_kernel<<<gridSize, blockSize>>>(M, N, O, 1, d_u);
    set_bnd_kernel<<<gridSize, blockSize>>>(M, N, O, 2, d_v);
    set_bnd_kernel<<<gridSize, blockSize>>>(M, N, O, 3, d_w);
    cudaDeviceSynchronize();
}


// Step function for density
void dens_step(int M, int N, int O, float *d_x, float *d_x0, float *d_u, float *d_v,
               float *d_w, float diff, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  int blockSize = 512;
  int numBlocks = (size + blockSize - 1) / blockSize;

  add_source_kernel<<<numBlocks, blockSize>>>(M, N, O, d_x, d_x0, dt);
  cudaDeviceSynchronize();

  diffuse(M, N, O, 0, d_x0, d_x, diff, dt);
  advect(M, N, O, 0, d_x, d_x0, d_u, d_v, d_w, dt);

}

// Step function for velocity
void vel_step(int M, int N, int O, float *d_u, float *d_v, float *d_w, float *d_u0,
              float *d_v0, float *d_w0, float visc, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  int blockSize = 512;
  int numBlocks = (size + blockSize - 1) / blockSize;
  
  add_source_kernel<<<numBlocks, blockSize>>>(M, N, O, d_u, d_u0, dt);
  cudaDeviceSynchronize();
  add_source_kernel<<<numBlocks, blockSize>>>(M, N, O, d_v, d_v0, dt);
  cudaDeviceSynchronize();
  add_source_kernel<<<numBlocks, blockSize>>>(M, N, O, d_w, d_w0, dt);
  cudaDeviceSynchronize();

  diffuse(M, N, O, 1, d_u0, d_u, visc, dt);
  diffuse(M, N, O, 2, d_v0, d_v, visc, dt);
  diffuse(M, N, O, 3, d_w0, d_w, visc, dt);

  project(M, N, O, d_u0, d_v0, d_w0, d_u, d_v);

  advect(M, N, O, 1, d_u, d_u0, d_u0, d_v0, d_w0, dt);
  advect(M, N, O, 2, d_v, d_v0, d_u0, d_v0, d_w0, dt);
  advect(M, N, O, 3, d_w, d_w0, d_u0, d_v0, d_w0, dt);

  project(M, N, O, d_u, d_v, d_w, d_u0, d_v0);
}