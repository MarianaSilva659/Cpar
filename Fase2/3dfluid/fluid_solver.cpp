#include "fluid_solver.h"
#include <cmath>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define BLOCKSIZE 14
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define LINEARSOLVERTIMES 20

// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
}

// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float *x) {
  #pragma omp parallel 
  {
  int i, j;

  // Set boundary on faces
  #pragma omp for
  for (j = 1; j <= M; j++) {
    for (i = 1; i <= N; i++) {
      x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
      x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
    }
  }
  #pragma omp for
  for (j = 1; j <= N; j++) {
    for (i = 1; i <= O; i++) {
      x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
      x[IX(M + 1, i, j)] = b == 1 ? -x[IX(M, i, j)] : x[IX(M, i, j)];
    }
  }
  #pragma omp for
  for (j = 1; j <= M; j++) {
    for (i = 1; i <= O; i++) {
      x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
      x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
    }
  }
  }
  // Set corners
  x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
  x[IX(M + 1, 0, 0)] =
      0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
  x[IX(0, N + 1, 0)] =
      0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
  x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                    x[IX(M + 1, N + 1, 1)]);
}

void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c, old_x, change;
    int l = 0;
    const float invC = 1.f / c;
    const float divAC = a / c;

    do {
        max_c = 0.0f;

        // Região paralela compartilhada
        #pragma omp parallel 
        {
            float local_max_c = 0.0f; // Variável local para evitar conflitos

            // Loop para elementos vermelhos
            
            #pragma omp for schedule(static) private(old_x, change)  collapse(2)
            for (int k = 1; k <= O; k++) {
                for (int j = 1; j <= N; j++) {
                    for (int i = 1 + (k + j) % 2; i <= M; i += 2) {
                        int idx = IX(i, j, k);
                        float neighbor_sum = x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)];

                        old_x = x[idx];
                        x[idx] = invC * x0[idx] + divAC * neighbor_sum;

                        change = fabs(x[idx] - old_x);
                        if (change > local_max_c) local_max_c = change;
                    }
                }
            }

            // Loop para elementos pretos
            #pragma omp for schedule(static) private(old_x, change) collapse(2)
            for (int k = 1; k <= O; k++) {
                for (int j = 1; j <= N; j++) {
                    for (int i = 1 + (k + j + 1) % 2; i <= M; i += 2) {
                        int idx = IX(i, j, k);
                       float neighbor_sum = x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)];

                        old_x = x[idx];
                        x[idx] = invC * x0[idx] + divAC * neighbor_sum;

                        change = fabs(x[idx] - old_x);
                        if (change > local_max_c) local_max_c = change;
                    }
                }
            }

            // Redução final para max_c
            #pragma omp critical
            {
                if (local_max_c > max_c) max_c = local_max_c;
            }
        }

        // Ajuste das bordas (não paralelizado)
        set_bnd(M, N, O, b, x);

    } while (max_c > tol && ++l < 20);
}




// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff,
             float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}
// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;
  #pragma omp parallel for collapse(2) schedule(static)
  for (int k = 1; k <= M; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= O; i++) {
        int xCenter = IX(i, j, k);
        float x = i - dtX * u[xCenter];
        float y = j - dtY  * v[xCenter];
        float z = k - dtZ  * w[xCenter];

        // Clamp to grid boundaries
        if (x < 0.5f)
          x = 0.5f;
        if (x > M + 0.5f)
          x = M + 0.5f;
        if (y < 0.5f)
          y = 0.5f;
        if (y > N + 0.5f)
          y = N + 0.5f;
        if (z < 0.5f)
          z = 0.5f;
        if (z > O + 0.5f)
          z = O + 0.5f;

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
  }
  set_bnd(M, N, O, b, d);
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float *u, float *v, float *w, float *p,
             float *div) {
  float max = (1.f / MAX(M, MAX(N, O)))* -0.5f;
  // Primeira parte permanece inalterada
  #pragma omp parallel for collapse(2) schedule(static)
  for (int k = 1; k <= M; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= O; i++) {
        int xCenter = IX(i, j, k);
        div[xCenter] =
            (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
             v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) * max;
        p[xCenter] = 0;
      }
    }
  }

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);
  #pragma omp parallel for collapse(2) schedule(static)
  for (int k = 1; k <= M; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= O; i++) {
        int xCenter = IX(i, j, k);
        u[xCenter] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[xCenter] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[xCenter] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
      }
    }
  }


  // Restante do código
  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}

// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt) {
  add_source(M, N, O, x, x0, dt);
  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {
  add_source(M, N, O, u, u0, dt);
  add_source(M, N, O, v, v0, dt);
  add_source(M, N, O, w, w0, dt);
  SWAP(u0, u);
  diffuse(M, N, O, 1, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse(M, N, O, 2, v, v0, visc, dt);
  SWAP(w0, w);
  diffuse(M, N, O, 3, w, w0, visc, dt);
  project(M, N, O, u, v, w, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  SWAP(w0, w);
  advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project(M, N, O, u, v, w, u0, v0);
}
