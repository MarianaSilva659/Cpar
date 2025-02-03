#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>
#include <cuda.h>
#include <chrono>

#define SIZE 168

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

// Fluid simulation arrays
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;
static float *dens, *dens_prev;

// Function to allocate simulation data
int allocate_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  u = new float[size];
  v = new float[size];
  w = new float[size];
  u_prev = new float[size];
  v_prev = new float[size];
  w_prev = new float[size];
  dens = new float[size];
  dens_prev = new float[size];

  if (!u || !v || !w || !u_prev || !v_prev || !w_prev || !dens || !dens_prev) {
    std::cerr << "Cannot allocate memory" << std::endl;
    return 0;
  }
  return 1;
}

// Function to clear the data (set all to zero)
void clear_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    u[i] = v[i] = w[i] = u_prev[i] = v_prev[i] = w_prev[i] = dens[i] =
        dens_prev[i] = 0.0f;
  }
}

// Free allocated memory
void free_data() {
  delete[] u;
  delete[] v;
  delete[] w;
  delete[] u_prev;
  delete[] v_prev;
  delete[] w_prev;
  delete[] dens;
  delete[] dens_prev;
}

// Apply events (source or force) for the current timestep
__global__ void apply_events_kernel(Event *events, int num_events, float *d_u, float *d_v, float *d_w, float *d_dens, int ijk ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_events){
    Event event = events[idx];
    if (event.type == ADD_SOURCE) {
      d_dens[ijk] = event.density;
    } else if (event.type == APPLY_FORCE) {
      d_u[ijk] = event.force.x;
      d_v[ijk] = event.force.y;
      d_w[ijk] = event.force.z;
    }
  }
}

void apply_events(const std::vector<Event> &events, float *d_dens, float *d_u, float *d_v, float *d_w, int ijk) {

  int size = events.size();
  Event *d_events;
  cudaMalloc((void **)&d_events, size * sizeof(Event));
  cudaMemcpy(d_events, events.data(), size * sizeof(Event), cudaMemcpyHostToDevice);
  int threads_per_block = 256;
  int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

  if (blocks_per_grid > 0)
  apply_events_kernel<<<blocks_per_grid, threads_per_block>>>(d_events, size, d_u, d_v, d_w, d_dens, ijk);
  cudaFree(d_events);
}

// Function to sum the total density
float sum_density() {
  float total_density = 0.0f;
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    total_density += dens[i];
  }
  return total_density;
}

// Simulation loop
void simulate(EventManager &eventManager, int timesteps) {
  
  int size = (M + 2) * (N + 2) * (O + 2);
  float *d_u, *d_uprev, *d_v, *d_vprev, *d_w, *d_wprev,*d_dens, *d_dens_prev;

  cudaMalloc((void**)&d_u, size * sizeof(float));
  cudaMalloc((void**)&d_uprev, size * sizeof(float));
  cudaMalloc((void**)&d_vprev, size * sizeof(float));
  cudaMalloc((void**)&d_v, size * sizeof(float));
  cudaMalloc((void**)&d_w, size * sizeof(float));
  cudaMalloc((void**)&d_wprev, size * sizeof(float));
  cudaMalloc((void**)&d_dens, size * sizeof(float));
  cudaMalloc((void**)&d_dens_prev, size * sizeof(float));


  cudaMemcpy(d_u, u, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_uprev,u_prev, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vprev, v_prev, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wprev, w_prev, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dens, dens, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dens_prev, dens_prev, size * sizeof(float), cudaMemcpyHostToDevice);
  int i = M >> 1, j = N >> 1, k = O >> 1;
  const int ijk = IX(i,j,k);
  for (int t = 0; t < timesteps; t++) {
    // Get the events for the current timestep
    std::vector<Event> events = eventManager.get_events_at_timestamp(t);

    // Apply events to the simulation
    apply_events(events, d_dens, d_u, d_v, d_w, ijk);

    // Perform the simulation steps
    vel_step(M, N, O, d_u, d_v, d_w, d_uprev, d_vprev, d_wprev, visc, dt);
    dens_step(M, N, O, d_dens, d_dens_prev, d_u, d_v, d_w, diff, dt);

  }
  cudaMemcpy(dens, d_dens, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_u);
  cudaFree(d_uprev);
  cudaFree(d_v);
  cudaFree(d_vprev);
  cudaFree(d_w);
  cudaFree(d_wprev);
  cudaFree(d_dens);
  cudaFree(d_dens_prev);
}

int main() {
  // Initialize EventManager
  EventManager eventManager;
  eventManager.read_events("events.txt");

  // Get the total number of timesteps from the event file
  int timesteps = eventManager.get_total_timesteps();

  // Allocate and clear data
  if (!allocate_data())
    return -1;
  clear_data();

  // Run simulation with events
  simulate(eventManager, timesteps);

  // Print total density at the end of simulation
  float total_density = sum_density();
  std::cout << "Total density after " << timesteps
            << " timesteps: " << total_density << std::endl;

  // Free memory
  free_data();

  return 0;
}