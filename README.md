# Hybrid Parallel System

A parallel computing project that demonstrates hybrid parallelism by combining **MPI** (distributed memory), **OpenMP** (shared memory), and **CUDA** (GPU acceleration).

---

## 📌 Project Overview

The system runs **11 MPI processes** , each assigned a dedicated task:

| Rank | Role | Task |
|------|------|------|
| 0 | Master | Distributes data to all workers, collects results |
| 1 | Int Worker | Computes **factorial** of a given number using OpenMP |
| 2 | String Worker | Counts **vowels** in a string using OpenMP |
| 3 | File Worker | Splits file lines into **even/odd** output files using OpenMP sections |
| 4 | Matrix Coordinator | Distributes a **50×50 matrix dot product** across sub-workers |
| 5–9 | Matrix Sub-Workers | Each computes a chunk of the dot product using OpenMP |
| 10  | CUDA Worker | Performs **GPU matrix transpose** using a CUDA kernel |

---

## 🗂️ File Structure

```
hybrid-parallel-system/
├── main.cpp          # MPI + OpenMP logic for all ranks
├── cuda_kernel.cu    # CUDA kernel for matrix transpose
├── input.txt         # Sample input file for the file worker
└── README.md
```

---

## ⚙️ Build Instructions

### Without CUDA (10 processes)

```bash
mpicxx -fopenmp -o parallel_system main.cpp
```

### With CUDA (11 processes)

```bash
# Step 1: Compile the CUDA kernel into an object file
nvcc -c cuda_kernel.cu -o cuda_kernel.o

# Step 2: Compile and link everything together
mpicxx -fopenmp -DUSE_CUDA -o parallel_system main.cpp cuda_kernel.o -L/usr/local/cuda/lib64 -lcudart
```

---

## ▶️ Run Instructions

### Without CUDA

```bash
mpirun -np 10 ./parallel_system
```

### With CUDA

```bash
mpirun -np 11 ./parallel_system
```

> **Note:** Make sure `input.txt` exists in the same directory before running. It will be read by Rank 3 (file worker).

---

## 📥 Input

When running, the master (Rank 0) will prompt for:
1. **An integer** — to compute its factorial
2. **A string** — to count its vowels

The file worker reads from `input.txt` and produces:
- `even_lines.txt` — lines at even indices (0, 2, 4, ...)
- `odd_lines.txt` — lines at odd indices (1, 3, 5, ...)

---

## 📤 Sample Output

```
=== MASTER: sending data to workers ===
[Master] Enter a number for factorial: 10
[Master] Enter a string for vowel count: Hello
[Master] Sent 10! task to Rank 1
[Master] Sent "Hello" to Rank 2
[Master] Sent 12 lines to Rank 3
[Master] Sent 50x50 matrices to Rank 4

[Rank 1] Computing 10! ...
[Rank 1] 10! = 3628800
[Rank 2] Vowels count is 2
[Rank 3] Wrote 6 even lines
[Rank 3] Wrote 6 odd lines
[Rank 4] Distributing to sub-workers...
[Rank 5] Got 10 rows, computing...
...

=== MASTER: collecting results ===
[Master] From Rank 1: 10! = 3628800
[Master] Vowel count in "Hello": 2
[Master] File worker confirmed completion.
[Master] Matrix result: C[0][0]=... C[0][1]=...

=== ALL DONE ===
```

---

## 🔧 Requirements

- **MPI**: OpenMPI or MPICH
- **OpenMP**: Supported by GCC (`-fopenmp`)
- **CUDA** *(optional)*: NVIDIA GPU + CUDA Toolkit

---
