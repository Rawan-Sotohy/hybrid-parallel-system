#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
using namespace std;

// rank assignments
const int MASTER = 0;
const int INT_WORKER = 1;
const int STR_WORKER = 2;
const int FILE_WORKER = 3;
const int MATRIX_COORD = 4;
// ranks 5-9 are matrix sub-workers
const int CUDA_WORKER = 10;

// tags
const int TAG_INT = 10;
const int TAG_STR = 20;
const int TAG_FILE = 30;
const int TAG_MATRIX = 40;
const int TAG_CUDA = 50;
const int TAG_RESULT = 100;

const int MATRIX_SIZE = 50;

#ifdef USE_CUDA
extern void cuda_matrix_transpose(double* in, double* out, int n);
#endif

// rank 1: int worker
void run_int_worker() {
    MPI_Status status;
    int n;
    MPI_Recv(&n, 1, MPI_INT, MASTER, TAG_INT, MPI_COMM_WORLD, &status);

    cout << "[Rank 1] Computing " << n << "! ...\n";
    long long result = 1;
#pragma omp parallel for reduction(*:result)
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    cout << "[Rank 1] " << n << "! = " << result << "\n";

    MPI_Send(&result, 1, MPI_LONG_LONG_INT, MASTER, TAG_RESULT, MPI_COMM_WORLD);
}

// rank 2: string worker
void run_str_worker() {
    MPI_Status status;

    int str_len;
    MPI_Recv(&str_len, 1, MPI_INT, MASTER, TAG_STR, MPI_COMM_WORLD, &status);

    vector<char> buf(str_len);
    MPI_Recv(buf.data(), str_len, MPI_CHAR, MASTER, TAG_STR + 1, MPI_COMM_WORLD, &status);

    int count = 0;
    string vowels = "aeiouAEIOU";

    // Loop directly through the vector
#pragma omp parallel for reduction(+:count)
    for (int i = 0; i < str_len; i++) {
        // buf[i] works exactly like s[i]
        if (vowels.find(buf[i]) != string::npos)
            count++;
    }

    cout << "[Rank 2] Vowels count is "<< count << "\n";
    MPI_Send(&count, 1, MPI_INT, MASTER, TAG_RESULT, MPI_COMM_WORLD);
}

// rank 3: file worker 
void run_file_worker() {
    MPI_Status status;

    int num_lines;
    MPI_Recv(&num_lines, 1, MPI_INT, MASTER, TAG_FILE, MPI_COMM_WORLD, &status);

    vector<string> lines(num_lines);
    for (int i = 0; i < num_lines; i++) {
        int len;
        MPI_Recv(&len, 1, MPI_INT, MASTER, TAG_FILE + 1, MPI_COMM_WORLD, &status);
        vector<char> buf(len + 1, '\0');
        MPI_Recv(buf.data(), len, MPI_CHAR, MASTER, TAG_FILE + 2, MPI_COMM_WORLD, &status);
        lines[i] = string(buf.data(), len);
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            ofstream feven("even_lines.txt");
            for (int i = 1; i < num_lines; i += 2)
                feven << lines[i] << "\n";
            cout << "[Rank 3] Wrote " << num_lines / 2 << " even lines\n";
        }

#pragma omp section
        {
            ofstream fodd("odd_lines.txt");
            for (int i = 0; i < num_lines; i += 2)
                fodd << lines[i] << "\n";
            cout << "[Rank 3] Wrote " << (num_lines + 1) / 2 << " odd lines\n";
        }
    }
    int done = 1;
    MPI_Send(&done, 1, MPI_INT, MASTER, TAG_RESULT, MPI_COMM_WORLD);
    cout << "[Rank 3] Finished writing all files.\n";
}
/*
void run_file_worker() {
    MPI_Status status;

    // receive filename from master
    int fname_len;
    MPI_Recv(&fname_len, 1, MPI_INT, MASTER, TAG_FILE, MPI_COMM_WORLD, &status);
    vector<char> buf(fname_len + 1, '\0');
    MPI_Recv(buf.data(), fname_len, MPI_CHAR, MASTER, TAG_FILE + 1, MPI_COMM_WORLD, &status);
    string filename(buf.data(), fname_len);

    ifstream fin(filename);
    vector<string> lines;
    string line;
    while (getline(fin, line)) lines.push_back(line);
    fin.close();

    int num_lines = lines.size();
    cout << "[Rank 3] Read " << num_lines << " lines from " << filename << "\n";

#pragma omp parallel sections
    {
#pragma omp section
        {
            ofstream feven("even_lines.txt");
            for (int i = 1; i < num_lines; i += 2)
                feven << lines[i] << "\n";
            cout << "[Rank 3] Wrote " << num_lines / 2 << " even lines\n";
        }

#pragma omp section
        {
            ofstream fodd("odd_lines.txt");
            for (int i = 0; i < num_lines; i += 2)
                fodd << lines[i] << "\n";
            cout << "[Rank 3] Wrote " << (num_lines + 1) / 2 << " odd lines\n";
        }
    }

    int done = 1;
    MPI_Send(&done, 1, MPI_INT, MASTER, TAG_RESULT, MPI_COMM_WORLD);
    cout << "[Rank 3] Finished writing all files.\n";
}
*/

// rank 4: matrix coordinator
void run_matrix_coordinator() {
    MPI_Status status;
    const int N = MATRIX_SIZE;
    const int NUM_HELPERS = 5;
    const int FIRST_HELPER = 5;

    vector<double> A(N * N), B(N * N), C(N * N, 0.0);

    MPI_Recv(A.data(), N * N, MPI_DOUBLE, MASTER, TAG_MATRIX, MPI_COMM_WORLD, &status);
    MPI_Recv(B.data(), N * N, MPI_DOUBLE, MASTER, TAG_MATRIX + 1, MPI_COMM_WORLD, &status);

    cout << "[Rank 4] Distributing to sub-workers...\n";

    // everyone needs B for the dot product, split rows of A evenly
    int rows_per_worker = N / NUM_HELPERS; //50 / 5 = 10
    for (int w = 0; w < NUM_HELPERS; w++) {
        int start_row = w * rows_per_worker;

        MPI_Send(B.data(), N * N, MPI_DOUBLE, FIRST_HELPER + w, TAG_MATRIX + 1, MPI_COMM_WORLD);
        MPI_Send(&rows_per_worker, 1, MPI_INT, FIRST_HELPER + w, TAG_MATRIX + 2, MPI_COMM_WORLD);
        MPI_Send(&A[start_row * N], rows_per_worker * N, MPI_DOUBLE, FIRST_HELPER + w, TAG_MATRIX + 3, MPI_COMM_WORLD);
    }

    // collect results in order
    for (int w = 0; w < NUM_HELPERS; w++) {
        int start_row = w * rows_per_worker;
        MPI_Recv(&C[start_row * N], rows_per_worker * N, MPI_DOUBLE, FIRST_HELPER + w, TAG_RESULT, MPI_COMM_WORLD, &status);
    }

    cout << "[Rank 4] Done, sending result to master\n";
    MPI_Send(C.data(), N * N, MPI_DOUBLE, MASTER, TAG_RESULT, MPI_COMM_WORLD);
}

//ranks 5-9: matrix sub-workers 
// each gets a chunk of rows from A and the full B
void run_matrix_worker(int rank) {
    MPI_Status status;
    const int N = MATRIX_SIZE;

    vector<double> B(N * N);
    MPI_Recv(B.data(), N * N, MPI_DOUBLE, MATRIX_COORD, TAG_MATRIX + 1, MPI_COMM_WORLD, &status);

    int row_count;
    MPI_Recv(&row_count, 1, MPI_INT, MATRIX_COORD, TAG_MATRIX + 2, MPI_COMM_WORLD, &status);

    vector<double> A_chunk(row_count * N);
    MPI_Recv(A_chunk.data(), row_count * N, MPI_DOUBLE, MATRIX_COORD, TAG_MATRIX + 3, MPI_COMM_WORLD, &status);

    cout << "[Rank " << rank << "] Got " << row_count << " rows, computing...\n";

    vector<double> C_chunk(row_count * N, 0.0);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < row_count; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A_chunk[i * N + k] * B[k * N + j];
            }
            C_chunk[i * N + j] = sum;
        }
    }

    cout << "[Rank " << rank << "] Done, sending back\n";
    MPI_Send(C_chunk.data(), row_count * N, MPI_DOUBLE, MATRIX_COORD, TAG_RESULT, MPI_COMM_WORLD);
}

// rank 10: CUDA worker
void run_cuda_worker() {
#ifdef USE_CUDA
    MPI_Status status;
    const int N = MATRIX_SIZE;

    vector<double> mat(N * N), result(N * N);
    MPI_Recv(mat.data(), N * N, MPI_DOUBLE, MASTER, TAG_CUDA, MPI_COMM_WORLD, &status);

    cout << "[Rank 10] Running GPU transpose...\n";
    cuda_matrix_transpose(mat.data(), result.data(), N);
    cout << "[Rank 10] Done\n";

    MPI_Send(result.data(), N * N, MPI_DOUBLE, MASTER, TAG_RESULT, MPI_COMM_WORLD);
#else
    cerr << "[Rank 10] ERROR: compiled without USE_CUDA\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
}

// rank 0: master 
void run_master(int size) {
    MPI_Status status;
    bool has_cuda = (size == 11);

    cout << "\n=== MASTER: sending data to workers ===\n";

    // 1. send to int worker - compute 10!
    int n;
    cout << "[Master] Enter a number for factorial: ";
    cin >> n;
    MPI_Send(&n, 1, MPI_INT, INT_WORKER, TAG_INT, MPI_COMM_WORLD);
    cout << "[Master] Sent "<<n<<"! task to Rank 1\n";

    // 2. send to string worker - count vowels
    string test_str;
    cout << "[Master] Enter a string for vowel count: ";
    cin >> test_str;
    int str_len = test_str.size();
    MPI_Send(&str_len, 1, MPI_INT, STR_WORKER, TAG_STR, MPI_COMM_WORLD);
    MPI_Send(test_str.c_str(), str_len, MPI_CHAR, STR_WORKER, TAG_STR + 1, MPI_COMM_WORLD);
    cout << "[Master] Sent \"" << test_str << "\" to Rank 2\n";
    
    // 3. send to file worker
    ifstream fin("input.txt");
    vector<string> file_lines;
    string line;
    while (getline(fin, line)) file_lines.push_back(line);
    fin.close();

    int num_lines = file_lines.size();
    MPI_Send(&num_lines, 1, MPI_INT, FILE_WORKER, TAG_FILE, MPI_COMM_WORLD);
    for (string l : file_lines) {
        int len = l.size();
        MPI_Send(&len, 1, MPI_INT, FILE_WORKER, TAG_FILE + 1, MPI_COMM_WORLD);
        MPI_Send(l.c_str(), len, MPI_CHAR, FILE_WORKER, TAG_FILE + 2, MPI_COMM_WORLD);
    }
    cout << "[Master] Sent " << num_lines << " lines to Rank 3\n";

    /*
    string filename = "input.txt";
    int fname_len = filename.size();
    MPI_Send(&fname_len, 1, MPI_INT, FILE_WORKER, TAG_FILE, MPI_COMM_WORLD);
    MPI_Send(filename.c_str(), fname_len, MPI_CHAR, FILE_WORKER, TAG_FILE + 1, MPI_COMM_WORLD);
    cout << "[Master] Sent filename to Rank 3\n";
    */

    // 4. send matrices to coordinator
    const int N = MATRIX_SIZE;
    vector<double> A(N * N), B(N * N);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)(i % 10 + 1);
        B[i] = (double)((i * 2) % 7 + 1);
    }
    MPI_Send(A.data(), N * N, MPI_DOUBLE, MATRIX_COORD, TAG_MATRIX, MPI_COMM_WORLD);
    MPI_Send(B.data(), N * N, MPI_DOUBLE, MATRIX_COORD, TAG_MATRIX + 1, MPI_COMM_WORLD);
    cout << "[Master] Sent 50x50 matrices to Rank 4\n";

    // 5. CUDA bonus
    if (has_cuda) {
        MPI_Send(A.data(), N * N, MPI_DOUBLE, CUDA_WORKER, TAG_CUDA, MPI_COMM_WORLD);
        cout << "[Master] Sent matrix to Rank 10 (CUDA)\n";
    }

    // collect all results
    cout << "\n=== MASTER: collecting results ===\n";

    long long int_result;
    MPI_Recv(&int_result, 1, MPI_LONG_LONG_INT, INT_WORKER, TAG_RESULT, MPI_COMM_WORLD, &status);
    cout << "[Master] From Rank 1: "<<n<<"!= " << int_result << "\n";

    int str_result;
    MPI_Recv(&str_result, 1, MPI_INT, STR_WORKER, TAG_RESULT, MPI_COMM_WORLD, &status);
    cout << "[Master] Vowel count in \"" << test_str << "\": " << str_result << "\n";

    int file_worker_done;
    MPI_Recv(&file_worker_done, 1, MPI_INT, FILE_WORKER, TAG_RESULT, MPI_COMM_WORLD, &status);
    cout << "[Master] File worker confirmed completion.\n";

    vector<double> C(N * N);
    MPI_Recv(C.data(), N * N, MPI_DOUBLE, MATRIX_COORD, TAG_RESULT, MPI_COMM_WORLD, &status);
    cout << "[Master] Matrix result: C[0][0]=" << C[0] << " C[0][1]=" << C[1] << "\n";

    if (has_cuda) {
        vector<double> gpu_result(N * N);
        MPI_Recv(gpu_result.data(), N * N, MPI_DOUBLE, CUDA_WORKER, TAG_RESULT, MPI_COMM_WORLD, &status);
        cout << "[Master] GPU transpose: [0][1]=" << gpu_result[N] << " (was " << A[N] << ")\n"; //index = (row * n) + col
    }

    cout << "\n=== ALL DONE ===\n";
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 10 && size != 11) {
        if (rank == 0)
            cerr << "Run with -np 10 or -np 11 (with CUDA)\n";
        MPI_Finalize();
        return 1;
    }

    switch (rank) {
    case 0:  run_master(size);         break;
    case 1:  run_int_worker();         break;
    case 2:  run_str_worker();         break;
    case 3:  run_file_worker();        break;
    case 4:  run_matrix_coordinator(); break;
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:  run_matrix_worker(rank);  break;
    case 10: run_cuda_worker();        break;
    }

    MPI_Finalize();
    return 0;
}