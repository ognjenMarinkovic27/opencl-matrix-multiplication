__kernel void multiply_matrices(const int m, const int n, const int k, __global const float* A, __global const float* B, __global float* C) {
    
    const int row = get_global_id(0);
    const int column = get_global_id(1);

    float result = 0.0f;
    for(int i = 0; i < k; i++) {
        result += A[row*k + i] * B[n*i + column];
    }

    C[row * n + column] = result;

}