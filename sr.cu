#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <bits/stdc++.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

int threads_per_block = 256;

bool find_b (vector<int>& vec, int val){
    for (int i = 0; i < vec.size(); i++)
    {
        if (vec[i] == val)
            return true;
    }
    return false;
}

__global__ void manhattan_distance_kernel(const float *matrix, const float *input, int matrix_rows, float *distances, int* count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < matrix_rows) {
        if (matrix[tid * 3 + 1] == input[1]) {
            atomicAdd(&distances[int(matrix[tid * 3]) -1], fabs(matrix[tid*3+2] - input[2]));
            atomicAdd(&count[int(matrix[tid * 3]) -1 ], 1);
        }
    }
}

__global__ void euclidian_distance_kernel(const float *matrix, const float *input, int matrix_rows, float *distances, int* count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < matrix_rows) {
        if (matrix[tid * 3 + 1] == input[1]) {
            atomicAdd(&distances[int(matrix[tid * 3]) -1 ], pow((matrix[tid*3+2] - input[2]),2));
            atomicAdd(&count[int(matrix[tid * 3]) -1 ], 1);
        }
    }
}

__global__ void dot_product_kernel(const float *input, const float *matrix, int matrix_rows, float *distances) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < matrix_rows) {
        if (matrix[tid * 3 + 1] == input[1]) {
            atomicAdd(&distances[int(matrix[tid * 3]) -1], (matrix[tid*3+2] * input[2]));
        }
    }
}

__global__ void norm_vector_kernel(const float *matrix, int matrix_rows, float* vector_acc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < matrix_rows) {
        atomicAdd(&vector_acc[int(matrix[tid * 3]) -1], matrix[tid*3+2]*matrix[tid*3+2]);
    }
}

__global__ void varianza_kernel(const float *matrix, float* vector_acc, int matrix_rows, float* mean ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < matrix_rows){
        atomicAdd(&vector_acc[int(matrix[tid * 3]) -1], pow((matrix[tid*3+2]-mean[int(matrix[tid * 3]) -1]), 2));
    } 
        
}

__global__ void covarianza_kernel(const float *matrix, const float *input, int matrix_rows, float *vector_acc, float* mean) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < matrix_rows) {
        if (matrix[tid * 3 + 1] == input[1]) {
            atomicAdd(&vector_acc[int(matrix[tid * 3]) -1], (input[2]-mean[int(input[0])-1])*
                                                            (matrix[tid * 3 + 2]-mean[int(matrix[tid * 3])-1]));
        }
    }
}

__global__ void media_kernel(const float *matrix, int matrix_rows, float* vector_acc, float* vector_n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < matrix_rows) {
        atomicAdd(&vector_acc[int(matrix[tid * 3]) - 1], matrix[tid*3 + 2]);
        atomicAdd(&vector_n[int(matrix[tid * 3]) - 1], 1);
    }
}

void pearson(float *& d_matrix, float *& input, int num_rows, int n_users, int input_rows, int usr_id, float * matrix, int thr, int kn, int k_peliculas, int user2)
{
    int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;
    float* d_vector_acc;
    cudaMalloc((void**)&d_vector_acc, n_users * sizeof(float));
    cudaMemset(d_vector_acc, 0, n_users * sizeof(float));

    float* d_vector_n;
    cudaMalloc((void**)&d_vector_n, n_users * sizeof(float));
    cudaMemset(d_vector_n, 0, n_users * sizeof(float));

    media_kernel<<<blocks_per_grid, threads_per_block>>>(d_matrix, num_rows, d_vector_acc, d_vector_n);
    cudaDeviceSynchronize();
    
    float* vector_acc = new float[n_users];
    float* vector_n = new float[n_users];
    cudaMemcpy(vector_acc, d_vector_acc, n_users * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vector_n, d_vector_n, n_users * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_users; i++){
        vector_acc[i] = vector_acc[i]/vector_n[i];
    }

    cudaMemcpy(d_vector_acc, vector_acc, n_users * sizeof(float), cudaMemcpyHostToDevice);

    float* d_varianza;
    cudaMalloc((void**)&d_varianza, n_users * sizeof(float));
    varianza_kernel<<<blocks_per_grid, threads_per_block>>>(d_matrix, d_varianza, num_rows, d_vector_acc);
    cudaDeviceSynchronize();

    float* varianza = new float[n_users];
    cudaMemcpy(varianza, d_varianza, n_users * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_users; i++){
        varianza[i] = varianza[i]/vector_n[i];
    } 

    float* d_covarianza;
    cudaMalloc((void**)&d_covarianza, n_users * sizeof(float));
    float* d_input;
    cudaMalloc((void **)&d_input, 3 * sizeof(float));
    for (int i = 0; i < input_rows; i++)
    {
        cudaMemcpy(d_input, &input[i*3], 3 * sizeof(float), cudaMemcpyHostToDevice);
        covarianza_kernel<<<blocks_per_grid, threads_per_block>>>(d_matrix, d_input, num_rows, d_covarianza, d_vector_acc);
        cudaDeviceSynchronize();
    }

    float* covarianza = new float[n_users];
    cout<<endl;
    cudaMemcpy(covarianza, d_covarianza, n_users * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_users; i++){
        covarianza[i] = covarianza[i]/vector_n[i];
    }
        
    std::vector<std::pair<float, int>> pares(n_users);
    float tmp;
    for (int i = 0; i < n_users; ++i) {
        tmp = covarianza[i]/sqrt(varianza[usr_id-1]*varianza[i]);
        pares[i] = std::make_pair(tmp, i);
    }

    cout << usr_id <<"   Distancia al usuario: " << pares[user2-1].second + 1 <<" = "<< pares[user2-1].first << endl << endl;

    std::sort(pares.begin(), pares.end());
    cout<<"User ID: "<<usr_id<<endl;
    int tmpkn = kn;
    for (int i = n_users-1; i > n_users - kn - 2; --i) {
        if (pares[i].second + 1 != usr_id){
            if ( tmpkn==0 )
                break;
            tmpkn--;

            std::cout << "   Similitud al usuario: " << pares[i].second + 1 <<" = "<< pares[i].first << std::endl;
        }
    }

    tmpkn = kn;    
    vector<int> peliculas;
    for (int i = n_users-kn-1; i < n_users; ++i){ 
        if (pares[i].second + 1 != usr_id){
            if ( tmpkn==0 )
                break;
            tmpkn--;

            for (int j = 0; j < num_rows; j++)
            {
                if (pares[i].second == matrix[j*3])
                {
                    if (matrix[j*3+2] >= thr && !find_b(peliculas, matrix[j*3+1]))
                        peliculas.push_back(matrix[j*3+1]);
                }
            }
        }
    } 

    cout<<"Peliculas recomendadas"<<endl;
    int k = 0;
    for (int i = peliculas.size()-1; i >= 0 ; --i)
    {
        cout<<peliculas[i]<<" - ";
        if (k >= k_peliculas - 1)
            break;
        
        k++;
    }
    cout<<endl;

    
    cudaFree(d_vector_acc);
    cudaFree(d_vector_n);
    cudaFree(varianza);
    cudaFree(d_input);
    cudaFree(covarianza);
}


void cosine(float *& d_matrix, float *& input, int n_users, int num_rows, int input_rows, int usr_id, float * matrix, int thr, int kn, int k_peliculas, int user2)
{
    int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;

    float *distances = new float[n_users];
    float *vector_acc = new float[n_users];

    float* d_input, * d_distances, * d_vector_acc;
    cudaMalloc((void **)&d_input, 3 * sizeof(float));
    cudaMalloc((void **)&d_distances, n_users * sizeof(float));
    cudaMalloc((void **)&d_vector_acc, n_users * sizeof(float));

    norm_vector_kernel<<<blocks_per_grid, threads_per_block>>>(d_matrix, num_rows, d_vector_acc);
    cudaMemcpy(vector_acc, d_vector_acc, n_users * sizeof(float), cudaMemcpyDeviceToHost);

    

    for (int i = 0; i < input_rows; i++){
        cudaMemcpy(d_input, &input[i*3], 3 * sizeof(float), cudaMemcpyHostToDevice);
        dot_product_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_matrix, num_rows, d_distances);
        cudaDeviceSynchronize();
    }
    
    
    cudaMemcpy(distances, d_distances, n_users * sizeof(float), cudaMemcpyDeviceToHost);
    

    float x_mod;
    for (int i = 0; i < input_rows; i++){
        x_mod += (input[i*3+2]*input[i*3+2]);
    }
    x_mod = sqrt(x_mod);
    
    
    std::vector<std::pair<float, int>> pares(n_users);
    for (int i = 0; i < n_users; ++i) {
        vector_acc[i] = sqrt(vector_acc[i]);
        distances[i] = distances[i]/(vector_acc[i]*x_mod);
        pares[i] = std::make_pair(distances[i], i);
    }

    cout << usr_id <<"   Similitud al usuario: " << pares[user2-1].second + 1 <<" = "<< pares[user2-1].first << endl << endl;

    std::sort(pares.begin(), pares.end());
    cout<<"User ID: "<<usr_id<<endl;
    int tmpkn = kn;
    for (int i = n_users-1; i > n_users - kn - 2; --i) {
        if (pares[i].second + 1 != usr_id){
            if ( tmpkn==0 )
                break;
            tmpkn--;

            std::cout << "   Similitud al usuario: " << pares[i].second + 1 <<" = "<< pares[i].first << std::endl;
        }
    }        

    std::vector<int> user_movies;
    for (int i = 0; i < num_rows; i++)
    { 
        if (matrix[i*3] == usr_id){
            int j = 0;
            while (matrix[(i+j)*3] == usr_id )
            {
                user_movies.push_back(matrix[(i+j)*3 + 1]);
                j++;
            }
            break;
        }
    }
    cout<<endl;
    
    tmpkn = kn;
    vector<int> peliculas;
    for (int i = n_users-kn-1; i < n_users; ++i){ 
        if (pares[i].second + 1 != usr_id){
            if ( tmpkn==0 )
                break;
            tmpkn--;

            for (int j = 0; j < num_rows; j++)
            {
                if (pares[i].second + 1 == matrix[j*3])
                {
                    if (matrix[j*3+2] >= thr && !find_b(peliculas, matrix[j*3+1]) && !find_b(user_movies, matrix[j*3+1]))
                        peliculas.push_back(matrix[j*3+1]);
                }
            }
        }
    } 

    cout<<"Peliculas recomendadas"<<endl;
    int k = 0;
    for (int i = peliculas.size()-1; i >= 0 ; --i)
    {
        cout<<peliculas[i]<<" - ";
        if (k >= k_peliculas - 1)
            break;
        
        k++;
    }
    cout<<endl;
}

void euclidean(float *& d_matrix, float *& input, int num_rows, int n_users, int input_rows, int usr_id, float * matrix, int thr, int kn, int k_peliculas, int user2){
    int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;
    float* distances = new float[n_users];
    float* d_distances;
    int* count = new int[n_users];
    int* d_count;

    cudaMalloc((void **)&d_distances, n_users * sizeof(float));
    cudaMalloc((void **)&d_count, n_users * sizeof(int));
    cudaMemset(d_count, 0, n_users * sizeof(int));

    float* d_input;
    cudaMalloc((void **)&d_input, 3 * sizeof(float));
    for (int i = 0; i < input_rows; i++)
    {
        cudaMemcpy(d_input, &input[i*3], 3 * sizeof(float), cudaMemcpyHostToDevice);
        euclidian_distance_kernel<<<blocks_per_grid, threads_per_block>>>(d_matrix, d_input, num_rows, d_distances, d_count);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(distances, d_distances, n_users * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(count, d_count, n_users * sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<std::pair<float, int>> pares(n_users);
    for (int i = 0; i < n_users; ++i) {
        distances[i] = sqrt(distances[i]);     
        pares[i] = std::make_pair(distances[i], i);
    }

    cout << usr_id <<"   Distancia al usuario: " << pares[user2-1].second + 1 <<" = "<< pares[user2-1].first << endl << endl;
    cout<< "Count " <<count[user2] <<endl;

    std::sort(pares.begin(), pares.end());
    cout<<"User ID: "<<usr_id<<endl;
    int tmpkn = kn;
    for (int i = 0; i < n_users; ++i){ 
        if (pares[i].second + 1 != usr_id && count[pares[i].second] > 0){
            if ( tmpkn==0 )
                break;
            tmpkn--;

            std::cout << "   Distancia al usuario: " << pares[i].second + 1 <<" = "<< pares[i].first << std::endl;    
        }
    }

    std::vector<int> user_movies;
    for (int i = 0; i < num_rows; i++)
    { 
        if (matrix[i*3] == usr_id){
            int j = 0;
            while (matrix[(i+j)*3] == usr_id )
            {
                user_movies.push_back(matrix[(i+j)*3 + 1]);
                j++;
            }
            break;
        }
    }
    cout<<endl;
    
    tmpkn = kn;
    vector<int> peliculas;
    for (int i = 0; i < n_users; ++i){ 
        if (pares[i].second + 1 != usr_id && count[pares[i].second] > 0){
            if ( tmpkn==0 )
                break;
            tmpkn--;

            for (int j = 0; j < num_rows; j++)
            {
                if (pares[i].second + 1 == matrix[j*3])
                {
                    if (matrix[j*3+2] >= thr && !find_b(peliculas, matrix[j*3+1]) && !find_b(user_movies, matrix[j*3+1]))
                        peliculas.push_back(matrix[j*3+1]);
                }
            }     
        }
    }

    cout<<"Peliculas recomendadas"<<endl;
    int k = 0;
    for (int i = peliculas.size()-1; i >= 0 ; --i)
    {
        cout<<peliculas[i]<<" - ";
        if (k >= k_peliculas - 1)
            break;
        
        k++;
    }
    cout<<endl;
    
}

void manhattan(float *& d_matrix, float *& input, int num_rows, int n_users, int input_rows, int usr_id, float * matrix, int thr, int kn, int k_peliculas, int user2){
    int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;
    float* distances = new float[n_users];
    float* d_distances;
    int* count = new int[n_users];
    int* d_count;

    cudaMalloc((void **)&d_distances, n_users * sizeof(float));
    cudaMalloc((void **)&d_count, n_users * sizeof(int));

    float* d_input;
    cudaMalloc((void **)&d_input, 3 * sizeof(float));
    for (int i = 0; i < input_rows; i++)
    {
        cudaMemcpy(d_input, &input[i*3], 3 * sizeof(float), cudaMemcpyHostToDevice);
        manhattan_distance_kernel<<<blocks_per_grid, threads_per_block>>>(d_matrix, d_input, num_rows, d_distances, d_count);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(distances, d_distances, n_users * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(count, d_count, n_users * sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<std::pair<float, int>> pares(n_users);
    for (int i = 0; i < n_users; ++i) 
        pares[i] = std::make_pair(distances[i], i);

    cout << usr_id <<"   Distancia al usuario: " << pares[user2-1].second + 1 <<" = "<< pares[user2-1].first << endl << endl;

    std::sort(pares.begin(), pares.end());
    cout<<"User ID: "<<usr_id<<endl;
    int tmpkn = kn;
    for (int i = 0; i < n_users; ++i){ 
        if (pares[i].second + 1 != usr_id && count[pares[i].second] > 0){
            if ( tmpkn==0 )
                break;
            tmpkn--;

            std::cout << "   Distancia al usuario: " << pares[i].second + 1 <<" = "<< pares[i].first << std::endl;    
        }
    }   
    

    std::vector<int> user_movies;
    for (int i = 0; i < num_rows; i++)
    { 
        if (matrix[i*3] == usr_id){
            int j = 0;
            while (matrix[(i+j)*3] == usr_id )
            {
                user_movies.push_back(matrix[(i+j)*3 + 1]);
                j++;
            }
            break;
        }
    }
    cout<<endl;
    
    tmpkn = kn;
    vector<int> peliculas;
    for (int i = 0; i < n_users; ++i){ 
        if (pares[i].second + 1 != usr_id && count[pares[i].second] > 0){
            if ( tmpkn==0 )
                break;
            tmpkn--;

            for (int j = 0; j < num_rows; j++)
            {
                if (pares[i].second + 1 == matrix[j*3])
                {
                    if (matrix[j*3+2] >= thr && !find_b(peliculas, matrix[j*3+1]) && !find_b(user_movies, matrix[j*3+1]))
                        peliculas.push_back(matrix[j*3+1]);
                }
            }     
        }
    }

    cout<<"Peliculas recomendadas"<<endl;
    int k = 0;
    for (int i = peliculas.size()-1; i >= 0 ; --i)
    {
        cout<<peliculas[i]<<" - ";
        if (k >= k_peliculas - 1)
            break;
        
        k++;
    }
    cout<<endl;

}

int main() {
    auto start = high_resolution_clock::now();
    // Definir el nombre del archivo CSV
    std::string filename = "u1.base";
    //std::string filename = "ratings.csv";
    //std::string filename = "Movie_Ratings2.csv";
    int num_rows = 80001 - 1; // Reemplaza con el número real de filas 33832163 , 442
    int num_cols = 3;  // Reemplaza con el número real de columnas

    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo " << filename << std::endl;
        return 1;
    }

    size_t matrix_size = num_rows * num_cols * sizeof(float);
    float *matrix_flat = new float[matrix_size];

    std::string line;
    std::string token;
    size_t pos = 0, k = 0;

    if (std::getline(file, line)) 
        cout<<"line"<<line<<endl;

    for (int i = 0; i < num_rows; ++i) {
        if (!std::getline(file, line)) {
            std::cerr << "Error: Se esperaban más filas en el archivo." << std::endl;
            return 1;
        }
        while ((pos = line.find('\t')) != std::string::npos) {
            token = line.substr(0, pos); 
            matrix_flat[i*num_cols+k] = stof(token);
            line.erase(0, pos + 1);
            k++;
        }
        k = 0; pos=0;
    }
    
    file.close();
    
    int n_users = matrix_flat[(num_rows-1)*3];
    std::cout << "Usuarios: "<< n_users << std::endl;

    auto stop = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(stop - start);
 
    cout << "Tiempo de lectura: "
         << duration1.count() << " ms" << endl<<endl;


    start = high_resolution_clock::now();
    int input_rows = 0;
    float *d_matrix;
    int userID_input = 1;
    float *input;

    for (int i = 0; i < num_rows; ++i){
        if (matrix_flat[i*3] == userID_input)
        {
            int j = i;
            while (matrix_flat[j*3] == userID_input)
            {
                j++;
                input_rows++;
            }

            input = new float[(j-i)*3];
            for (int k = 0; k < (j-i); k++)
            {
                input[k*3] = matrix_flat[(k+i)*3];
                input[k*3+1] = matrix_flat[(k+i)*3+1];
                input[k*3+2] = matrix_flat[(k+i)*3+2];
            }
            break;
        }
    }

    cout<<matrix_flat[2205*3]<<endl;
    cout<<matrix_flat[2205*3+1]<<endl;

    cudaMalloc((void **)&d_matrix, num_rows * num_cols * sizeof(float));
    cudaMemcpy(d_matrix, matrix_flat, num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice);

    //manhattan(d_matrix, input, num_rows, n_users, input_rows, userID_input, matrix_flat, 4, 10, 10, 1);
    euclidean(d_matrix, input, num_rows, n_users, input_rows, userID_input, matrix_flat, 4, 10, 10, 34);
    //cosine(d_matrix, input, n_users, num_rows, input_rows, userID_input, matrix_flat, 4, 10, 10, 1);
    //pearson(d_matrix,input,num_rows,n_users,input_rows,userID_input, matrix_flat, 4, 10, 10, 1);

    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(stop - start);
    cout <<endl<<endl<< "Tiempo de ejecucion KNN: "
         << duration2.count() << " ms" << endl;
    cout << "TOTAL: "
         << duration1.count() + duration2.count() << " ms" << endl;

    return 0;
}