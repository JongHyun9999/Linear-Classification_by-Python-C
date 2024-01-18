#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


// 하이퍼 파라미터 선언
#define input_size 256
#define hidden_size1 96
#define hidden_size2 48
#define output_size 7
#define sample_num 105

#define learning_rate 0.001
#define epochs 50


// 매트릭스 조작 함수들 =========================================
// 2차원 배열을 동적으로 할당하고 초기화하는 함수
double** createMatrix(int row, int col) {
    double** matrix = (double**)malloc(row * sizeof(double*));
    for (int i = 0; i < row; i++) {
        matrix[i] = (double*)malloc(col * sizeof(double));
        for (int j = 0; j < col; j++) {
            //matrix[i][j] = i * col + j;
            matrix[i][j] = (2.0 * (double)rand() / RAND_MAX - 1.0) / 2.0;
        }
    }
    return matrix;
}

// 2차원 배열을 동적으로 할당하는데, 그 값을 0으로 초기화.
double** createMatrix0(int row, int col) {
    double** matrix = (double**)malloc(row * sizeof(double*));
    for (int i = 0; i < row; i++) {
        matrix[i] = (double*)malloc(col * sizeof(double));
        for (int j = 0; j < col; j++) {
            matrix[i][j] = 0.0;
        }
    }
    return matrix;
}

double** transposeMatrix(int row, int col, double** matrix) {
    double** result = createMatrix(col, row);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

// 2차원 배열을 출력하는 함수
void print(int row, int col, double** matrix) {
    // 7 X 105
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            printf("%f\t", matrix[j][i]);
        }
        printf("\n");
    }
    printf("======================\n");
}

// 2차원 배열을 해제하는 함수
void freeMatrix(int row, double** matrix) {
    for (int i = 0; i < row; i++) {
        free(matrix[i]);
    }
    free(matrix);
    //printf("해제 완료\n");
}


// 행렬 곱셈 함수
double** matrixDotProduct(double** A, double** B, int m, int n, int p) {
    double** new_matrix = createMatrix0(m, p);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                new_matrix[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return new_matrix;
}

// 매트릭스 element-wise 뺄셈
double** subtractMatrix(double** A, double** B, int rows, int cols) {
    double** new_matrix = createMatrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            new_matrix[i][j] = A[i][j] - B[i][j];
        }
    }
    return new_matrix;
}

// 매트릭스 element-wise 곱셈
double** multiplyMatrix(double** A, double** B, int rows, int cols) {
    double** new_matrix = createMatrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            new_matrix[i][j] = A[i][j] * B[i][j];
        }
    }
    return new_matrix;
}


// utility 레이어 구현 ===============================================
double** ReLU(int rows, int cols, double** matrix) {
    double** new_matrix = createMatrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            new_matrix[i][j] = (matrix[i][j] > 0) ? matrix[i][j] : 0;
        }
    }
    return new_matrix;
}

double** ReLU_derivative(int rows, int cols, double** matrix) {
    double** new_matrix = createMatrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            new_matrix[i][j] = (matrix[i][j] > 0) ? 1 : 0;
        }
    }
    return new_matrix;
}

double** Sigmoid(int rows, int cols, double** matrix) {
    double** new_matrix = createMatrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix[i][j] > 0.0) {
                new_matrix[i][j] = 1.0 / (1.0 + exp(-matrix[i][j]));
            }
            else {
                new_matrix[i][j] = exp(matrix[i][j]) / (1.0 + exp(matrix[i][j]));
            }

        }
    }
    return new_matrix;
}

// Softmax 함수
void Softmax(int rows, int cols, double** matrix) {
    for (int i = 0; i < rows; i++) {
        // 각 행에 대해 Softmax 함수 적용
        double sumExp = 0.0;

        // 각 원소에 대해 지수 함수 계산 및 합 계산
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = exp(matrix[i][j]);
            sumExp += matrix[i][j];
        }

        // Softmax 함수 적용
        for (int j = 0; j < cols; j++) {
            matrix[i][j] /= sumExp;
        }
    }
}

double cross_entropy_loss(double** label, double** outputs) {
    double cross_entropy_val = 0;

    for (int i = 0; i < sample_num; ++i) {
        for (int j = 0; j < output_size; ++j) {
            double true_prob = label[i][j];
            double predicted_prob = outputs[i][j];

            if (true_prob > 0) {
                cross_entropy_val -= true_prob * log2(predicted_prob);
            }
        }
    }
    cross_entropy_val = -cross_entropy_val / (sample_num * output_size);
    return cross_entropy_val;
}


void gradientDiscent(double** w, double** dw, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            w[i][j] = w[i][j] - learning_rate * dw[i][j];
        }
    }
    return;
}


// 신경망 계산 ================================================
typedef struct NeuralNet {
    double** w1;
    double** w2;
    double** w3;
    double** b1;
    double** b2;
    double** b3;
}NN;

// 순전파 결과인 dictionary를 구현한 구조체
typedef struct dict {
    double** a0;
    double** z1;
    double** a1;
    double** z2;
    double** a2;
    double** z3;
    double** a3;
}DICT;

NN initNetwork() {
    NN network;
    network.w1 = createMatrix(hidden_size1, input_size);
    network.w2 = createMatrix(hidden_size2, hidden_size1);
    network.w3 = createMatrix(output_size, hidden_size2);

    return network;
}

DICT propagate(NN network, double** X) {

    // 2 hidden layer, 1 output layer 순전파 진행
    double** z1 = matrixDotProduct(network.w1, X, hidden_size1, input_size, sample_num);
    double** a1 = ReLU(hidden_size1, sample_num, z1);

    double** z2 = matrixDotProduct(network.w2, a1, hidden_size2, hidden_size1, sample_num);
    double** a2 = ReLU(hidden_size2, sample_num, z2);

    double** z3 = matrixDotProduct(network.w3, a2, output_size, hidden_size2, sample_num);
    double** a3 = Sigmoid(output_size, sample_num, z3);

    //Softmax(output_size, sample_num, a3);

    DICT dict = { X, z1, a1, z2, a2, z3, a3 };

    return dict;
}


void backpropagate(NN network, DICT dict, double** label_matrix) {
    // 라벨 매트릭스 생성하기

    // gradient 계산하기. chain rule.


    double** transposed_a3 = transposeMatrix(output_size, sample_num, dict.a3);
    double** dz_3 = subtractMatrix(label_matrix, transposed_a3, sample_num, output_size);
    double** transposed_dz_3 = transposeMatrix(sample_num, output_size, dz_3);
    double** transposed_a2 = transposeMatrix(hidden_size2, sample_num, dict.a2);
    double** dw_3 = matrixDotProduct(transposed_dz_3, transposed_a2, output_size, sample_num, hidden_size2);
    double** transposed_w3 = transposeMatrix(output_size, hidden_size2, network.w3);
    double** da_2 = matrixDotProduct(transposed_w3, transposed_dz_3, hidden_size2, output_size, sample_num);

    double** relu_z2 = ReLU_derivative(hidden_size2, sample_num, dict.z2);
    double** dz_2 = multiplyMatrix(da_2, relu_z2, hidden_size2, sample_num);
    double** transposed_a1 = transposeMatrix(hidden_size1, sample_num, dict.a1);
    double** dw_2 = matrixDotProduct(dz_2, transposed_a1, hidden_size2, sample_num, hidden_size1);
    double** transposed_w2 = transposeMatrix(hidden_size2, hidden_size1, network.w2);
    double** da_1 = matrixDotProduct(transposed_w2, dz_2, hidden_size1, hidden_size2, sample_num);

    double** relu_z1 = ReLU_derivative(hidden_size1, sample_num, dict.z1);
    double** dz_1 = multiplyMatrix(da_1, relu_z1, hidden_size1, sample_num);
    double** transposed_a0 = transposeMatrix(input_size, sample_num, dict.a0);
    double** dw_1 = matrixDotProduct(dz_1, transposed_a0, hidden_size1, sample_num, input_size);
    double** transposed_w1 = transposeMatrix(hidden_size1, input_size, network.w1);
    double** da_0 = matrixDotProduct(transposed_w1, dz_1, input_size, hidden_size1, sample_num);



    // 경사하강법으로 가중치 업데이트
    gradientDiscent(network.w1, dw_1, hidden_size1, input_size);
    gradientDiscent(network.w2, dw_2, hidden_size2, hidden_size1);
    gradientDiscent(network.w3, dw_3, output_size, hidden_size2);


    freeMatrix(output_size, dz_3);
    freeMatrix(output_size, dw_3);
    freeMatrix(hidden_size2, da_2);

    freeMatrix(hidden_size2, dz_2);
    freeMatrix(hidden_size2, dw_2);
    freeMatrix(hidden_size1, da_1);

    freeMatrix(hidden_size1, dz_1);
    freeMatrix(hidden_size1, dw_1);
    freeMatrix(input_size, da_0);

    /*freeMatrix(sample_num, transposed_a0);
    freeMatrix(sample_num, transposed_a1);
    freeMatrix(sample_num, transposed_a2);
    freeMatrix(sample_num, transposed_a3);

    freeMatrix(input_size, transposed_w1);
    freeMatrix(hidden_size1, transposed_w2);
    freeMatrix(hidden_size2, transposed_w3);
    freeMatrix(output_size, transposed_dz_3);

    freeMatrix(hidden_size1, relu_z1);
    freeMatrix(hidden_size2, relu_z2);*/


    return;
}

int test_calAccuracy(double** outputs, char* label) {
    // 각 row 별로 최대값의 인덱스 찾기. -> 분류 결과 0~6
    int index_list[20];
    for (int i = 0; i < 20; i++) {
        double row_max = outputs[i][0];
        int max_index = 0;
        for (int j = 1; j < output_size; j++) {
            if (row_max < outputs[i][j]) {
                row_max = outputs[i][j];
                max_index = j;
            }
        }
        index_list[i] = max_index;
    }

    // 알파벳 리스트 label을 알파벳에 따라서 0~6의 숫자로 인코딩.
    int label_list[20];
    for (int i = 0; i < 20; i++) {
        int target_label;
        if (label[i] == 't') {
            target_label = 0;
        }
        else if (label[i] == 'u') {
            target_label = 1;
        }
        else if (label[i] == 'v') {
            target_label = 2;
        }
        else if (label[i] == 'w') {
            target_label = 3;
        }
        else if (label[i] == 'x') {
            target_label = 4;
        }
        else if (label[i] == 'y') {
            target_label = 5;
        }
        else if (label[i] == 'z') {
            target_label = 6;
        }
        // 인코딩된 숫자 넣어주기.
        label_list[i] = target_label;
    }

    // 이제 두 리스트를 비교하며 일치하는 개수 세기
    int count = 0;
    for (int i = 0; i < 20; i++) {
        //printf("정답 : %c %d \t 예측 : %d \n", label[i], label_list[i], index_list[i]);
        if (index_list[i] == label_list[i]) {
            count++;
        }
    }
    printf("정답 개수 : %d \n", count);

    return count;
}

void test(NN network, double** X, char* label) {

    // 2 hidden layer, 1 output layer 순전파 진행
    double** z1 = matrixDotProduct(network.w1, X, hidden_size1, input_size, 20);
    double** a1 = ReLU(hidden_size1, 20, z1);

    double** z2 = matrixDotProduct(network.w2, a1, hidden_size2, hidden_size1, 20);
    double** a2 = ReLU(hidden_size2, 20, z2);

    double** z3 = matrixDotProduct(network.w3, a2, output_size, hidden_size2, 20);
    double** a3 = Sigmoid(output_size, 20, z3);


    double** transposed_output = transposeMatrix(output_size, 20, a3);

    int count = test_calAccuracy(transposed_output, label);

    return;
}


// 주어진 레이블로 레이블링된 매트릭스를 생성하는 함수.
double** one_hot_encoding(char* label) {
    double** label_matrix = (double**)malloc(sample_num * sizeof(double*));

    for (int i = 0; i < sample_num; i++) {
        label_matrix[i] = (double*)malloc(output_size * sizeof(double));

        for (int j = 0; j < output_size; j++) {
            label_matrix[i][j] = 0.0;
        }

        if (label[i] == 't') {
            label_matrix[i][0] = 1.0;
        }
        else if (label[i] == 'u') {
            label_matrix[i][1] = 1.0;
        }
        else if (label[i] == 'v') {
            label_matrix[i][2] = 1.0;
        }
        else if (label[i] == 'w') {
            label_matrix[i][3] = 1.0;
        }
        else if (label[i] == 'x') {
            label_matrix[i][4] = 1.0;
        }
        else if (label[i] == 'y') {
            label_matrix[i][5] = 1.0;
        }
        else if (label[i] == 'z') {
            label_matrix[i][6] = 1.0;
        }
    }

    return label_matrix;
}

int calAccuracy(double** outputs, char* label) {
    // 각 row 별로 최대값의 인덱스 찾기. -> 분류 결과 0~6
    int index_list[sample_num];
    for (int i = 0; i < sample_num; i++) {
        double row_max = outputs[i][0];
        int max_index = 0;
        for (int j = 1; j < output_size; j++) {
            if (row_max < outputs[i][j]) {
                row_max = outputs[i][j];
                max_index = j;
            }
        }
        index_list[i] = max_index;
    }

    // 알파벳 리스트 label을 알파벳에 따라서 0~6의 숫자로 인코딩.
    int label_list[sample_num];
    for (int i = 0; i < sample_num; i++) {
        int target_label;
        if (label[i] == 't') {
            target_label = 0;
        }
        else if (label[i] == 'u') {
            target_label = 1;
        }
        else if (label[i] == 'v') {
            target_label = 2;
        }
        else if (label[i] == 'w') {
            target_label = 3;
        }
        else if (label[i] == 'x') {
            target_label = 4;
        }
        else if (label[i] == 'y') {
            target_label = 5;
        }
        else if (label[i] == 'z') {
            target_label = 6;
        }
        // 인코딩된 숫자 넣어주기.
        label_list[i] = target_label;
    }

    // 이제 두 리스트를 비교하며 일치하는 개수 세기
    int count = 0;
    for (int i = 0; i < sample_num; i++) {
        //printf("정답 : %c %d \t 예측 : %d \n", label[i], label_list[i], index_list[i]);
        if (index_list[i] == label_list[i]) {
            count++;
        }
    }
    printf("정답 개수 : %d \n", count);

    return count;
}

void forward(NN network, double** X, char* label) {
    // 순전파 + 역전파 반복
    for (int i = 0; i < epochs; i++) {
        printf("epochs : %d \n", i + 1);
        DICT dict = propagate(network, X);
        //print(output_size, sample_num, dict.a3);
        double** label_matrix = one_hot_encoding(label);
        backpropagate(network, dict, label_matrix);

        double** transposed_output = transposeMatrix(output_size, sample_num, dict.a3);
        printf("training loss : %f \n", cross_entropy_loss(label_matrix, transposed_output));
        int count = calAccuracy(transposed_output, label);

        // 메모리 해제
        freeMatrix(hidden_size1, dict.z1);
        freeMatrix(hidden_size1, dict.a1);

        freeMatrix(hidden_size2, dict.z2);
        freeMatrix(hidden_size2, dict.a2);

        freeMatrix(output_size, dict.z3);
        freeMatrix(output_size, dict.a3);
    }
}

// ===================================================
#define IMAGE_SIZE 16
#define MAX_PIXEL_VALUE 255

void read_image(const char* filename, double** image, int i) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }

    // PGM 헤더 읽기
    char magic[3];
    fscanf(file, "%2s", magic);
    if (magic[0] != 'P' || magic[1] != '5') {
        fprintf(stderr, "Invalid PGM format\n");
        fclose(file);
        exit(1);
    }

    int width, height, max_pixel_value;
    fscanf(file, "%d %d %d", &width, &height, &max_pixel_value);

    // 이미지 크기 확인
    if (width != IMAGE_SIZE || height != IMAGE_SIZE || max_pixel_value != MAX_PIXEL_VALUE) {
        fprintf(stderr, "Invalid image dimensions or pixel values\n");
        fclose(file);
        exit(1);
    }

    // 의도치 않은 개행 문자가 있는 경우, 무시
    while (fgetc(file) != '\n');

    // 이미지 데이터 읽기
    unsigned char pixel;
    for (int j = 0; j < input_size; ++j) {
        fread(&pixel, sizeof(unsigned char), 1, file);
        image[j][i] = (double)pixel / 255.0;  // 0에서 255까지의 값을 0에서 255까지의 부동소수점 값으로 변환
    }


    fclose(file);
}


int main() {
    srand(time(NULL));

    // 0. 입력데이터 사진 불러오기 ================================
    double** image_data = createMatrix0(input_size, sample_num);
    const char* filenames[sample_num] = {
        "./image/train/train1.pgm","./image/train/train2.pgm","./image/train/train3.pgm","./image/train/train4.pgm","./image/train/train5.pgm","./image/train/train6.pgm","./image/train/train7.pgm","./image/train/train8.pgm","./image/train/train9.pgm","./image/train/train10.pgm",
        "./image/train/train11.pgm","./image/train/train12.pgm","./image/train/train13.pgm","./image/train/train14.pgm","./image/train/train15.pgm","./image/train/train16.pgm","./image/train/train17.pgm","./image/train/train18.pgm","./image/train/train19.pgm","./image/train/train20.pgm",
        "./image/train/train21.pgm","./image/train/train22.pgm","./image/train/train23.pgm","./image/train/train24.pgm","./image/train/train25.pgm","./image/train/train26.pgm","./image/train/train27.pgm","./image/train/train28.pgm","./image/train/train29.pgm","./image/train/train30.pgm",
        "./image/train/train31.pgm","./image/train/train32.pgm","./image/train/train33.pgm","./image/train/train34.pgm","./image/train/train35.pgm","./image/train/train36.pgm","./image/train/train37.pgm","./image/train/train38.pgm","./image/train/train39.pgm","./image/train/train40.pgm",
        "./image/train/train41.pgm","./image/train/train42.pgm","./image/train/train43.pgm","./image/train/train44.pgm","./image/train/train45.pgm","./image/train/train46.pgm","./image/train/train47.pgm","./image/train/train48.pgm","./image/train/train49.pgm","./image/train/train50.pgm",
        "./image/train/train51.pgm","./image/train/train52.pgm","./image/train/train53.pgm","./image/train/train54.pgm","./image/train/train55.pgm","./image/train/train56.pgm","./image/train/train57.pgm","./image/train/train58.pgm","./image/train/train59.pgm","./image/train/train60.pgm",
        "./image/train/train61.pgm","./image/train/train62.pgm","./image/train/train63.pgm","./image/train/train64.pgm","./image/train/train65.pgm","./image/train/train66.pgm","./image/train/train67.pgm","./image/train/train68.pgm","./image/train/train69.pgm","./image/train/train70.pgm",
        "./image/train/train71.pgm","./image/train/train72.pgm","./image/train/train73.pgm","./image/train/train74.pgm","./image/train/train75.pgm","./image/train/train76.pgm","./image/train/train77.pgm","./image/train/train78.pgm","./image/train/train79.pgm","./image/train/train80.pgm",
        "./image/train/train81.pgm","./image/train/train82.pgm","./image/train/train83.pgm","./image/train/train84.pgm","./image/train/train85.pgm","./image/train/train86.pgm","./image/train/train87.pgm","./image/train/train88.pgm","./image/train/train89.pgm","./image/train/train90.pgm",
        "./image/train/train91.pgm","./image/train/train92.pgm","./image/train/train93.pgm","./image/train/train94.pgm","./image/train/train95.pgm","./image/train/train96.pgm","./image/train/train97.pgm","./image/train/train98.pgm","./image/train/train99.pgm","./image/train/train100.pgm",
        "./image/train/train101.pgm","./image/train/train102.pgm","./image/train/train103.pgm","./image/train/train104.pgm","./image/train/train105.pgm"
    };
    char label[sample_num] = {
        't', 'u', 'v', 'w', 'x', 'y', 'z', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        't', 'u', 'v', 'w', 'x', 'y', 'z', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        't', 'u', 'v', 'w', 'x', 'y', 'z', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        't', 'u', 'v', 'w', 'x', 'y', 'z', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        't', 'u', 'v', 'w', 'x', 'y', 'z', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        't', 'u', 'v', 'w', 'x', 'y', 'z', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        't', 'u', 'v', 'w', 'x', 'y', 'z', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        't', 'u', 'v', 'w', 'x', 'y', 'z'
    };


    // 이미지 읽기
    for (int i = 0; i < sample_num; i++) {
        read_image(filenames[i], image_data, i);
    }


    // 1. 뉴럴네트워크 생성 및 가중치 초기화
    NN network = initNetwork();

    // 2. 순전파 진행
    forward(network, image_data, label);

    // 3. 테스트 진행
    const char* test_filenames[20] = {
        "./image/test/test1.pgm","./image/test/test2.pgm","./image/test/test3.pgm","./image/test/test4.pgm","./image/test/test5.pgm","./image/test/test6.pgm","./image/test/test7.pgm","./image/test/test8.pgm","./image/test/test9.pgm","./image/test/test10.pgm",
        "./image/test/test11.pgm","./image/test/test12.pgm","./image/test/test13.pgm","./image/test/test14.pgm","./image/test/test15.pgm","./image/test/test16.pgm","./image/test/test17.pgm","./image/test/test18.pgm","./image/test/test19.pgm","./image/test/test20.pgm"
    };

    char test_label[20] = {
        't', 'u', 'v', 'w', 'x', 'y', 'z', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        't', 'u', 'v', 'w', 'x', 'y'
    };

    double** test_data = createMatrix0(input_size, 20);
    for (int i = 0; i < 20; i++) {
        read_image(test_filenames[i], test_data, i);
    }
    test(network, test_data, label);

    return 0;
}