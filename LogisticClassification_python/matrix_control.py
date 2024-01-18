# 전치행렬 구현
def transposeMatrix(x):
    transposed_matrix = []
    for i in range(len(x[0])):
        temp_list = []
        for j in range(len(x)):
            temp_list.append(x[j][i])
        transposed_matrix.append(temp_list)

    return transposed_matrix


# 행렬 reshape 구현
def reshapeMatrix(matrix, rows, cols):
    # 입력된 행과 열의 개수에 맞게 리스트를 재구성
    if rows * cols != len(matrix) * len(matrix[0]):
        raise ValueError("새로운 행과 열의 개수가 원본과 일치하지 않습니다.")
    
    # 새로운 행렬을 생성하고 값 복사
    reshaped_matrix = [[0] * cols for _ in range(rows)]
    flat_matrix = [item for sublist in matrix for item in sublist]
    for i in range(rows):
        for j in range(cols):
            reshaped_matrix[i][j] = flat_matrix[i * cols + j]
    
    return reshaped_matrix


# 매트릭스 내적 함수
def matrixDotProduct(matrix1, matrix2):
    # matrix1의 열 수와 matrix2의 행 수가 같은지 확인
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")

    # 내적 계산
    result = []
    for row in matrix1:
        new_row = []
        for col in zip(*matrix2):  # Transpose the second matrix for easier calculation
            new_row.append(sum(x * y for x, y in zip(row, col)))
        result.append(new_row)

    return result


# 두 매트릭스 간 element-wise 뺄셈
def subtractMatrix(matrix1, matrix2):
    # 두 행렬의 크기가 같은지 확인
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions")

    # 결과를 저장할 빈 행렬 생성
    result = []
    
    # 행렬의 각 요소에 대해 뺄셈 연산 수행
    for i in range(len(matrix1)):
        result_row = [matrix1[i][j] - matrix2[i][j] for j in range(len(matrix1[0]))]
        result.append(result_row)
    
    return result

# 두 매트릭스 간 element-wise 덧셈
def addMatrix(matrix1, matrix2):
    # 두 행렬의 크기가 같은지 확인
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions")

    # 결과를 저장할 빈 행렬 생성
    result = []
    
    # 행렬의 각 요소에 대해 뺄셈 연산 수행
    for i in range(len(matrix1)):
        result_row = [matrix1[i][j] + matrix2[i][j] for j in range(len(matrix1[0]))]
        result.append(result_row)
    
    return result


def dimMatrix3to2(matrix):
    # 3차원 매트릭스 전치 후 2차원으로 바꾸기
    changed_matrix = []
    for i in range(len(matrix)):
        changed_matrix.append(transposeMatrix(matrix[i])[0])    
    return transposeMatrix(changed_matrix)


# 매트릭스에 스칼라값 곱해주는 함수
def multiplyScalar(matrix, scalar):
    # 결과를 저장할 빈 리스트 생성
    result = []

    # 행렬의 각 행에 대해 연산 수행
    for row in matrix:
        result_row = [item * scalar for item in row]
        result.append(result_row)
    
    return result

# 두 매트릭스 간 element-wise 곱셈
def element_wise_multiply(matrix1, matrix2):
    # 두 행렬의 크기가 같은지 확인
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions")

    # 결과를 저장할 빈 행렬 생성
    result = []

    # 각 요소에 대해 곱셈 연산 수행
    for i in range(len(matrix1)):
        result_row = [matrix1[i][j] * matrix2[i][j] for j in range(len(matrix1[0]))]
        result.append(result_row)

    return result

# 매트릭스들의 모든 요소들의 합을 구하는 함수
def sumMatrix(x):
    total_sum = 0
    for row in x:
        for elem in row:
            total_sum += elem
    return total_sum


# 매트릭스들의 값 출력하는 함수
def printMatrix(x):
    for row in x:
        print([row]);
    
    return