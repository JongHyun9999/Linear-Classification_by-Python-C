from matrix_control import transposeMatrix
import math


# ReLU 함수 구현
def ReLU(matrix):
    # 결과를 저장할 빈 리스트 생성
    result = []

    for row in matrix:
        result_row = [max(0,element) for element in row]
        result.append(result_row)

    return result

# ReLU함수의 미분인 계단 함수 구현
def ReLU_derivatived(matrix):
    # 결과를 저장할 빈 리스트 생성
    result = []

    # 행렬의 각 요소에 대해 ReLU 미분 연산 수행
    for row in matrix:
        result_row = [1 if element > 0 else 0 for element in row]
        result.append(result_row)

    return result

# 시그모이드 함수를 구현
def sigmoid(x):
    if isinstance(x, (int, float)):
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1 + exp_x)
    elif isinstance(x, list):
        if isinstance(x[0], (int, float)):
            return [sigmoid(val) for val in x]
        elif isinstance(x[0], list):
            return [[sigmoid(val) for val in row] for row in x]
    else:
        raise ValueError("올바른 입력 형식이 아닙니다.")


# def Softmax(matrix):
#     softmaxed_matrix = []

#     # (14X7)
#     for i in range(len(matrix)):
#         sum_value = 0
#         result = []

#         for j in range(len(matrix[0])):
#             sum_value = sum_value + matrix[i][j]
#         for j in range(len(matrix[0])):
#             result.append(matrix[i][j]/sum_value)
#         softmaxed_matrix.append(result)  
#     return transposeMatrix(softmaxed_matrix)
# import math


# 정답 라벨과 출력값 간의 크로스 엔트로피 오차 함수 구현.
def cross_entropy(prob_dist_true, prob_dist_predicted):
    if len(prob_dist_true) != len(prob_dist_predicted):
        raise ValueError("두 확률 분포는 같은 길이여야 합니다.")

    cross_entropy_val = 0
    epsilon = 1e-12  # 아주 작은 양수 값

    for true_prob, predicted_prob in zip(prob_dist_true, prob_dist_predicted):
        if true_prob > 0:
            predicted_prob = max(epsilon, predicted_prob)
            cross_entropy_val -= true_prob * math.log(predicted_prob, 2)

    return cross_entropy_val

# 모델의 출력결과를 입력받아 라벨과 비교해가면서 올바르게 맞춘 갯수를 카운트하는 함수.
def calAccuracy(result, label):
    index_list = []

    for rows in result:
        rows_max = rows[0]
        max_index = 0
        for i in range(1, len(rows)):
            if rows_max < rows[i]:
                rows_max = rows[i]
                max_index = i
        index_list.append(max_index)

    labeled_list = []
    for i in range(len(label)):
        target_label = 0
        if(label[i] == 'u'):
            target_label = 1
        elif(label[i] == 'v'):
            target_label = 2
        elif(label[i] == 'w'):
            target_label = 3
        elif(label[i] == 'x'):
            target_label = 4
        elif(label[i] == 'y'):
            target_label = 5
        elif(label[i] == 'z'):
            target_label = 6
        
        labeled_list.append(target_label)
    
    count = 0
    for i in range(len(index_list)):
        if(index_list[i] == labeled_list[i]):
            count = count + 1
    
    print(f'정답 개수 : {count}')

    return count

# 라벨 알파벳을 출력레이어와 사이즈를 맞춰 인코딩하는 함수.
# 정답만 1, 나머지는 0으로 인코딩 한다.
def one_hot_encoding(label):
    label_t = [1, 0, 0, 0, 0, 0, 0]
    label_u = [0, 1, 0, 0, 0, 0, 0]
    label_v = [0, 0, 1, 0, 0, 0, 0]
    label_w = [0, 0, 0, 1, 0, 0, 0]
    label_x = [0, 0, 0, 0, 1, 0, 0]
    label_y = [0, 0, 0, 0, 0, 1, 0]
    label_z = [0, 0, 0, 0, 0, 0, 1]

    labeled_list = []
    for i in range(len(label)):
        target_label = label_t
        if(label[i] == 'u'):
            target_label = label_u
        elif(label[i] == 'v'):
            target_label = label_v
        elif(label[i] == 'w'):
            target_label = label_w
        elif(label[i] == 'x'):
            target_label = label_x
        elif(label[i] == 'y'):
            target_label = label_y
        elif(label[i] == 'z'):
            target_label = label_z
        
        labeled_list.append(target_label)
    
    return labeled_list