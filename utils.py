import numpy as np
def calculate_diff(dct_blocks, P, Q, x, y, direction):
    # if direction == 'LR' and Q < dct_blocks.shape[1] - 1:
    #     Diff = dct_blocks[P, Q, x, y] - dct_blocks[P, Q + 1, x, y]
    # elif direction == 'UD' and P < dct_blocks.shape[0] - 1:
    #     Diff = dct_blocks[P, Q, x, y] - dct_blocks[P + 1, Q, x, y]
    # elif direction == 'DU' and P > 0:
    #     Diff =  dct_blocks[P, Q, x, y] - dct_blocks[P - 1, Q, x, y]
    # elif direction == 'RL' and Q > 0:
    #     Diff = dct_blocks[P, Q, x, y] - dct_blocks[P, Q - 1, x, y]
    # else:
    #     Diff = 0  # For edge blocks, Diff is set to 0

    result = 0

    if direction == 'LR' and Q < dct_blocks.shape[1] - 1:
        Diff = dct_blocks[P, Q, x, y] - dct_blocks[P, Q + 1, x, y-1]
        result = dct_blocks[P, Q + 1, x, y-1]
    elif direction == 'UD' and P < dct_blocks.shape[0] - 1:
        Diff = dct_blocks[P, Q, x, y] - dct_blocks[P + 1, Q, x, y-1]
        result = dct_blocks[P + 1, Q, x, y-1]
    elif direction == 'DU' and P > 0:
        Diff =  dct_blocks[P, Q, x, y] - dct_blocks[P - 1, Q, x, y-1]
        result = dct_blocks[P - 1, Q, x, y-1]
    elif direction == 'RL' and Q > 0:
        Diff = dct_blocks[P, Q, x, y] - dct_blocks[P, Q - 1, x, y-1]
        result = dct_blocks[P, Q - 1, x, y-1]
    else:
        Diff, result = 0  # For edge blocks, Diff is set to 0
    # Diff, result = dct_blocks[P, Q, x, y] - dct_blocks[P, Q, x, y-1]
    return Diff, result

def calculate_M(dct_blocks, i, j, Z, block_size):
    zigzag = np.concatenate([np.diagonal(dct_blocks[i, j][::-1, :], k)[::(2 * (k % 2) - 1)] for k in range(1 - block_size, block_size)])
    Med = np.median(zigzag[1:10])
    DC = dct_blocks[i, j, 0, 0]

    # Compute modification parameter
    if abs(DC) > 1000 or abs(DC) < 1:
        M = Z * Med
    else:
        M = Z * (DC - Med) / DC
    return M if M > 1e-3 else 1.0

def f4(dct_blocks, watermark_bit, Diff, Th, K, i, j, Z, block_size, x, y, direction, M):
    if watermark_bit == 1:
        if Diff > Th - K:
            while Diff > Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
                # print(i, j, x, y, M, Diff)
        elif Diff < K and Diff > -Th / 2:
            while Diff < K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
                # print(i, j, x, y, M, Diff)
        elif Diff < -Th / 2:
            while Diff > -Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
                # print(i, j, x, y, M, Diff)
    else:
        if Diff > Th / 2:
            while Diff <= Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
                # print(i, j, x, y, M, Diff)
        elif Diff > -K and Diff < Th / 2:
            while Diff >= -K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
                # print(i, j, x, y, M, Diff)
        elif Diff < K - Th:
            while Diff <= -Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
                # print(i, j, x, y, M, Diff)
def f4_test(dct_blocks, watermark_bit, Diff, Th, K, i, j, Z, block_size, x, y, direction, result):
    if watermark_bit == 1:
        if Diff > Th - K:
            if Diff > Th - K:
                dct_blocks[i, j, x, y] = result + Th - K
        elif Diff < K and Diff > -Th / 2:
            if Diff < K:
                dct_blocks[i, j, x, y] = result + K
        elif Diff < -Th / 2:
            if Diff > -Th - K:
                dct_blocks[i, j, x, y] = result - Th -K
                # print(i, j, x, y, M, Diff)
    else:
        if Diff > Th / 2:
            if Diff <= Th + K:
                dct_blocks[i, j, x, y] = result + Th + K
                # print(i, j, x, y, M, Diff)
        elif Diff > -K and Diff < Th / 2:
            if Diff >= -K:
                dct_blocks[i, j, x, y] = result - K
        elif Diff < K - Th:
            if Diff <= -Th + K:
                dct_blocks[i, j, x, y] = result - Th + K
                # print(i, j, x, y, M, Diff)

def f16(dct_blocks, watermark_bit, Diff, Th, K, i, j, Z, block_size, x, y, direction, M):
    if watermark_bit == 1:
        if Diff > 6.5 * Th:
            while Diff < 7 * Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif 6 * Th - K < Diff < 6.5 * Th:
            while Diff > 6 * Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif 4.5 * Th < Diff < 5 * Th + K:
            while Diff < 5 * Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif 4 * Th - K < Diff < 4.5 * Th:
            while Diff > 4 * Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif 2.5 * Th < Diff < 3 * Th + K:
            while Diff < 3 * Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif 2 * Th - K < Diff < 2.5 * Th:
            while Diff > 2 * Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif 0.5 * Th < Diff < Th + K:
            while Diff < Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -K < Diff < 0.5 * Th:
            while Diff > -K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -1.5 * Th < Diff < -Th + K:
            while Diff < -Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -2 * Th - K < Diff < -1.5 * Th:
            while Diff > -2 * Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -3.5 * Th < Diff < -3 * Th + K:
            while Diff < -3 * Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -4 * Th - K < Diff < -3.5 * Th:
            while Diff > -4 * Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -5.5 * Th < Diff < -5 * Th + K:
            while Diff < -5 * Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -6 * Th - K < Diff < -5.5 * Th:
            while Diff > -6 * Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif Diff < -7 * Th + K:
            while Diff < -7 * Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
    else:
        if Diff > 7 * Th - K:
            while Diff > 7 * Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif 5.5 * Th < Diff < 6 * Th + K:
            while Diff < 6 * Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif 5 * Th - K < Diff < 5.5 * Th:
            while Diff > 5 * Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif 3.5 * Th < Diff < 4 * Th + K:
            while Diff < 4 * Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif 3 * Th - K < Diff < 3.5 * Th:
            while Diff > 3 * Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif 1.5 * Th < Diff < 2 * Th + K:
            while Diff < 2 * Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif Th - K < Diff < 1.5 * Th:
            while Diff > Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -0.5 * Th < Diff < K:
            while Diff < K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -Th - K < Diff < -0.5 * Th:
            while Diff > -Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -2.5 * Th < Diff < -2 * Th + K:
            while Diff < -2 * Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -3 * Th - K < Diff < -2.5 * Th:
            while Diff > -3 * Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -4.5 * Th < Diff < -4 * Th + K:
            while Diff < -4 * Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -5 * Th - K < Diff < -4.5 * Th:
            while Diff > -5 * Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -6.5 * Th < Diff < -6 * Th + K:
            while Diff < -6 * Th + K:
                dct_blocks[i, j, x, y] += M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)
        elif -7 * Th - K < Diff < -6.5 * Th:
            while Diff > -7 * Th - K:
                dct_blocks[i, j, x, y] -= M
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)
                M = calculate_M(dct_blocks, i, j, Z, block_size)