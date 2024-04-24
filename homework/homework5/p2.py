from math import sqrt, log

def general_error(N, dvc=10, delta=0.05):
    return sqrt(8 / N * log(4 * ((2 * N) ** dvc + 1) / delta))

left, right = int(1), int(1e8)
ans = 1

while left <= right:
    mid = (left + right) // 2
    if general_error(mid) <= 0.05:
        ans = mid
        right = mid - 1
    else:
        left = mid + 1

print('The minimum sample size is N = ', ans)
print(f'N = {ans - 1}, generalization error = {general_error(ans - 1)}')
print(f'N = {ans}, generalization error = {general_error(ans)}')