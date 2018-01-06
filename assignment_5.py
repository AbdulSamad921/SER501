import numpy as np


# ============================== Counting Pond ================================


def count_ponds(G):
    # check if the string is empty or not
    if(len(G) == 0):
        return 0
    m = len(G)
    n = len(G[0])
    total = 0
    # making a 2d array to keep track of visited positions
    visited = [[0 for j in range(len(G[0]))]for i in range(len(G))]
    # traversing each point in our 2d array
    for i in range(0, m):
        for j in range(0, n):
            # if the node has not been alreaddy visited
            if(visited[i][j] == 0 and G[i][j] == '#'):
                total = total + 1
                # apply DFS to the node
                DFS(G, i, j, visited)
    return total


def DFS(G, i, j, visited):
    # x y would help us to access all the 8 neighbors
    x = [-1, -1, -1, 0, 0, 1, 1, 1]
    y = [-1, 0, 1, -1, 1, -1, 0, 1]
    # setting the node to true
    visited[i][j] = 1
    # loop for all the 8 neighbors
    for d in range(0, 8):
        a = x[d] + i
        b = y[d] + j
        # check if the node is not in the corner and check if it is not out of
        # range and if it has already been visited or not
        if(a >= 0 and b >= 0 and a < len(G) and b < len(G[0]) and
           visited[a][b] == 0 and G[a][b] == '#'):
                DFS(G, a, b, visited)
# ======================== Longest Ordered Subsequence ========================


def longest_ordered_subsequence(L):
    n = len(L)
    # if the array is empty
    if(n == 0):
        return 0
    # set an array of len n to 1
    T = [1 for i in range(n)]
    # iterating over every value
    for i in range(1, n):
        for j in range(0, i):
            # check if the previous value is greater or not
            if(L[i] > L[j]):
                if(T[j] + 1 > T[i]):
                    T[i] = T[j] + 1
    check = 0
    # check the values of the new array
    # will only take those values that are greater then the previous
    for k in range(0, len(T)):
        if(T[k] > T[check]):
            check = k
    return T[check]
# =============================== Supermarket =================================


def supermarket(Items):
    n = len(Items)
    # if array is empty
    if(n == 0):
        return 0
    # sort items according to their deadline
    Items = sorted(Items, key=lambda x: x[1])
    # make an array to keep track of values
    T = [0 for i in range(n)]
    # setting prices of the Items to the array
    for i in range(0, n):
        T[i] = Items[i][0]
    # iterating over each elements
    for i in range(1, n):
        for j in range(0, i):
            # if deadlines are not same
            if(Items[j][1] != Items[i][1]):
                # if the value of index at T is
                # greater after adding with Item value
                if(T[j] + Items[i][0] > T[i]):
                    T[i] = Items[i][0] + T[j]
    # return max value of the array T
    return np.amax(T)
# ============================== Reducibility =================================


def partition_set_solver(S):
    total = sum(S)

    if total & 1 == 1:
        return False

    total >>= 1
    n = len(S) + 1

    dp = [[False for i in range(total + 1)] for j in range(n)]

    for i in range(n):
        dp[i][0] = True

    for i in range(1, n):
        for j in range(1, total + 1):
            dp[i][j] = dp[i - 1][j]
            if j >= S[i - 1]:
                dp[i][j] = dp[i][j] or dp[i - 1][j - S[i - 1]]
    return dp[n - 1][total]


def subset_sum_solver(S, n):
    # if target value is greater
    if(n < 0):
        return False
    # if S is empty
    if(len(S) == 0):
        return False
    s = sum(S)
    # if sum is less then the target value
    if(s < n):
        return False
    # if 2*target is less then sum
    if(2 * n < s):
        value = s - 2 * n
    # if 2*target greater then the sum
    if(2 * n >= s):
        value = 2 * n - s
    # append value to S
    S.append(value)
    # run partition_set_solver and return value
    return partition_set_solver(S)
# =============================== Unit tests ==================================


def test_suite():

    if count_ponds(["---#--------###-#---",
"----#-----#--#----#-",
"-#--#-####---#------",
"-------###---#-#-#--",
"----#-----------#---",
"--#-#-----#------##-",
"#-#--#--##----#--#--",
"---------#--------#-",
"-#-----#-##----#----",
"#-##-#-#-#--#------#",
"----#-#-#-###-##---#",
"-##---#--#--#--#----",
"----#---####--#-----",
"#--#---------------#",
"-#--###-#------#-#--",
"#-#-------#-#-------",
"-##-####-#--#-----#-",
"-----#-------------#",
"#-#-----#--#--------",
"------#-####---#----"]) == 31:
        print('passed')
    else:
        print('failed')

    if longest_ordered_subsequence([1000]) == 1:
        print('passed')
    else:
        print('failed')
    #print(supermarket([[(50, 1), (70, 1), (80, 1), (60, 2), (40, 3)]]))
    if supermarket([(50, 1), (70, 1), (80, 1), (60, 2), (40, 3)]) == 180:
        print('passed')
    else:
        print('failed')
    if (subset_sum_solver([1, 2, 3, 4], -1) is False):
        print('passed')
    else:
        print('failed')


if __name__ == '__main__':
    test_suite()
