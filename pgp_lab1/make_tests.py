N = [335, 3355, 33554, 335544, 3355443, 33554431]

for i in range(len(N)):
    file_name = 'tests/test_{}.t'.format(N[i])
    with open(file_name, "w+") as file:
        file.write('{}\n'.format(N[i]))
        for j in range(1, N[i]+1):
            file.write('{} '.format(j))
        file.write('\n')