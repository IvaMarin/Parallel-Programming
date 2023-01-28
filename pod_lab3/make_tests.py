N = [160, 1600, 16000, 160000, 1600000]

for i in range(len(N)):
    file_name = 'tests/int{}'.format(N[i])
    with open(file_name, "wb+") as file:
        file.write(N[i].to_bytes(4, byteorder='little'))
        for j in range(N[i], 0, -1):
            file.write(j.to_bytes(4, byteorder='little'))