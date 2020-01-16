from util.data import read_struct

data = read_struct("sample.csv")

# print(data)
# print(type(data))
print()
header = list(data.dtype.names)
print(header)
print()
print(header[0])
print(data[header[0]])


