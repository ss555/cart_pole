import torch
# setting device on GPU if available, else CPU
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
#
# print(torch.cuda.current_device())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())