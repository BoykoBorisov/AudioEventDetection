import torch

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def get_audiofile_name_from_audioset_csv_row(row):
    return row[0].strip(",") + "_" + str(row[1].strip().strip(",")) + ".wav"
