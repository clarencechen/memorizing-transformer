import torch
import typing


def cycle(iterator: typing.Iterable) -> typing.Iterable[typing.Any]:
    """Create a repeating iterator from an iterator generator."""
    while True:
        for element in iterator:
            yield element

def decode_record(inst):
    output = {}
    output["input_ids_0"] = torch.tensor(inst["input_ids_0"], dtype = torch.long)
    output["mask_0"] = (output["input_ids_0"] != 0).float()
    if "input_ids_1" in inst:
        output["input_ids_1"] = torch.tensor(inst["input_ids_1"], dtype = torch.long)
        output["mask_1"] = (output["input_ids_1"] != 0).float()
    output["label"] = torch.tensor(inst["label"], dtype = torch.long)
    return output
