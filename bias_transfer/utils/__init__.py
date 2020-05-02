def stringify(x):
    if type(x) is dict:
        x = ".".join(["{}_{}".format(k, v) for k, v in x.items()])
    return str(x)


