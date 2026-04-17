import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    arr = []
    for pos in range(seq_len):
        temp = []
        for i in range(d_model):
            two_i = (i//2)*2
            if i%2 == 1:
                temp.append(np.cos(pos/(base**(two_i/d_model))))
            else:
                temp.append(np.sin(pos/(base**(two_i/d_model))))
        arr.append(temp)
    return np.array(arr)