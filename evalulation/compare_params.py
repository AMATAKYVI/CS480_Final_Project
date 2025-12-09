# Function for counting model parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters())
