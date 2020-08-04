def NLL(y_true, y_hat):
    return -y_hat.log_prob(y_true)