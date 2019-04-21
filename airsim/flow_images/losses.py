import numpy as np
import torch

def percentage_loss(pred, actual):
    with torch.no_grad():
        size = list(pred.shape)
        # The below mask will be used later to average errors over only non-airfoil pixels
        mask = torch.where(actual == 0, torch.zeros_like(actual), torch.ones_like(actual))
        # Computing the per-pixel 'percentage' loss
        diff = torch.sub(pred, actual)

        # Below, we account for the effect of taking the square root of the square
        diff = torch.abs(diff)
        actual = torch.abs(actual)
        
        errors_n = torch.div(diff, actual)
        # The below line gets rid of all NaNs, setting them to 0 (happens very rarely)
        errors_n[errors_n != errors_n] = 0
        # Get rid of all error values within the airfoil
        errors_n_post_mask = torch.mul(errors_n, mask)
        average_numerator = errors_n_post_mask.view(size[0], size[1], -1).sum(2)
        average_denominator = mask.view(size[0], size[1], -1).sum(2)
        avg_errors_per_batch_elem = torch.div(average_numerator, average_denominator)
        mean_percent_pressure_error = torch.mean(avg_errors_per_batch_elem)
        #pdb.set_trace()
        return mean_percent_pressure_error

def foil_mse_loss(pred, actual):
    size = list(actual.shape)

    # The below mask will be used later to average errors over only non-airfoil pixels
    mask = torch.where(actual == 0, torch.zeros_like(actual), torch.ones_like(actual))
    # Zero out all predictions on the airfoil, since those don't count toward the loss
    pred = torch.mul(pred, mask)
    # Computing the per-pixel error (not squared yet)
    errors = torch.sub(pred, actual)
    squared_errors = torch.mul(errors, errors)

    sum_squared_errors = squared_errors.view(size[0], size[1], -1).sum(2)
    denominators_per_sum = mask.view(size[0], size[1], -1).sum(2)

    mse_per_image = torch.div(sum_squared_errors, denominators_per_sum)
    
    mse = torch.sum(mse_per_image)
    return mse