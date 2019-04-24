import numpy as np
import torch
import pdb

def median_percentage_loss(pred, actual):
        size = list(pred.shape)

        diff = torch.sub(pred, actual)
        diff = torch.abs(diff)
        actual = torch.abs(actual)

        errors_n = torch.div(diff, actual)

        errors_n[errors_n != errors_n] = 0

        max_err = torch.max(errors_n)
        min_err = torch.min(errors_n)

        mask = torch.where(actual == 0, max_err * torch.ones_like(actual), min_err * torch.ones_like(actual))
        maskinv = torch.where(actual != 0, max_err * torch.ones_like(actual), min_err * torch.ones_like(actual))

        large_values_within_airfoil = torch.max(mask, errors_n)
        small_values_within_airfoil = torch.min(maskinv, errors_n)

        large_values_squashed = large_values_within_airfoil.view(size[0], size[1], -1)
        small_values_squashed = small_values_within_airfoil.view(size[0], size[1], -1)

        concatenated_errors = torch.cat((large_values_squashed, small_values_squashed), 2)

        median_error_per_sample = concatenated_errors.median(2)[0]

        sum_percent_pressure_error = torch.sum(median_error_per_sample)

        return sum_percent_pressure_error


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
        #median_error_per_sample = errors_n_post_mask.view(size[0], size[1], -1).median(2)[0]
        avg_errors_per_batch_elem = torch.div(average_numerator, average_denominator)
        mean_percent_pressure_error = torch.mean(avg_errors_per_batch_elem)
        #pdb.set_trace()
        return mean_percent_pressure_error
        #return sum_percent_pressure_error

def foil_mse_loss(pred, actual):
    size = list(actual.shape)

    # The below mask will be used later to average errors over only non-airfoil pixels
    mask = torch.where(actual == 0, torch.zeros_like(actual), torch.ones_like(actual))
    # Computing the per-pixel error (not squared yet)
    errors = torch.sub(pred, actual)
    squared_errors = torch.mul(errors, errors)

    sum_squared_errors = squared_errors.view(size[0], size[1], -1).sum(2)
    denominators_per_sum = mask.view(size[0], size[1], -1).sum(2)
    denominators_per_sum = torch.where(denominators_per_sum == 0,
            torch.ones_like(denominators_per_sum) * (256**2), denominators_per_sum)

    mse_per_image = torch.div(sum_squared_errors, denominators_per_sum)

    mse = torch.sum(mse_per_image)
    return mse
