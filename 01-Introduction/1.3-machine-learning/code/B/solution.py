import pandas as pd

def most_frequent(nums):
    nums_series = pd.Series(nums)
    return nums_series.mode()[0]