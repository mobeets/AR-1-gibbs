import numpy as np
import pandas as pd

def load(infile):
    return pd.read_csv(infile)

def main(infile='temp.csv'):
    df = load(infile)

if __name__ == '__main__':
    main()
