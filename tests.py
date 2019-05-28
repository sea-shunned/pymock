from pathlib import Path

import numpy as np
import pandas as pd

def validate_results(results_df):
    # Load the validation results
    df_path = Path.cwd() / "validation" / "validation_results.csv"
    validation_df = pd.read_csv(
        df_path,
        index_col=False
    )
    # Test that the current run matches
    pd.testing.assert_frame_equal(
        validation_df.drop("time", axis=1),
        results_df.drop("time", axis=1),
        check_dtype=False
    )
    