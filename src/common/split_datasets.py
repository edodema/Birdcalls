"""
Utilities function for splitting the dataset in train and split, I would have put it in
src.common.utils but there is a circular import with src.pl_data.dataset.

- Build:
    - birdcalls_split_dataset
    - soundscapes_split_dataset

- Retrieve:
    - birdcalls_get_splits
    - soundscapes_get_splits
"""
import numpy as np
import pandas as pd
from src.pl_data.dataset import BirdcallDataset, SoundscapeDataset
from typing import Optional


def birdcalls_split_dataset(
    data_path: str,
    split_path: Optional[str],
    p: float = 0.8,
    autosave: bool = True,
    **kwargs
):
    """
    Create a dataset that splits a dataset in (p*100)% train and 100(1-p)% eval, this is done
    probabilistically so in not necessarily the same yet is quite good.
    :param data_path: Path of the original CSV data file, the one used in the Dataset object.
    :param split_path: The path of the file we want, eventually, to save on the dataframe.
    :param p: The probability of being in train/train set size as a percentage of the whole dataset..
    :param autosave: If true the df is saved to split_path.
    :param kwargs:
    :return: The new DataFrame object.
    """
    assert (
        0 <= p and p <= 1
    ), "The probability of a sample being in the train set must be between 0 and 1!"
    dataset = BirdcallDataset(csv_path=data_path)

    # Randomly choose if each sample will be in the train or eval split.
    # 1 means that the value is in the train set and 0 that it is in the eval one.
    p_train = (np.random.rand(len(dataset.rating)) <= p).astype(int)

    # Build up a DatFrame adding it.
    series = [
        dataset.primary_label,
        dataset.scientific_name,
        dataset.common_name,
        dataset.filename,
        dataset.rating,
        pd.Series(data=p_train, index=None, name="train"),
    ]
    df = pd.concat(series, axis=1)
    if autosave:
        df.to_csv(path_or_buf=split_path, index=False)
    return df


def soundscapes_split_dataset(
    data_path: str,
    split_path: Optional[str],
    p: float = 0.8,
    autosave: bool = True,
    **kwargs
):
    """
    Create a dataset that splits a dataset in (p*100)% train and 100(1-p)% eval, this is done
    probabilistically so in not necessarily the same yet is quite good.
    :param data_path: Path of the original CSV data file, the one used in the Dataset object.
    :param split_path: The path of the file we want, eventually, to save on the dataframe.
    :param p: The probability of being in train/train set size as a percentage of the whole dataset..
    :param autosave: If true the df is saved to split_path.
    :param kwargs:
    :return: The new DataFrame object.
    """
    assert (
        0 <= p and p <= 1
    ), "The probability of a sample being in the train set must be between 0 and 1!"
    dataset = SoundscapeDataset(csv_path=data_path)

    # Randomly choose if each sample will be in the train or eval split.
    # 1 means that the value is in the train set and 0 that it is in the eval one.
    p_train = (np.random.rand(len(dataset.birds)) <= p).astype(int)

    # # Build up a DatFrame adding it.
    series = [
        pd.Series(data=dataset.row_id, index=None, name="row_id"),
        pd.Series(data=dataset.site, index=None, name="site"),
        pd.Series(data=dataset.audio_id, index=None, name="audio_id"),
        pd.Series(data=dataset.seconds, index=None, name="seconds"),
        pd.Series(data=dataset.birds, index=None, name="birds"),
        pd.Series(data=p_train, index=None, name="train"),
    ]
    df = pd.concat(series, axis=1)
    if autosave:
        df.to_csv(path_or_buf=split_path, index=False)
    return df


def birdcalls_get_splits(
    split_path: str,
    train_path: Optional[str] = None,
    eval_path: Optional[str] = None,
    autosave: bool = False,
    **kwargs
):
    """
    Once we have out new DataFrame where each sample is annotated as being in train or eval
    we want to actually retrieve the train and eval files.
    :param split_path: Path of the file to read.
    :param train_path: Path to save to the train split.
    :param eval_path: Path to save to the eval split.
    :param autosave: If true save the filtered train and eval dataframes.
    :param kwargs:
    :return: A dictionary with the traina and eval DataFrames.
    """
    df = pd.read_csv(filepath_or_buffer=split_path)
    eval_df = df.loc[df["train"] == 0]
    train_df = df.loc[df["train"] == 1]
    if autosave:
        train_df.to_csv(path_or_buf=train_path)
        eval_df.to_csv(path_or_buf=eval_path)
    return {"train": train_df, "eval": eval_df}


def soundscapes_get_splits(
    split_path: str,
    train_path: Optional[str] = None,
    eval_path: Optional[str] = None,
    autosave: bool = False,
    **kwargs
):
    """
    Once we have out new DataFrame where each sample is annotated as being in train or eval
    we want to actually retrieve the train and eval files.
    N.B. This function is a copy of the previous one, it exists only to make things easier with Hydra.
    :param split_path: Path of the file to read.
    :param train_path: Path to save to the train split.
    :param eval_path: Path to save to the eval split.
    :param autosave: If true save the filtered train and eval dataframes.
    :param kwargs:
    :return: A dictionary with the traina and eval DataFrames.
    """
    df = pd.read_csv(filepath_or_buffer=split_path, index_col=False)
    eval_df = df.loc[df["train"] == 0]
    train_df = df.loc[df["train"] == 1]
    if autosave:
        train_df.to_csv(path_or_buf=train_path)
        eval_df.to_csv(path_or_buf=eval_path)
    return {"train": train_df, "eval": eval_df}
