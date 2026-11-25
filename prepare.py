"""Create the fixed split and sample from the training rows."""

from sklearn.model_selection import train_test_split

SPLIT_SEED = 2026


def split_tickets(tickets, seed=SPLIT_SEED):
    train, remaining = train_test_split(
        tickets,
        test_size=0.40,
        stratify=tickets["label"],
        random_state=seed,
    )
    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        stratify=remaining["label"],
        random_state=seed,
    )
    return train, validation, test


def sample_training_rows(train, size, seed):
    if size == len(train):
        return train.copy()

    sample, _ = train_test_split(
        train,
        train_size=size,
        stratify=train["label"],
        random_state=seed,
    )
    return sample
