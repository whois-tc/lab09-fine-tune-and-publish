"""Invented support messages for Lakeshore Bike Share, for Lab 09."""

import numpy as np
import pandas as pd

LABELS = ["billing", "bike_fault", "station", "account"]

_CORE = {
    "billing": [
        "I was charged twice for my ride on {day}.",
        "My credit card was billed {amount} but the ride only lasted {mins} minutes.",
        "There is a {amount} fee on my statement that I do not recognize.",
        "Please refund the overage charge from {day}.",
        "The receipt for my monthly pass shows the wrong price.",
        "I cancelled my pass but I was still billed this month.",
        "Why was I charged an unlock fee when I have an annual membership?",
        "My invoice for {month} does not match the rides I took.",
        "The amount taken from my bank is higher than what the app promised.",
        "I am being asked to pay for a trip I never finished.",
        "Somebody deducted money from me again after I returned the cycle on time.",
        "Is it normal that my subscription cost went up without notice?",
        "The deposit hold on my card has not been released yet.",
    ],
    "bike_fault": [
        "The brakes on bike {num} are squeaking and feel weak.",
        "The rear tire on bike {num} was flat when I unlocked it.",
        "The chain came off during my ride.",
        "The seat on bike {num} will not stay up and keeps sliding down.",
        "The front light is broken so the bike is unsafe at night.",
        "The pedal fell off after I rode it for a few blocks.",
        "The bell is missing and the handlebar is loose.",
        "The gears will not shift on bike {num}.",
        "Something rattles loudly every time I go over a bump.",
        "The wheel wobbles and the steering feels unsafe.",
        "It was hard to push the cranks around, like something was stuck.",
        "The kickstand snapped and the bicycle keeps falling over.",
        "The cycle pulled to one side the whole way and I nearly fell.",
    ],
    "station": [
        "The dock at {place} would not release the bike.",
        "The station at {place} is full and I could not return my bike.",
        "There were no bikes available at {place} this morning.",
        "The kiosk screen at {place} is frozen.",
        "The station at {place} has been empty all week.",
        "The dock light stayed red and my bike would not lock in.",
        "The terminal at {place} does not accept my code.",
        "Can you add more docks at {place}? It is always full.",
        "I walked to the corner of {place} and every slot was taken.",
        "The bay where I parked the cycle never confirmed that I had returned it.",
        "Nothing was left to ride near {place} during rush hour.",
        "The post that locks the bike clicks but never lets go.",
        "The stand near {place} has an unreadable panel.",
    ],
    "account": [
        "I forgot my password and the reset email never arrived.",
        "I cannot log in to the app after the update.",
        "Please change the email address on my account.",
        "My membership shows as expired but I renewed it.",
        "How do I delete my account and my data?",
        "The app says my phone number is already in use.",
        "I want to switch from a monthly pass to an annual membership.",
        "My profile name is spelled wrong.",
        "The sign-in screen keeps going in circles.",
        "I never got the verification code on my phone.",
        "Can I transfer my plan to my partner?",
        "My student discount has disappeared from my profile.",
        "I would like to pause my plan while I am away for the summer.",
    ],
}
_VAGUE = [
    "This happened again today.",
    "I am not happy with the service.",
    "Please look into this as soon as possible.",
    "Nothing works the way it should.",
    "I need help with my last ride.",
    "Something is wrong and I do not know who to ask.",
]
_OPEN = ["", "", "Hi,", "Hello team,", "Good morning,", "To whom it may concern,"]
_CLOSE = ["", "", "Thanks.", "Please advise.", "This is the second time.", "Thank you."]
_PLACES = [
    "Union Station",
    "Harbourfront",
    "Queen and Spadina",
    "High Park",
    "Bloor and Bathurst",
    "Distillery District",
    "Liberty Village",
    "Riverdale Park",
]
_DAYS = ["Monday", "Tuesday", "Friday", "Saturday", "Sunday", "last week"]
_MONTHS = ["May", "June", "July", "August", "September"]


def _swap_letters(text: str, rng: np.random.Generator) -> str:
    """Swap two neighbouring letters inside one word, as a typing slip."""
    words = text.split(" ")
    spots = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if not spots:
        return text
    i = int(rng.choice(spots))
    w = words[i]
    j = int(rng.integers(1, len(w) - 2))
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(words)


def make_tickets(n: int = 1200, seed: int = 2026) -> pd.DataFrame:
    """Return n invented messages with a label: billing, bike_fault, station, account.

    About one message in eight says nothing about the topic, so its label cannot
    be known from the text. About one in ten mixes two topics, and the label is
    the one that comes first.
    """
    rng = np.random.default_rng(seed)
    labels = rng.choice(LABELS, size=n, p=[0.25, 0.30, 0.25, 0.20])
    rows = []
    for label in labels:
        if rng.random() < 0.12:
            body = str(rng.choice(_VAGUE))
        else:
            first = str(rng.choice(_CORE[label]))
            body = first
            if rng.random() < 0.10:
                other = str(rng.choice([k for k in LABELS if k != label]))
                body = first + " " + str(rng.choice(_CORE[other]))
        body = body.format(
            day=rng.choice(_DAYS),
            amount=f"${rng.choice([2.5, 4.0, 7.5, 12.0, 15.0]):.2f}",
            mins=int(rng.integers(4, 40)),
            month=rng.choice(_MONTHS),
            num=int(rng.integers(1000, 9999)),
            place=rng.choice(_PLACES),
        )
        text = " ".join(
            part
            for part in (str(rng.choice(_OPEN)), body, str(rng.choice(_CLOSE)))
            if part
        )
        if rng.random() < 0.08:
            text = _swap_letters(text, rng)
        rows.append((text, label))
    frame = pd.DataFrame(rows, columns=["text", "label"])
    frame.insert(0, "ticket_id", np.arange(70000, 70000 + n))
    return frame
