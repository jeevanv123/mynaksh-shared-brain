from datetime import date

import pytest

from app.profile.astrology import sun_sign_for
from app.profile.models import UserProfile, parse_date_text


@pytest.mark.parametrize(
    "dob, sign",
    [
        (date(1995, 8, 15), "Leo"),
        (date(1990, 1, 1), "Capricorn"),   # year-boundary wrap
        (date(1990, 12, 25), "Capricorn"),
        (date(2000, 1, 20), "Aquarius"),   # first day of a sign
        (date(2000, 3, 20), "Pisces"),     # last day before Aries
        (date(2000, 3, 21), "Aries"),
    ],
)
def test_sun_sign_boundaries(dob, sign):
    assert sun_sign_for(dob).sun_sign == sign


def test_astro_profile_has_supporting_attributes():
    a = sun_sign_for(date(1995, 8, 15))
    assert (a.element, a.ruling_planet) == ("Fire", "Sun")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("15 August 1995", date(1995, 8, 15)),
        ("August 15, 1995", date(1995, 8, 15)),
        ("1995-08-15", date(1995, 8, 15)),
        ("15/08/1995", date(1995, 8, 15)),
        ("3rd Jan 2001", date(2001, 1, 3)),
        ("sometime in the 90s", None),
        ("31 February 2000", None),
    ],
)
def test_parse_date_text(text, expected):
    assert parse_date_text(text) == expected


def test_profile_merge_ignores_empty_and_protected_fields():
    p = UserProfile(user_id="u", name="Rahul")
    merged = p.merged_with({"name": "", "birth_place": "Delhi", "user_id": "hacked", "date_of_birth_text": "15 August 1995"})
    assert merged.user_id == "u" and merged.name == "Rahul"
    assert merged.birth_place == "Delhi" and merged.date_of_birth == date(1995, 8, 15)
    assert merged.astro.sun_sign == "Leo"
