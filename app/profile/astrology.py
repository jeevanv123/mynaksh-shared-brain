"""Stubbed astrology.

The assignment explicitly does not want an astrology engine. This module gives
the conversational layer just enough structure to personalize: a Western sun
sign from the birth date, plus the element, modality and ruling planet that go
with it. A real engine (Vedic rashi, nakshatra, dasha) would plug in behind the
same `AstroProfile` shape.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date

# (month, day) the sign starts on, sign name, element, modality, ruling planet
_SIGNS = [
    ((1, 20), "Aquarius", "Air", "Fixed", "Saturn"),
    ((2, 19), "Pisces", "Water", "Mutable", "Jupiter"),
    ((3, 21), "Aries", "Fire", "Cardinal", "Mars"),
    ((4, 20), "Taurus", "Earth", "Fixed", "Venus"),
    ((5, 21), "Gemini", "Air", "Mutable", "Mercury"),
    ((6, 21), "Cancer", "Water", "Cardinal", "Moon"),
    ((7, 23), "Leo", "Fire", "Fixed", "Sun"),
    ((8, 23), "Virgo", "Earth", "Mutable", "Mercury"),
    ((9, 23), "Libra", "Air", "Cardinal", "Venus"),
    ((10, 23), "Scorpio", "Water", "Fixed", "Mars"),
    ((11, 22), "Sagittarius", "Fire", "Mutable", "Jupiter"),
    ((12, 22), "Capricorn", "Earth", "Cardinal", "Saturn"),
]

_TRAITS = {
    "Aries": "bold, action-first, thrives on fresh starts",
    "Taurus": "steady, patient, values security and tangible results",
    "Gemini": "curious, communicative, energised by variety",
    "Cancer": "intuitive, protective, guided by emotional security",
    "Leo": "confident, expressive, drawn to leadership and recognition",
    "Virgo": "analytical, diligent, perfects through detail",
    "Libra": "balanced, diplomatic, seeks harmony and fairness",
    "Scorpio": "intense, strategic, transforms through depth",
    "Sagittarius": "optimistic, exploratory, learns by ranging widely",
    "Capricorn": "disciplined, ambitious, builds for the long term",
    "Aquarius": "independent, inventive, motivated by ideas and community",
    "Pisces": "empathetic, imaginative, moves by intuition",
}


@dataclass(frozen=True)
class AstroProfile:
    sun_sign: str
    element: str
    modality: str
    ruling_planet: str
    traits: str

    def as_dict(self) -> dict:
        return asdict(self)


def sun_sign_for(dob: date) -> AstroProfile:
    """Return the Western sun sign for a birth date."""
    md = (dob.month, dob.day)
    current = _SIGNS[-1]  # Capricorn wraps the year boundary (Dec 22 - Jan 19)
    for entry in _SIGNS:
        if md >= entry[0]:
            current = entry
    _, sign, element, modality, planet = current
    return AstroProfile(sign, element, modality, planet, _TRAITS[sign])
