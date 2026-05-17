"""Neural Geometry Lab: small, practical experiments inspired by Goodfire's neural geometry posts."""

from .core import LoadedModel, extract_activations, load_lm, save_activation_bundle
from .datasets import (
    HOURS,
    MONTHS,
    WEEKDAYS,
    make_addition_dataset,
    make_cyclic_addition_dataset,
    make_month_dataset,
    make_number_dataset,
    make_weekday_dataset,
)
from .geometry import circular_concept_metrics, fit_fourier_probe, fit_fourier_probes

__all__ = [
    "LoadedModel",
    "extract_activations",
    "load_lm",
    "save_activation_bundle",
    "WEEKDAYS",
    "MONTHS",
    "HOURS",
    "make_addition_dataset",
    "make_cyclic_addition_dataset",
    "make_month_dataset",
    "make_number_dataset",
    "make_weekday_dataset",
    "circular_concept_metrics",
    "fit_fourier_probe",
    "fit_fourier_probes",
]
