"""BedSynth AI - synthetic hospital operations data generator built on SDV."""
from __future__ import annotations

import os
import warnings

# SDV prints tqdm progress bars on every sample call; silence them for library/API use.
# Must be set before tqdm/sdv are imported anywhere.
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore", category=FutureWarning)

__version__ = "1.0.0"
__all__ = ["__version__"]
