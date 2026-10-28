"""
FORECAST - kept so the old command still works; it now simply runs predict.py.

    python forecast.py --canal-level -0.38 --threshold 0.5
is the same as
    python predict.py --canal-level -0.38 --threshold 0.5
See predict.py for all options (live forecast, demo mode with --date, ...).
"""
from predict import main

main()
