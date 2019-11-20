  
"""
- The module allows for the download of popular causl inferenfce data
- This allows for quick implementation in a variety of different contexts
- All datasets are cleaned in the loading process

Data at a glance
----------------
1. Kevin Hilstorm's MineThatData email marketing dataset (email marketing)
2. The Primary Biliary Cirrhosis dataset provided by the Mayo Clinic (medical trials)
3. A dataset on microfinance from The Institute for Financial Management Research in Chennai, India (socio-economic)
"""

from .download_utils import download_file, get_download_paths

from .hillstrom import download_hillstrom, load_hillstrom
from .mayo_pbc import download_mayo_pbc, load_mayo_pbc
from .ifmr_microfinance import downlaod_ifmr_microfinance, load_ifmr_microfinance