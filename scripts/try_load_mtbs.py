import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.historical_fires import create_fire_spread_dataset

events=[('CA4004820181108',2018),('CA4017120210808',2021),('NV3757820190807',2019)]

ds=create_fire_spread_dataset(events, size=(256,256), cache_dir=Path('outputs/test_mtbs'))
print('Loaded events count:', len(ds))
for d in ds:
    print(d['event_id'], d['severity'].shape)
