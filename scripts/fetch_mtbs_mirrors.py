import requests
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'mtbs_mirrors'
OUT.mkdir(parents=True, exist_ok=True)

# Candidate URL templates to try (format with event_id and year)
TEMPLATES = [
    # Cloud SDSC mirror (long filename used previously)
    "https://cloud.sdsc.edu/v1/AUTH_mtbs/Raster/MTBS_BurnSeverityData/Collection2/ingestedl3data/{year}/{event_id}/{event_id}_20160209_20170316_CropScenes_20170331_20170406_20170410_20170410_20170417_20170418_20170418_20170420_20170421_20170422_20170425_20170612_20171108_20171208_20171208_20180101_20180101_mtbs_burn_severity.tif",
    # Simpler guess: event_id with suffix
    "https://cloud.sdsc.edu/v1/AUTH_mtbs/Raster/MTBS_BurnSeverityData/Collection2/ingestedl3data/{year}/{event_id}/{event_id}_mtbs_burn_severity.tif",
    # OpenTopography-like path (guess)
    "https://opentopography.s3.sdsc.edu/mtbs/{year}/{event_id}/{event_id}_burn_severity.tif",
    # Another guess (generic)
    "https://s3.amazonaws.com/mtbs/Collection2/{year}/{event_id}/{event_id}_burn_severity.tif",
]


def try_download(event_id, year):
    for t in TEMPLATES:
        url = t.format(event_id=event_id, year=year)
        out_path = OUT / f"{event_id}_{year}_mtbs_guess.tif"
        try:
            print('Trying', url)
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 1024:
                out_path.write_bytes(r.content)
                print('Saved to', out_path)
                return out_path
        except Exception as e:
            print('Error', e)
    return None


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: fetch_mtbs_mirrors.py <EVENT_ID> <YEAR>')
        sys.exit(1)
    eid = sys.argv[1]
    yr = sys.argv[2]
    p = try_download(eid, yr)
    if p is None:
        print('No mirror found for', eid)
    else:
        print('Downloaded:', p)
