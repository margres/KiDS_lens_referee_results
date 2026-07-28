"""
Fetch KiDS DR4 g/u/i-band tile FITS files (r-band already covered by
fetch_cutouts.py) and extract per-target cutouts, for the 223 known+new
grade A/B targets -- prep work for a possible future multi-band fit
(color helps separate lens/source/interloper light, see
NOTES_ring_residual_issue.md discussion). Manifest is the *public*, clean
KiDS DR4 "ugri Coadds" batch-download script (predictable
KiDS_DR4.0_{RA}_{DEC}_{band}_sci.fits URLs, no opaque per-file hash, unlike
the r-band "det_sci" manifest) -- see kids.strw.leidenuniv.nl/DR4/access.php.

Targets don't have a KIDS_TILE column (unlike targets.csv) since they come
from known_lenses_AB.csv / KiDS_SGL_candidates.csv (RA/DEC only) -- tile
assignment is by nearest tile-grid-centre match, band-independent (same
grid every band).

Usage: python3 fetch_multiband_cutouts.py --band g
       python3 fetch_multiband_cutouts.py --band u
       python3 fetch_multiband_cutouts.py --band i
"""
import os, re, time, subprocess, json, argparse
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u

HERE = os.path.dirname(os.path.abspath(__file__))
CUTOUT_SIZE = 151

parser = argparse.ArgumentParser()
parser.add_argument('--band', required=True, choices=['u', 'g', 'i'])
parser.add_argument('--limit', type=int, default=None)
args = parser.parse_args()

OUTDIR = os.path.join(HERE, f'lens_cutouts_fits_{args.band}')
os.makedirs(OUTDIR, exist_ok=True)
LOG = os.path.join(HERE, f'fetch_log_{args.band}.txt')
DONE_TILES_PATH = os.path.join(HERE, f'done_tiles_{args.band}.json')


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + "\n")


known = pd.read_csv(os.path.join(HERE, '..', 'referee_revisions', 'known_lenses_AB.csv'))
known = known.rename(columns={'RA': 'RAJ2000', 'DEC': 'DECJ2000'})[['ID', 'RAJ2000', 'DECJ2000']]
cand = pd.read_csv(os.path.join(HERE, '..', 'referee_revisions', 'KiDS_SGL_candidates.csv'))
cand = cand[cand['Grade'].isin(['A', 'B'])].copy()
cand['ID'] = 'KiDSDR4 ' + cand['KiDS_ID'].astype(str)
cand = cand.rename(columns={'RA': 'RAJ2000', 'DEC': 'DECJ2000'})[['ID', 'RAJ2000', 'DECJ2000']]

targets = pd.concat([known, cand], ignore_index=True)
targets['safe_id'] = targets['ID'].str.replace(' ', '_').str.replace('/', '_')
targets = targets.drop_duplicates(subset='safe_id').reset_index(drop=True)
if args.limit is not None:
    targets = targets.iloc[:args.limit]

# Parse the clean public manifest: KiDS_DR4.0_{ra}_{dec}_{band}_sci.fits
url_map = {}
tile_centres = []
with open(os.path.join(HERE, 'kids_dr4.0_sci_wget.sh')) as f:
    for line in f:
        m = re.search(rf'wget (\S+/(KiDS_DR4\.0_(\S+?)_(\S+?)_{args.band}_sci\.fits))', line)
        if m:
            url, fname, ra_s, dec_s = m.group(1), m.group(2), m.group(3), m.group(4)
            ra, dec = float(ra_s), float(dec_s)
            url_map[(ra, dec)] = url
            tile_centres.append((ra, dec))

log(f"Loaded {len(url_map)} '{args.band}'-band tile URLs from public manifest")

tile_centres = np.array(tile_centres)


def nearest_tile(ra, dec):
    d = np.hypot(tile_centres[:, 0] - ra, tile_centres[:, 1] - dec)
    idx = np.argmin(d)
    return tuple(tile_centres[idx])


targets['tile'] = [nearest_tile(r, d) for r, d in zip(targets['RAJ2000'], targets['DECJ2000'])]
log(f"{targets['tile'].nunique()} unique tiles needed for {len(targets)} targets")

done_tiles = json.load(open(DONE_TILES_PATH)) if os.path.exists(DONE_TILES_PATH) else {}


def save_done():
    json.dump({str(k): v for k, v in done_tiles.items()}, open(DONE_TILES_PATH, 'w'))


def robust_download(url, dest, max_attempts=80, per_attempt_timeout=25):
    if os.path.exists(dest):
        os.remove(dest)
    for attempt in range(1, max_attempts + 1):
        subprocess.run(
            ['wget', '-q', f'--timeout={per_attempt_timeout}', '-c', url, '-O', dest],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        sz = os.path.getsize(dest) if os.path.exists(dest) else 0
        if sz > 0:
            try:
                with fits.open(dest) as hdul:
                    _ = hdul[0].data.shape
                return True, attempt, sz
            except Exception:
                pass
    return False, max_attempts, os.path.getsize(dest) if os.path.exists(dest) else 0


def process_tile(tile, rows):
    tile_key = str(tile)
    if tile_key in done_tiles:
        return
    if tile not in url_map:
        log(f"{tile}: NO URL, skipping")
        done_tiles[tile_key] = 'no_url'
        save_done()
        return

    tmp_path = os.path.join(HERE, f'_tmp_tile_{args.band}.fits')
    t0 = time.time()
    ok, attempts, sz = robust_download(url_map[tile], tmp_path)
    dt = time.time() - t0

    if not ok:
        log(f"{tile}: FAILED after {attempts} attempts, {sz/1e6:.1f}MB, {dt:.0f}s")
        done_tiles[tile_key] = 'failed'
        save_done()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return

    try:
        with fits.open(tmp_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            wcs = WCS(header)

        n_saved = 0
        for _, row in rows.iterrows():
            try:
                coord = SkyCoord(ra=row['RAJ2000'] * u.deg, dec=row['DECJ2000'] * u.deg)
                cut = Cutout2D(data, coord, size=CUTOUT_SIZE, wcs=wcs, mode='partial', fill_value=np.nan)
                out_path = os.path.join(OUTDIR, f"{row['safe_id']}.fits")
                hdu = fits.PrimaryHDU(data=cut.data, header=cut.wcs.to_header())
                hdu.writeto(out_path, overwrite=True)
                n_saved += 1
            except Exception as e:
                log(f"   cutout failed for {row['ID']}: {e}")

        log(f"{tile}: OK, {attempts} attempts, {sz/1e6:.0f}MB, {dt:.0f}s, saved {n_saved}/{len(rows)} cutouts")
        done_tiles[tile_key] = 'ok'
    except Exception as e:
        log(f"{tile}: FITS read/cutout error: {e}")
        done_tiles[tile_key] = 'error'
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        save_done()


tile_groups = {tile: targets[targets['tile'] == tile] for tile in targets['tile'].unique()}

t_start = time.time()
for i, tile in enumerate(tile_groups, 1):
    process_tile(tile, tile_groups[tile])
    if i % 20 == 0:
        elapsed = time.time() - t_start
        log(f"PROGRESS: {i}/{len(tile_groups)} tiles processed, {elapsed/60:.1f} min elapsed")

log(f"ALL '{args.band}'-BAND TILES PROCESSED")
n_ok = sum(1 for v in done_tiles.values() if v == 'ok')
n_fail = sum(1 for v in done_tiles.values() if v in ('failed', 'error', 'no_url'))
log(f"FINAL SUMMARY: {n_ok} ok, {n_fail} failed/no_url out of {len(tile_groups)} tiles, total time {(time.time()-t_start)/60:.1f} min")
