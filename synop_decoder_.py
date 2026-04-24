#!/usr/bin/env python3
"""
SYNOP Decoder
=============
Decodes WMO SYNOP (FM-12) reports from a CSV input file.

Input CSV columns : WMOIND, YEAR, MONTH, DAY, HOUR, MIN, REPORT
Output CSV columns: station_id, datetime, lat, lon, temp, pressure,
                    humidity, wind_speed, wind_dir, cloud_cover,
                    visibility, rain_3h

Usage
-----
    python synop_decoder.py input.csv output.csv

Optional flags
--------------
    --sep <char>    Column separator of the input file  (default: auto-detect)
    --lat-lon <file> Path to a WMO station lat/lon CSV (columns: wmo_id,lat,lon)
                    Falls back to None if the file is absent or the station
                    is not listed.
"""

import csv
import sys
import argparse
import os
import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Optional: small bundled lookup table for common WMO stations.
# You can replace / extend this with a full station list CSV via --lat-lon.
# ---------------------------------------------------------------------------
STATION_COORDS: dict[str, tuple[float | None, float | None]] = {
    # fmt: off
    # WMO-ID : (lat, lon)
    "98444": (13.15,  123.73),   # Legaspi, Philippines (example)
    # Add more as needed …
    # fmt: on
}


# ---------------------------------------------------------------------------
# SYNOP group decoders
# ---------------------------------------------------------------------------

def decode_irix(group: str) -> dict:
    """Section 0 indicator group  iRiXhVV  (5 chars)."""
    out = {}
    if len(group) != 5:
        return out
    # iR – precipitation indicator  (ignored for output)
    # iX – station type / present-weather indicator  (ignored)
    # h  – height of cloud base (code table 1600)
    h_code = group[2]
    h_table = {"0": 0, "1": 50, "2": 100, "3": 200, "4": 300,
               "5": 600, "6": 1000, "7": 1500, "8": 2000, "9": 2500, "/": None}
    out["cloud_base_m"] = h_table.get(h_code)
    # VV – visibility (code table 4377)
    vv = group[3:5]
    out["visibility_m"] = _decode_vv(vv)
    return out


def _decode_vv(vv: str) -> float | None:
    """Convert VV code (00-99) to metres."""
    if "/" in vv:
        return None
    v = int(vv)
    if 0 <= v <= 50:
        return v * 100           # 00-50 → 0 … 5 000 m
    elif 51 <= v <= 80:
        return (v - 50) * 1000 + 5000   # 51-80 → 6 000 … 35 000 m
    elif 81 <= v <= 89:
        return (v - 80) * 5000 + 35000  # 81-89 → 40 000 … 80 000 m
    elif v == 90:
        return None   # < 50 m
    elif v == 99:
        return None   # sky obscured
    return None


def decode_nddff(group: str) -> dict:
    """Cloud / wind group  Nddff  (5 chars)."""
    out = {}
    if len(group) != 5:
        return out
    # N – total cloud cover (oktas 0-8, / = obscured)
    n = group[0]
    if n == "/":
        out["cloud_cover"] = None
    elif n.isdigit():
        out["cloud_cover"] = int(n)   # oktas
    # dd – wind direction in tens of degrees (00-36, 00=calm, 99=variable)
    dd = group[1:3]
    if dd.isdigit():
        dd_val = int(dd)
        out["wind_dir"] = None if dd_val == 99 else dd_val * 10
    # ff – wind speed (m/s or knots depending on iW; assume m/s)
    ff = group[3:5]
    if ff.isdigit():
        out["wind_speed"] = int(ff)
    return out


def decode_1sTTT(group: str) -> dict:
    """Air temperature group  1SnTTT."""
    out = {}
    if len(group) != 5 or group[0] != "1":
        return out
    sign = group[1]
    ttt = group[2:5]
    if ttt.isdigit() and sign in ("0", "1"):
        t = int(ttt) / 10.0
        out["temp"] = -t if sign == "1" else t
    return out


def decode_2sTdTdTd(group: str) -> dict:
    """Dew-point group  2SnTdTdTd → derive relative humidity."""
    out = {}
    if len(group) != 5 or group[0] != "2":
        return out
    sign = group[1]
    ttt = group[2:5]
    if ttt.isdigit() and sign in ("0", "1"):
        td = int(ttt) / 10.0
        if sign == "1":
            td = -td
        out["dewpoint"] = td
    return out


def decode_3PPPP(group: str) -> dict:
    """Station pressure group  3PPPP."""
    out = {}
    if len(group) != 5 or group[0] != "3":
        return out
    pppp = group[1:5]
    if pppp.isdigit():
        p = int(pppp) / 10.0
        # Values below 100 are in hundreds (e.g. 0980 = 980.0 hPa)
        # Convention: if p < 100.0 → add 1000
        if p < 100.0:
            p += 1000.0
        out["pressure_station"] = p
    return out


def decode_4PPPP(group: str) -> dict:
    """Sea-level pressure group  4PPPP."""
    out = {}
    if len(group) != 5 or group[0] != "4":
        return out
    pppp = group[1:5]
    if pppp.isdigit():
        p = int(pppp) / 10.0
        if p < 100.0:
            p += 1000.0
        out["pressure"] = p
    return out


def decode_6RRRt(group: str) -> dict:
    """Precipitation group  6RRRt."""
    out = {}
    if len(group) != 5 or group[0] != "6":
        return out
    rrr = group[1:4]
    t   = group[4]          # duration indicator
    if rrr.isdigit():
        r = int(rrr)
        if r == 990:
            rain = 0.0      # trace
        elif r >= 991:
            rain = (r - 990) / 10.0
        else:
            rain = float(r)
        # Map duration to 3-hour bucket (t=1→6h, t=2→12h, t=3→18h, t=4→24h,
        # t=5→1h, t=6→2h, t=7→3h, t=8→9h, t=9→15h)
        out["rain_duration_code"] = t
        # We store the raw amount; the column is named rain_3h by convention
        out["rain_3h"] = rain
    return out


def decode_7wwW1W2(group: str) -> dict:
    """Present/past weather group  7wwW1W2  (informational, not in output)."""
    return {}


def decode_8NhClCmCh(group: str) -> dict:
    """Cloud group  8NhClCmCh."""
    out = {}
    if len(group) != 5 or group[0] != "8":
        return out
    # Nh – cloud cover of low / middle cloud (oktas)
    nh = group[1]
    if nh.isdigit():
        out["cloud_cover_low"] = int(nh)
    return out


# ---------------------------------------------------------------------------
# Humidity from temperature and dew-point (Magnus formula)
# ---------------------------------------------------------------------------

def relative_humidity(T: float, Td: float) -> float:
    """Approximate RH (%) from temperature and dew-point (°C)."""
    return 100.0 * (
        (17.625 * Td) / (243.04 + Td) - (17.625 * T) / (243.04 + T)
    )
    # Simpler approximation: RH ≈ 100 - 5*(T - Td)
    # We use the Magnus formula for better accuracy.


def _rh_magnus(T: float, Td: float) -> float:
    import math
    a, b = 17.625, 243.04
    gamma_T  = a * T  / (b + T)
    gamma_Td = a * Td / (b + Td)
    return round(100.0 * math.exp(gamma_Td - gamma_T), 1)


# ---------------------------------------------------------------------------
# Main SYNOP parser
# ---------------------------------------------------------------------------

def parse_synop(report: str) -> dict:
    """
    Parse a single SYNOP FM-12 report string.
    Returns a dict with decoded fields (None where not reported).
    """
    result = {
        "temp": None,
        "pressure": None,
        "humidity": None,
        "wind_speed": None,
        "wind_dir": None,
        "cloud_cover": None,
        "visibility_m": None,
        "rain_3h": None,
    }

    # Clean up and split
    report = report.strip().rstrip("=").strip()
    tokens = report.split()

    if not tokens:
        return result

    # Skip section header tokens (AAXX, BBXX, OOXX) and YYGGiw
    idx = 0
    if tokens[idx] in ("AAXX", "BBXX", "OOXX"):
        idx += 1   # skip section id
    if idx < len(tokens):
        idx += 1   # skip YYGGiw (day/hour/wind-indicator)
    if idx < len(tokens):
        idx += 1   # skip station number (IIiii)

    # Parse remaining 5-char groups in section 1
    # Stop at section 2 / 3 markers (222, 333, 444, 555)
    dewpoint = None
    while idx < len(tokens):
        g = tokens[idx]
        idx += 1

        if g in ("222", "333", "444", "555"):
            break

        if len(g) != 5:
            continue

        lead = g[0]

        if lead == "0":                          # iRiXhVV
            d = decode_irix(g)
            result["visibility_m"] = d.get("visibility_m")

        elif lead == "N" or (lead.isdigit() and g[1:3].isdigit() and g[3:5].isdigit() and lead in "012345678/"):
            # Nddff – must be first group after station id
            # We try to match Nddff heuristically:
            # N is 0-9 or /, dd is 00-99, ff is 00-99
            # We already consumed iRiXhVV above; this should be next
            if lead in "012345678/":
                dd = g[1:3]
                ff = g[3:5]
                if (dd.isdigit() or dd == "//") and (ff.isdigit() or ff == "//"):
                    d = decode_nddff(g)
                    result.update({k: v for k, v in d.items() if k in result})

        if lead == "1":
            d = decode_1sTTT(g)
            if "temp" in d:
                result["temp"] = d["temp"]

        elif lead == "2":
            d = decode_2sTdTdTd(g)
            if "dewpoint" in d:
                dewpoint = d["dewpoint"]

        elif lead == "3":
            d = decode_3PPPP(g)
            # Only use station pressure if sea-level not yet found
            if "pressure_station" in d and result["pressure"] is None:
                result["pressure"] = d["pressure_station"]

        elif lead == "4":
            d = decode_4PPPP(g)
            if "pressure" in d:
                result["pressure"] = d["pressure"]

        elif lead == "5":
            pass  # pressure tendency – skip

        elif lead == "6":
            d = decode_6RRRt(g)
            if "rain_3h" in d:
                result["rain_3h"] = d["rain_3h"]

        elif lead == "7":
            pass  # present weather – skip

        elif lead == "8":
            d = decode_8NhClCmCh(g)
            # Use low-cloud oktas as cloud_cover if not yet set
            if result["cloud_cover"] is None and "cloud_cover_low" in d:
                result["cloud_cover"] = d["cloud_cover_low"]

        elif lead == "9":
            pass  # additional data – skip

    # Derive humidity
    if result["temp"] is not None and dewpoint is not None:
        result["humidity"] = _rh_magnus(result["temp"], dewpoint)

    return result


# ---------------------------------------------------------------------------
# Nddff group needs a dedicated second pass because the lead digit is
# ambiguous (it can be 0-8 or /).  Re-implement cleanly.
# ---------------------------------------------------------------------------

def _extract_nddff(tokens: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Find and decode the Nddff group.  In FM-12 Section 1 it is always
    the FIRST 5-char data group after the station number.
    Returns (decoded_dict, consumed_index).
    """
    out = {}
    if start_idx >= len(tokens):
        return out, start_idx
    g = tokens[start_idx]
    if len(g) == 5:
        n, dd, ff = g[0], g[1:3], g[3:5]
        n_ok  = n  in "012345678/"
        dd_ok = dd.isdigit() or dd == "//"
        ff_ok = ff.isdigit() or ff == "//"
        if n_ok and dd_ok and ff_ok:
            if n.isdigit():
                out["cloud_cover"] = int(n)
            if dd.isdigit():
                dd_val = int(dd)
                out["wind_dir"] = None if dd_val == 99 else dd_val * 10
            if ff.isdigit():
                out["wind_speed"] = int(ff)
            return out, start_idx + 1
    return out, start_idx


def parse_synop_v2(report: str) -> dict:
    """
    Improved two-pass SYNOP parser.
    """
    result = {
        "temp": None,
        "pressure": None,
        "humidity": None,
        "wind_speed": None,
        "wind_dir": None,
        "cloud_cover": None,
        "visibility_m": None,
        "rain_3h": None,
    }

    report = report.strip().rstrip("=").strip()
    tokens = report.split()
    if not tokens:
        return result

    idx = 0
    # Skip AAXX / BBXX / OOXX
    if idx < len(tokens) and tokens[idx] in ("AAXX", "BBXX", "OOXX"):
        idx += 1
    # Skip YYGGiw
    if idx < len(tokens):
        idx += 1
    # Skip IIiii (station number)
    if idx < len(tokens):
        idx += 1

    # ---- group 0: iRiXhVV
    # This group is ALWAYS present in FM-12 land stations and comes first.
    # It has the form iRiXhVV where iR∈{0-4}, iX∈{1-7}, h∈{0-9/}, VV∈{00-99}.
    # We simply consume the first 5-char group as iRiXhVV, then the next as Nddff.
    if idx < len(tokens) and len(tokens[idx]) == 5:
        d = decode_irix(tokens[idx])
        result["visibility_m"] = d.get("visibility_m")
        idx += 1

    # ---- group Nddff
    nd, idx = _extract_nddff(tokens, idx)
    result.update({k: v for k, v in nd.items() if k in result})

    # ---- remaining groups
    dewpoint = None
    while idx < len(tokens):
        g = tokens[idx]
        idx += 1

        if g in ("222", "333", "444", "555"):
            break
        if len(g) != 5:
            continue

        lead = g[0]
        if not lead.isdigit():
            continue

        if lead == "1":
            d = decode_1sTTT(g)
            if "temp" in d:
                result["temp"] = d["temp"]

        elif lead == "2":
            d = decode_2sTdTdTd(g)
            if "dewpoint" in d:
                dewpoint = d["dewpoint"]

        elif lead == "3":
            d = decode_3PPPP(g)
            if "pressure_station" in d and result["pressure"] is None:
                result["pressure"] = d["pressure_station"]

        elif lead == "4":
            d = decode_4PPPP(g)
            if "pressure" in d:
                result["pressure"] = d["pressure"]

        elif lead == "6":
            d = decode_6RRRt(g)
            if "rain_3h" in d:
                result["rain_3h"] = d["rain_3h"]

        elif lead == "8":
            d = decode_8NhClCmCh(g)
            if result["cloud_cover"] is None and "cloud_cover_low" in d:
                result["cloud_cover"] = d["cloud_cover_low"]

    # Derive humidity
    if result["temp"] is not None and dewpoint is not None:
        result["humidity"] = _rh_magnus(result["temp"], dewpoint)

    return result


# ---------------------------------------------------------------------------
# Station coordinate lookup
# ---------------------------------------------------------------------------

def load_station_coords(path: str) -> dict[str, tuple[float, float]]:
    coords = {}
    if not path or not os.path.isfile(path):
        return coords
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wmo = str(row.get("wmo_id", "")).strip()
            try:
                lat = float(row.get("lat", ""))
                lon = float(row.get("lon", ""))
                coords[wmo] = (lat, lon)
            except (ValueError, TypeError):
                pass
    return coords


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

OUTPUT_COLS = [
    "station_id", "datetime",
    "lat", "lon",
    "temp", "pressure", "humidity",
    "wind_speed", "wind_dir",
    "cloud_cover", "visibility_m",
    "rain_3h",
]


def detect_separator(path: str) -> str:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
    counts = {s: sample.count(s) for s in (",", ";", "\t", "|")}
    return max(counts, key=counts.get)


def process(input_path: str, output_path: str,
            sep: str | None = None,
            station_coords_path: str | None = None) -> int:

    ext_coords = load_station_coords(station_coords_path) if station_coords_path else {}

    if sep is None:
        sep = detect_separator(input_path)

    decoded_rows = 0
    errors = 0

    with open(input_path, newline="", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin, delimiter=sep)
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()

        for lineno, row in enumerate(reader, start=2):   # 2 = first data line
            try:
                # ---- identifiers
                station_id = str(row.get("WMOIND", row.get("wmoind", ""))).strip()
                year  = row.get("YEAR",  row.get("year",  "")).strip()
                month = row.get("MONTH", row.get("month", "")).strip()
                day   = row.get("DAY",   row.get("day",   "")).strip()
                hour  = row.get("HOUR",  row.get("hour",  "")).strip()
                minute = row.get("MIN",  row.get("min",   "0")).strip()
                report = row.get("REPORT", row.get("report", "")).strip()

                # Build datetime
                try:
                    dt = datetime(
                        int(year), int(month), int(day),
                        int(hour), int(minute or "0")
                    )
                    dt_str = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    dt_str = f"{year}-{month}-{day} {hour}:{minute}"

                # Coordinates
                lat, lon = ext_coords.get(station_id,
                           STATION_COORDS.get(station_id, (None, None)))

                # Decode SYNOP
                decoded = parse_synop_v2(report)

                out_row = {
                    "station_id": station_id,
                    "datetime":   dt_str,
                    "lat":        lat,
                    "lon":        lon,
                    "temp":       decoded["temp"],
                    "pressure":   decoded["pressure"],
                    "humidity":   decoded["humidity"],
                    "wind_speed": decoded["wind_speed"],
                    "wind_dir":   decoded["wind_dir"],
                    "cloud_cover": decoded["cloud_cover"],
                    "visibility_m": decoded["visibility_m"],
                    "rain_3h":    decoded["rain_3h"],
                }
                writer.writerow(out_row)
                decoded_rows += 1

            except Exception as exc:
                errors += 1
                print(f"  [WARN] line {lineno}: {exc}", file=sys.stderr)

    return decoded_rows, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decode SYNOP FM-12 reports from a CSV file."
    )
    parser.add_argument("input",  help="Input CSV file path")
    parser.add_argument("output", help="Output CSV file path")
    parser.add_argument("--sep",  default=None,
                        help="Column separator (auto-detected if omitted)")
    parser.add_argument("--lat-lon", dest="latlon", default=None,
                        help="Optional WMO station lat/lon CSV (columns: wmo_id,lat,lon)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Decoding {args.input}  →  {args.output}")
    n, err = process(args.input, args.output,
                     sep=args.sep, station_coords_path=args.latlon)
    print(f"Done. {n} rows decoded, {err} errors.")


if __name__ == "__main__":
    main()
