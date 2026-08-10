"""
FCL Workload Dashboard — Streamlit edition.


"""

import json
import math
import os
import re
import tempfile
import time
from io import BytesIO
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import truststore
    truststore.inject_into_ssl()
except Exception as exc:
    print(f"[SSL setup] truststore unavailable: {exc}")

try:
    import gspread
    from gspread.utils import rowcol_to_a1
    from google.oauth2.service_account import Credentials
    SHEETS_AVAILABLE = True
except Exception as exc:
    print(f"[Google Sheets setup] Google Sheets libraries unavailable: {exc}")
    SHEETS_AVAILABLE = False

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportLabImage,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

st.set_page_config(
    page_title="FCL Workload Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _secret(name, default=""):
    """Read a top-level Streamlit secret or environment variable safely."""
    env_value = os.getenv(name)
    if env_value not in (None, ""):
        return env_value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# ── CONFIG ──────────────────────────────────────────────────────────────────
GOOGLE_SHEET_ID = str(_secret("GOOGLE_SHEET_ID", "")).strip()
SHEET_TAB_NAME = str(_secret("RAW_SESSIONS_TAB", "Raw Sessions")).strip() or "Raw Sessions"
ROSTER_SHEET_TAB_NAME = str(_secret("ROSTER_TAB", "Position Groups")).strip() or "Position Groups"

BASE_URL = "https://statsportsproseries.com"
API_VERSION = "7"
API_KEY = str(_secret("STATSPORTS_API_KEY", "")).strip()
THIRD_PARTY_ID = API_KEY

REQUEST_TIMEOUT = 90
SLEEP_BETWEEN_DAYS = 0.10
MIN_DURATION_MIN = 0.25
NUMERIC_TOLERANCE = 0.001
KEY_COLS = ["date", "session_name", "player_name", "drill_name"]

FCL_ONLY = True
FCL_PATTERN = re.compile(r"(?:\bfcl\b|florida\s+complex(?:\s+league)?)", re.IGNORECASE)
FCL_SESSION_SCOPE_FIELDS = (
    "sessionName", "teamName", "team", "teamDetails", "sessionTeam",
    "squad", "squadName", "group", "groupName", "location", "sessionType",
)

EXCLUDED_DRILL_PREFIX = "B" + "irch -"

POSITION_GROUPS = [
    "Pitchers", "Catchers", "Infielders", "Outfielders", "Two-way", "Unassigned",
]
POSITION_GROUP_OPTIONS = [{"label": group, "value": group} for group in POSITION_GROUPS]

# Local fallback only. Streamlit Cloud should use the shared Google Sheet roster tab.
ROSTER_ASSIGNMENTS_FILE = os.path.join(
    os.path.expanduser("~"), ".fcl_workload_dashboard", "position_group_assignments.json"
)

REPORT_DAY_TYPES = ["Not set", "High", "Low"]
REPORT_POSITION_ORDER = ["Infielders", "Outfielders", "Catchers", "Pitchers", "Two-way", "Unassigned"]
HSR_TARGETS = {
    "Infielders": {
        "High": {"hsr_operator": ">", "hsr_value": 80.0, "accel_operator": ">=", "accel_value": 10.0, "logic": "or"},
        "Low": {"hsr_operator": "<", "hsr_value": 30.0, "accel_operator": "<", "accel_value": 8.0, "logic": "and"},
    },
    "Outfielders": {
        "High": {"hsr_operator": ">", "hsr_value": 100.0, "accel_operator": ">=", "accel_value": 10.0, "logic": "or"},
        "Low": {"hsr_operator": "<", "hsr_value": 30.0},
    },
    "Catchers": {
        "High": {"hsr_operator": ">", "hsr_value": 50.0},
        "Low": {"hsr_operator": "<", "hsr_value": 30.0},
    },
}

C_BG = "#F4F6FA"
C_WHITE = "#FFFFFF"
C_BORDER = "#E2E8F0"
C_RED = "#C8102E"
C_NAVY = "#041E42"
C_TEXT = "#1A2035"
C_MUTED = "#64748B"
C_GREEN = "#16A34A"
C_AMBER = "#D97706"
C_BLUE = "#2563EB"

NON_FIELD_PREFIXES = (
    "entire session", "lift", "cages", "isd", "activation",
    "dynamic", "warm up", "warm-up",
)


def _google_service_account_info():
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:
        pass
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        return json.loads(raw)
    return None


def get_sheets_client():
    if not SHEETS_AVAILABLE:
        raise RuntimeError("Google Sheets packages are not installed.")
    if not GOOGLE_SHEET_ID:
        raise RuntimeError("GOOGLE_SHEET_ID is missing from Streamlit Secrets.")
    info = _google_service_account_info()
    if not info:
        raise RuntimeError("[gcp_service_account] is missing from Streamlit Secrets.")
    credentials = Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(credentials)


_http = requests.Session()
_http.headers.update({
    "Internal": API_KEY,
    "api-version": API_VERSION,
    "Content-Type": "application/json",
})
_http.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1.2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
    raise_on_status=False,
)))


WEEKLY_METRICS = {
    "avg_player_hsr_m": {"label": "Avg player daily HSR", "axis": "HSR (m)", "digits": 0},
    "avg_player_accels": {"label": "Avg player daily accelerations", "axis": "Accelerations", "digits": 0},
    "avg_player_total_distance_m": {"label": "Avg player daily total distance", "axis": "Distance (m)", "digits": 0},
    "avg_player_hmld_m": {"label": "Avg player daily HMLD", "axis": "HMLD (m)", "digits": 0},
    "avg_player_sprint_distance_m": {"label": "Avg player daily sprint distance", "axis": "Sprint distance (m)", "digits": 0},
    "avg_player_mechanical_load": {"label": "Avg player daily mechanical load", "axis": "Mechanical load", "digits": 0},
    "avg_player_duration_min": {"label": "Avg player daily field duration", "axis": "Minutes", "digits": 0},
    "avg_player_hsr_per_min": {"label": "Avg player daily HSR / min", "axis": "HSR / min", "digits": 2},
    "avg_player_accels_per_min": {"label": "Avg player daily accelerations / min", "axis": "Accelerations / min", "digits": 2},
    "avg_player_top_speed_ms": {"label": "Avg player daily max speed", "axis": "Max speed (m/s)", "digits": 2},
    "avg_player_max_accel_ms2": {"label": "Avg player daily max acceleration", "axis": "Max acceleration (m/s²)", "digits": 2},
}
WEEKLY_DEFAULT_METRICS = [
    "avg_player_hsr_m", "avg_player_accels", "avg_player_hsr_per_min", "avg_player_accels_per_min",
]

def api_post(endpoint, payload):
    """Return useful diagnostics rather than flattening all failures into HTTP 0."""
    url = f"{BASE_URL}{endpoint}"
    try:
        response = _http.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {
                "error_type": "NON_JSON_RESPONSE",
                "detail": response.text[:750],
                "url": url,
            }
    except requests.exceptions.Timeout as exc:
        return 0, {"error_type": "TIMEOUT", "detail": repr(exc), "url": url}
    except requests.exceptions.SSLError as exc:
        return 0, {"error_type": "SSL_ERROR", "detail": repr(exc), "url": url}
    except requests.exceptions.ConnectionError as exc:
        return 0, {"error_type": "CONNECTION_ERROR", "detail": repr(exc), "url": url}
    except requests.exceptions.RequestException as exc:
        return 0, {"error_type": type(exc).__name__, "detail": repr(exc), "url": url}


def player_name(session_player):
    details = session_player.get("playerDetails") or {}
    for key in ("displayName", "name", "fullName"):
        value = details.get(key)
        if value:
            return str(value).strip()

    first_name = details.get("firstName", "")
    last_name = details.get("lastName", "")
    if first_name or last_name:
        return f"{first_name} {last_name}".strip()

    return f"Player {session_player.get('id', '?')}"


def player_position(session_player):
    """Read the most useful position label exposed by STATSports, when available."""
    details = session_player.get("playerDetails") or {}
    fields = (
        "position", "positionName", "primaryPosition", "primaryPositionName",
        "playerPosition", "role", "sportPosition", "positionCode",
    )
    for source in (details, session_player):
        if not isinstance(source, dict):
            continue
        for field in fields:
            value = source.get(field)
            if isinstance(value, dict):
                value = " ".join(str(v) for v in value.values() if v is not None)
            elif isinstance(value, (list, tuple, set)):
                value = " ".join(str(v) for v in value if v is not None)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def infer_position_group(position_raw):
    """Map common baseball labels to a coach-editable workload group."""
    raw = str(position_raw or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", " ", raw)
    tokens = set(compact.split())
    if not raw:
        return "Unassigned"
    if any(token in tokens for token in {"p", "sp", "rp", "lhp", "rhp"}) or "pitch" in raw:
        return "Pitchers"
    if "catch" in raw or tokens == {"c"} or "catcher" in tokens:
        return "Catchers"
    if any(token in tokens for token in {"ss", "1b", "2b", "3b", "if", "inf"}) or "infield" in raw or "shortstop" in raw:
        return "Infielders"
    if any(token in tokens for token in {"of", "lf", "cf", "rf"}) or "outfield" in raw:
        return "Outfielders"
    if "two way" in compact or "utility" in raw or "twp" in tokens:
        return "Two-way"
    return "Unassigned"


def kpi(drill, field, cast=None):
    value = (drill.get("drillKpi") or {}).get(field)
    if cast:
        try:
            return cast(value)
        except Exception:
            return np.nan

    try:
        numeric = float(value)
        return np.nan if math.isnan(numeric) else numeric
    except Exception:
        return np.nan


def safe_divide(numerator, denominator):
    try:
        numerator = float(numerator)
        denominator = float(denominator)
        if denominator <= 0 or math.isnan(numerator) or math.isnan(denominator):
            return np.nan
        return numerator / denominator
    except Exception:
        return np.nan


def is_excluded_drill(drill_name):
    return str(drill_name or "").lstrip().lower().startswith(EXCLUDED_DRILL_PREFIX.lower())


def clean_drill_name(drill_name):
    """Normalize display spacing for drills retained by this app."""
    return str(drill_name or "").strip()


def flatten_scope_text(value):
    """Convert a small session metadata value into searchable text."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(flatten_scope_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(flatten_scope_text(item) for item in value)
    return str(value)


def session_scope_text(session):
    """Collect likely session-level team/group labels without reading athlete fields."""
    return " ".join(
        flatten_scope_text(session.get(field))
        for field in FCL_SESSION_SCOPE_FIELDS
        if session.get(field) is not None
    )


def is_fcl_session(session):
    """True only when session-level naming/metadata identifies the FCL."""
    return bool(FCL_PATTERN.search(session_scope_text(session)))


def pull_day(day_string):
    """
    Pull FCL drills for one day. The excluded planning-prefix family and non-FCL
    sessions are removed before any row is created.
    """
    status, data = api_post(
        "/thirdpartyapi/api/thirdPartyData/getFullSessionsByDateRange",
        {
            "thirdPartyApiId": THIRD_PARTY_ID,
            "sessionStartDate": f"{day_string}T00:00:00",
            "sessionEndDate": f"{day_string}T23:59:59",
        },
    )

    if status != 200:
        if isinstance(data, dict):
            error_type = data.get("error_type", "API_ERROR")
            detail = str(data.get("detail", data))[:750]
        else:
            error_type = "API_ERROR"
            detail = str(data)[:750]
        print(f"  WARNING {day_string}: HTTP {status} | {error_type} | {detail}")
        return []

    rows = []
    day_value = date.fromisoformat(day_string)

    for session in (data if isinstance(data, list) else [data]):
        if not isinstance(session, dict):
            continue

        session_name = str(session.get("sessionName") or "")
        session_id = session.get("id") or session.get("sessionId") or ""
        scope_text = session_scope_text(session)

        if FCL_ONLY and not is_fcl_session(session):
            continue

        for session_player in (session.get("sessionPlayers") or session.get("players") or []):
            if not isinstance(session_player, dict):
                continue

            name = player_name(session_player)
            player_id = session_player.get("id") or session_player.get("playerId") or ""
            position_raw = player_position(session_player)
            position_group_default = infer_position_group(position_raw)
            drill_occurrences = {}

            for sequence, drill in enumerate(session_player.get("drills") or [], start=1):
                if not isinstance(drill, dict):
                    continue

                raw_name = str(drill.get("drillName") or "").strip()
                if not raw_name:
                    raw_name = "Unnamed Drill"

                # Excluded planning-prefix drills are removed at the raw-pull level.
                # They never enter charts, summaries, comparisons, or sheet sync.
                if is_excluded_drill(raw_name):
                    continue

                normalized_raw = raw_name.strip().lower()
                drill_occurrences[normalized_raw] = drill_occurrences.get(normalized_raw, 0) + 1
                occurrence = drill_occurrences[normalized_raw]

                duration_min = kpi(
                    drill,
                    "totalTime",
                    lambda value: float(value) / 60 if value is not None else np.nan,
                )
                hsr_distance = kpi(drill, "highSpeedRunningRel")
                accelerations = kpi(
                    drill,
                    "accelerationsRel",
                    lambda value: int(float(value)) if value is not None else np.nan,
                )

                display_name = clean_drill_name(raw_name)
                block_key = f"{raw_name}||occ={occurrence}"

                row = {
                    "date": day_string,
                    "week": f"{day_value.isocalendar()[0]}-W{day_value.isocalendar()[1]:02d}",
                    "week_start": (day_value - timedelta(days=day_value.weekday())).isoformat(),
                    "session_id": str(session_id),
                    "session_name": session_name,
                    "session_scope": "FCL",
                    "session_scope_source": scope_text,
                    "player_id": str(player_id),
                    "player_name": name,
                    "position_raw": position_raw,
                    "position_group_default": position_group_default,
                    "drill_name": raw_name,
                    "drill_name_display": display_name,
                    "drill_sequence": sequence,
                    "player_drill_occurrence": occurrence,
                    "block_key": block_key,
                    "top_speed_ms": kpi(drill, "maxSpeed"),
                    "max_accel_ms2": kpi(drill, "maxAcceleration"),
                    "n_sprints": kpi(
                        drill,
                        "sprints",
                        lambda value: int(float(value)) if value is not None else np.nan,
                    ),
                    "n_accelerations": accelerations,
                    "hsr_distance_m": hsr_distance,
                    "total_distance_m": kpi(drill, "distanceTotal"),
                    "hmld_m": kpi(drill, "hmld"),
                    "sprint_distance_m": kpi(drill, "sprintDistance"),
                    "mechanical_load": kpi(drill, "mechanicalLoad"),
                    "duration_min": duration_min,
                }
                row["hsr_per_min"] = safe_divide(hsr_distance, duration_min)
                row["accels_per_min"] = safe_divide(accelerations, duration_min)
                row["total_distance_per_min"] = safe_divide(row["total_distance_m"], duration_min)
                row["hmld_per_min"] = safe_divide(row["hmld_m"], duration_min)
                rows.append(row)

    return rows


def pull_range(start_string, end_string):
    start_day = date.fromisoformat(start_string)
    end_day = date.fromisoformat(end_string)
    total_days = (end_day - start_day).days + 1
    current_day = start_day
    all_rows = []

    print(f"\nPulling FCL STATSports drills {start_day} -> {end_day} ({total_days} days)")

    for completed in range(1, total_days + 1):
        day_string = current_day.isoformat()
        rows = pull_day(day_string)
        all_rows.extend(rows)
        print(f"  [{completed:>3}/{total_days}] {day_string} -> {len(rows)} rows")
        current_day += timedelta(days=1)
        time.sleep(SLEEP_BETWEEN_DAYS)

    if not all_rows:
        return pd.DataFrame()

    data = pd.DataFrame(all_rows)
    numeric_columns = [
        "drill_sequence",
        "player_drill_occurrence",
        "top_speed_ms",
        "max_accel_ms2",
        "n_sprints",
        "n_accelerations",
        "hsr_distance_m",
        "total_distance_m",
        "hmld_m",
        "sprint_distance_m",
        "mechanical_load",
        "duration_min",
        "hsr_per_min",
        "accels_per_min",
        "total_distance_per_min",
        "hmld_per_min",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.sort_values(
        ["date", "session_name", "player_name", "drill_sequence"]
    ).reset_index(drop=True)


# ── APPEND-ONLY GOOGLE SHEETS SYNC ───────────────────────────────────────────
def clean_for_sheets_value(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def prepare_sheet_df(data):
    output = data.copy()
    for column in output.select_dtypes(include="float").columns:
        output[column] = output[column].round(3)
    return output


def normalize_key_part(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip().lower()


def add_occurrence_key(data, base_key_columns=KEY_COLS):
    """
    Safe identity: date + session + player + drill + within-player occurrence.
    This protects repeated drills with the same name in the same session.
    """
    output = data.copy()
    for column in base_key_columns:
        if column not in output.columns:
            output[column] = ""

    base_key = output[base_key_columns].apply(
        lambda row: "||".join(normalize_key_part(value) for value in row),
        axis=1,
    )
    output["_base_upsert_key"] = base_key
    output["_occurrence"] = output.groupby("_base_upsert_key", sort=False).cumcount() + 1
    output["_upsert_key"] = output["_base_upsert_key"] + "||occ=" + output["_occurrence"].astype(str)
    return output


def read_existing_sheet(worksheet):
    values = worksheet.get_all_values()
    if not values:
        return [], pd.DataFrame()

    headers = [str(header).strip() for header in values[0]]
    raw_data = values[1:]
    if not raw_data:
        return headers, pd.DataFrame(columns=headers)

    width = len(headers)
    padded_rows = [(row + [""] * width)[:width] for row in raw_data]
    return headers, pd.DataFrame(padded_rows, columns=headers)


def ensure_headers(worksheet, existing_headers, desired_headers):
    """Expand only row 1. Existing data rows are never overwritten."""
    if not existing_headers:
        end_cell = rowcol_to_a1(1, len(desired_headers))
        worksheet.update(
            [desired_headers],
            range_name=f"A1:{end_cell}",
            value_input_option="USER_ENTERED",
        )
        return desired_headers

    missing_headers = [header for header in desired_headers if header not in existing_headers]
    if not missing_headers:
        return existing_headers

    final_headers = existing_headers + missing_headers
    end_cell = rowcol_to_a1(1, len(final_headers))
    worksheet.update(
        [final_headers],
        range_name=f"A1:{end_cell}",
        value_input_option="USER_ENTERED",
    )
    print(f"  Added sheet headers only: {missing_headers}")
    return final_headers


def build_existing_key_set(existing_data):
    keyed = add_occurrence_key(existing_data)
    keys = set()
    duplicates = 0

    for _, row in keyed.iterrows():
        key = row.get("_upsert_key", "")
        base = row.get("_base_upsert_key", "")
        if not base or base == "||||||":
            continue
        if key in keys:
            duplicates += 1
        keys.add(key)

    return keys, duplicates


def rows_to_values(data, headers):
    return [
        [clean_for_sheets_value(row.get(header, "")) for header in headers]
        for _, row in data.iterrows()
    ]


def append_missing_rows(worksheet, missing_data, sheet_headers, chunk_size=1000):
    if missing_data.empty:
        return 0

    values = rows_to_values(missing_data, sheet_headers)
    for index in range(0, len(values), chunk_size):
        worksheet.append_rows(
            values[index:index + chunk_size],
            value_input_option="USER_ENTERED",
        )
        time.sleep(0.5)

    return len(values)


def write_to_sheets_append_only(data):
    """
    Strict append-only behavior:
    - Existing rows are not cleared, patched, compared, or overwritten.
    - New headers may be added on row 1.
    - Only rows whose occurrence-aware identity is not already present are appended.
    """
    if not SHEETS_AVAILABLE:
        raise RuntimeError("Google Sheets packages are not installed in this environment.")

    client = get_sheets_client()
    sheet = client.open_by_key(GOOGLE_SHEET_ID)

    titles = [worksheet.title for worksheet in sheet.worksheets()]
    if SHEET_TAB_NAME in titles:
        worksheet = sheet.worksheet(SHEET_TAB_NAME)
    else:
        worksheet = sheet.add_worksheet(title=SHEET_TAB_NAME, rows=10000, cols=40)

    output = prepare_sheet_df(data)
    script_headers = list(output.columns)

    existing_headers, existing_data = read_existing_sheet(worksheet)
    sheet_headers = ensure_headers(worksheet, existing_headers, script_headers)

    # Re-read to make the existing data shape match any newly added headers.
    _, existing_data = read_existing_sheet(worksheet)
    for header in sheet_headers:
        if header not in existing_data.columns:
            existing_data[header] = ""
    existing_data = existing_data[sheet_headers]

    if existing_data.empty:
        aligned = output.copy()
        for header in sheet_headers:
            if header not in aligned.columns:
                aligned[header] = ""
        appended = append_missing_rows(worksheet, aligned[sheet_headers], sheet_headers)
        return {
            "pulled": len(output),
            "appended": appended,
            "already_present": 0,
            "duplicate_existing": 0,
            "duplicate_api": int(add_occurrence_key(output)["_upsert_key"].duplicated().sum()),
        }

    existing_keys, duplicate_existing = build_existing_key_set(existing_data)
    keyed_output = add_occurrence_key(output)
    duplicate_api = int(keyed_output["_upsert_key"].duplicated().sum())

    missing_rows = []
    already_present = 0
    for _, row in keyed_output.iterrows():
        if row["_upsert_key"] in existing_keys:
            already_present += 1
        else:
            missing_rows.append(row[output.columns])

    missing_data = (
        pd.DataFrame(missing_rows, columns=output.columns)
        if missing_rows
        else pd.DataFrame(columns=output.columns)
    )
    for header in sheet_headers:
        if header not in missing_data.columns:
            missing_data[header] = ""
    missing_data = missing_data[sheet_headers]

    appended = append_missing_rows(worksheet, missing_data, sheet_headers)
    print(
        f"  APPEND-ONLY sync complete: {appended} appended | "
        f"{already_present} already present | 0 existing rows changed"
    )

    return {
        "pulled": len(output),
        "appended": appended,
        "already_present": already_present,
        "duplicate_existing": duplicate_existing,
        "duplicate_api": duplicate_api,
    }


# ── DAILY INTENSITY DATA MODEL ───────────────────────────────────────────────
def clean_for_json(data):
    if data is None or data.empty:
        return []
    clean = data.copy().replace([np.inf, -np.inf], np.nan)
    clean = clean.where(pd.notnull(clean), None)
    return clean.to_dict("records")


def data_from_records(records):
    if not records:
        return pd.DataFrame()
    data = pd.DataFrame(records)

    numeric_columns = [
        "drill_sequence",
        "player_drill_occurrence",
        "top_speed_ms",
        "max_accel_ms2",
        "n_sprints",
        "n_accelerations",
        "hsr_distance_m",
        "total_distance_m",
        "hmld_m",
        "sprint_distance_m",
        "mechanical_load",
        "duration_min",
        "hsr_per_min",
        "accels_per_min",
        "total_distance_per_min",
        "hmld_per_min",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    return data



def _load_local_position_groups():
    try:
        if not os.path.exists(ROSTER_ASSIGNMENTS_FILE):
            return {}
        with open(ROSTER_ASSIGNMENTS_FILE, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            return {}
        return {
            str(name).strip(): str(group).strip()
            for name, group in raw.items()
            if str(name).strip() and str(group).strip() in POSITION_GROUPS
        }
    except Exception as exc:
        print(f"[Roster tags] Could not load local position groups: {exc}")
        return {}


def _save_local_position_groups(mapping):
    folder = os.path.dirname(ROSTER_ASSIGNMENTS_FILE)
    if folder:
        os.makedirs(folder, exist_ok=True)
    temp_path = f"{ROSTER_ASSIGNMENTS_FILE}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2, sort_keys=True)
    os.replace(temp_path, ROSTER_ASSIGNMENTS_FILE)
    return ROSTER_ASSIGNMENTS_FILE


def load_saved_position_groups():
    """Load roster tags from the shared Google Sheet, with a local fallback."""
    if SHEETS_AVAILABLE and GOOGLE_SHEET_ID:
        try:
            client = get_sheets_client()
            book = client.open_by_key(GOOGLE_SHEET_ID)
            titles = [ws.title for ws in book.worksheets()]
            if ROSTER_SHEET_TAB_NAME in titles:
                worksheet = book.worksheet(ROSTER_SHEET_TAB_NAME)
                values = worksheet.get_all_values()
                if values:
                    header = [str(x).strip().lower() for x in values[0]]
                    if "player_name" in header and "position_group" in header:
                        name_i = header.index("player_name")
                        group_i = header.index("position_group")
                        mapping = {}
                        for row in values[1:]:
                            name = row[name_i].strip() if name_i < len(row) else ""
                            group = row[group_i].strip() if group_i < len(row) else ""
                            if name and group in POSITION_GROUPS:
                                mapping[name] = group
                        if mapping:
                            return mapping
        except Exception as exc:
            print(f"[Roster tags] Google Sheet load unavailable; using local fallback: {exc}")
    return _load_local_position_groups()


def save_position_groups(roster_rows):
    """Persist player group tags to Google Sheets when configured; otherwise save locally."""
    mapping = roster_group_lookup(roster_rows)
    if SHEETS_AVAILABLE and GOOGLE_SHEET_ID:
        try:
            client = get_sheets_client()
            book = client.open_by_key(GOOGLE_SHEET_ID)
            titles = [ws.title for ws in book.worksheets()]
            if ROSTER_SHEET_TAB_NAME in titles:
                worksheet = book.worksheet(ROSTER_SHEET_TAB_NAME)
            else:
                worksheet = book.add_worksheet(title=ROSTER_SHEET_TAB_NAME, rows=500, cols=3)
            rows = [["player_name", "position_group", "updated_at"]]
            stamp = datetime.now().isoformat(timespec="seconds")
            rows.extend([[name, group, stamp] for name, group in sorted(mapping.items())])
            worksheet.clear()
            worksheet.update(rows, range_name=f"A1:C{max(1, len(rows))}", value_input_option="USER_ENTERED")
            return len(mapping), f"Google Sheet tab: {ROSTER_SHEET_TAB_NAME}"
        except Exception as exc:
            print(f"[Roster tags] Google Sheet save failed; using local fallback: {exc}")

    path = _save_local_position_groups(mapping)
    return len(mapping), path


def build_roster_assignment_rows(raw, existing_rows=None):
    """Return one athlete row with locally saved and current-session tags preserved."""
    if raw is None or raw.empty or "player_name" not in raw.columns:
        return []

    # Stored tags are the default. Unsaved edits in the current browser session
    # take priority so a data refresh does not discard work in progress.
    prior = load_saved_position_groups()
    for row in (existing_rows or []):
        name = str(row.get("player_name", "")).strip()
        group = str(row.get("position_group", "")).strip()
        if name and group in POSITION_GROUPS:
            prior[name] = group

    work = raw.copy()
    if "position_raw" not in work.columns:
        work["position_raw"] = ""
    work["position_raw"] = work["position_raw"].fillna("").astype(str)

    rows = []
    for athlete, group in work.groupby("player_name", dropna=True, sort=True):
        athlete = str(athlete).strip()
        if not athlete:
            continue
        available = [v.strip() for v in group["position_raw"].tolist() if str(v).strip()]
        detected = available[0] if available else ""
        # The API position is shown as a reference, but every new athlete
        # starts unassigned until staff deliberately tags the workload group.
        assigned = prior.get(athlete, "Unassigned")
        rows.append({
            "player_name": athlete,
            "position_raw": detected or "Not provided by API",
            "position_group": assigned if assigned in POSITION_GROUPS else "Unassigned",
        })
    return rows


def roster_group_lookup(roster_rows):
    lookup = {}
    for row in (roster_rows or []):
        name = str(row.get("player_name", "")).strip()
        group = str(row.get("position_group", "")).strip()
        if name:
            lookup[name] = group if group in POSITION_GROUPS else "Unassigned"
    return lookup



# ── PDF REPORT DATA / EXPORT ────────────────────────────────────────────────
def build_report_day_type_rows(period_start, period_end, existing_rows=None):
    """Return one editable day-type row per calendar day, preserving user edits."""
    if period_start is None or period_end is None:
        return []

    start = pd.to_datetime(period_start, errors="coerce")
    end = pd.to_datetime(period_end, errors="coerce")
    if pd.isna(start) or pd.isna(end) or end < start:
        return []

    prior = {}
    for row in (existing_rows or []):
        key = str(row.get("date", "")).strip()
        value = str(row.get("day_type", "")).strip()
        if key:
            prior[key] = value if value in REPORT_DAY_TYPES else "Not set"

    rows = []
    for stamp in pd.date_range(start.normalize(), end.normalize(), freq="D"):
        key = stamp.strftime("%Y-%m-%d")
        rows.append({
            "date": key,
            "day": stamp.strftime("%A"),
            "day_type": prior.get(key, "Not set"),
        })
    return rows


def report_day_type_lookup(day_type_rows):
    """Map ISO date to the manually selected High/Low report classification."""
    lookup = {}
    for row in (day_type_rows or []):
        key = str(row.get("date", "")).strip()
        value = str(row.get("day_type", "")).strip()
        if key:
            lookup[key] = value if value in REPORT_DAY_TYPES else "Not set"
    return lookup


def hsr_target_for(position_group, day_type):
    """Return the workload target dictionary for a position group and day type."""
    return (HSR_TARGETS.get(str(position_group or ""), {}) or {}).get(str(day_type or ""))


def _format_target_operator(operator):
    return {">=": "≥", "<=": "≤"}.get(str(operator), str(operator))


def _compare_target_value(value, operator, threshold):
    if pd.isna(value):
        return None
    value = float(value)
    threshold = float(threshold)
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    raise ValueError(f"Unsupported target operator: {operator}")


def format_hsr_target(position_group, day_type):
    target = hsr_target_for(position_group, day_type)
    if not target:
        return "No target"

    hsr_text = (
        f"HSR {_format_target_operator(target['hsr_operator'])} "
        f"{target['hsr_value']:.0f} m"
    )
    if "accel_operator" not in target or "accel_value" not in target:
        return hsr_text

    accel_text = (
        f"accels {_format_target_operator(target['accel_operator'])} "
        f"{target['accel_value']:.0f}"
    )
    joiner = " OR " if str(target.get("logic", "and")).lower() == "or" else " AND "
    return f"{hsr_text}{joiner}{accel_text}"


def evaluate_hsr_target(hsr_m, accelerations, position_group, day_type):
    """Return Hit/Miss/Pending/No target for one player-day workload row."""
    target = hsr_target_for(position_group, day_type)
    if day_type not in {"High", "Low"}:
        return "Pending"
    if not target:
        return "No target"

    hsr_value = pd.to_numeric(pd.Series([hsr_m]), errors="coerce").iloc[0]
    hsr_result = _compare_target_value(
        hsr_value,
        target["hsr_operator"],
        target["hsr_value"],
    )

    if "accel_operator" not in target or "accel_value" not in target:
        if hsr_result is None:
            return "No data"
        return "Hit" if hsr_result else "Miss"

    accel_value = pd.to_numeric(pd.Series([accelerations]), errors="coerce").iloc[0]
    accel_result = _compare_target_value(
        accel_value,
        target["accel_operator"],
        target["accel_value"],
    )
    if hsr_result is None or accel_result is None:
        return "No data"

    logic = str(target.get("logic", "and")).lower()
    hit = (hsr_result or accel_result) if logic == "or" else (hsr_result and accel_result)
    return "Hit" if hit else "Miss"


def format_target_result(position_group, day_type, target_status):
    """Show the result together with only the threshold that applies to this day."""
    threshold = format_hsr_target(position_group, day_type)
    status = str(target_status or "No target")
    if threshold == "No target":
        return "No target"
    if status in {"Hit", "Miss"}:
        return f"{status} · {threshold}"
    if status == "No data":
        return f"No data · {threshold}"
    if status == "Pending":
        return f"Pending · {threshold}"
    return status


def build_pdf_player_day_summary(data, roster_rows, day_type_rows, period_start, period_end):
    """
    Build one row per athlete and date for the target report. The input should already
    have the same field-block exclusions applied as the dashboard view.
    """
    if data is None or data.empty or period_start is None or period_end is None:
        return pd.DataFrame()

    start = pd.to_datetime(period_start, errors="coerce")
    end = pd.to_datetime(period_end, errors="coerce")
    if pd.isna(start) or pd.isna(end) or end < start:
        return pd.DataFrame()
    start, end = start.normalize(), end.normalize()

    position_lookup = roster_group_lookup(roster_rows)
    day_type_map = report_day_type_lookup(day_type_rows)

    use = data.copy()
    use["day_date"] = pd.to_datetime(use.get("date"), errors="coerce").dt.normalize()
    use = use[use["day_date"].notna() & use["day_date"].between(start, end)].copy()
    if use.empty:
        return pd.DataFrame()

    for column in ("hsr_distance_m", "n_accelerations", "duration_min"):
        if column not in use.columns:
            use[column] = np.nan
        use[column] = pd.to_numeric(use[column], errors="coerce")

    summary = (
        use.groupby(["day_date", "player_name"], as_index=False)
        .agg(
            hsr_m=("hsr_distance_m", "sum"),
            accelerations=("n_accelerations", "sum"),
            field_minutes=("duration_min", "sum"),
            sessions=("session_id", "nunique"),
        )
    )
    summary["date"] = summary["day_date"].dt.strftime("%Y-%m-%d")
    summary["day"] = summary["day_date"].dt.strftime("%a %b %-d")
    summary["position_group"] = summary["player_name"].astype(str).map(
        lambda value: position_lookup.get(value, "Unassigned")
    )
    summary["day_type"] = summary["date"].map(lambda value: day_type_map.get(value, "Not set"))
    summary["target"] = summary.apply(
        lambda row: format_hsr_target(row["position_group"], row["day_type"]), axis=1
    )
    summary["target_status"] = summary.apply(
        lambda row: evaluate_hsr_target(
            row["hsr_m"], row["accelerations"], row["position_group"], row["day_type"]
        ), axis=1
    )

    # Preserve a fixed baseball order before names, while retaining manual custom groups.
    order_map = {group: index for index, group in enumerate(REPORT_POSITION_ORDER)}
    summary["_position_order"] = summary["position_group"].map(order_map).fillna(len(order_map)).astype(int)
    summary = summary.sort_values(["day_date", "_position_order", "player_name"], kind="stable").reset_index(drop=True)
    return summary.drop(columns=["_position_order"])


def build_pdf_daily_summary(player_day_summary, day_type_rows, period_start, period_end):
    """Summarize daily target coverage for the report cover page."""
    day_rows = build_report_day_type_rows(period_start, period_end, day_type_rows)
    calendar = pd.DataFrame(day_rows)
    if calendar.empty:
        return calendar

    calendar["day_date"] = pd.to_datetime(calendar["date"], errors="coerce")
    if player_day_summary is None or player_day_summary.empty:
        calendar["players_recorded"] = 0
        calendar["targetable_players"] = 0
        calendar["hits"] = 0
        calendar["misses"] = 0
        calendar["hit_rate"] = np.nan
        return calendar

    work = player_day_summary.copy()
    work["is_targetable"] = work["target_status"].isin(["Hit", "Miss"])
    work["is_hit"] = work["target_status"].eq("Hit")
    work["is_miss"] = work["target_status"].eq("Miss")
    daily = (
        work.groupby("date", as_index=False)
        .agg(
            players_recorded=("player_name", "nunique"),
            targetable_players=("is_targetable", "sum"),
            hits=("is_hit", "sum"),
            misses=("is_miss", "sum"),
        )
    )
    daily["hit_rate"] = np.where(
        daily["targetable_players"] > 0,
        daily["hits"] / daily["targetable_players"] * 100,
        np.nan,
    )
    out = calendar.merge(daily, on="date", how="left")
    for column in ("players_recorded", "targetable_players", "hits", "misses"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    return out


def _player_axis_label(name):
    """Return a compact full-name label that stays readable below each bar pair."""
    parts = [part for part in str(name or "").strip().split() if part]
    if not parts:
        return "Player"
    if len(parts) == 1:
        return parts[0][:18]
    first = parts[0]
    last = " ".join(parts[1:])
    return f"{first}\n{last}"


def build_pdf_position_chart(day_data, report_date, day_type):
    """Create a polished two-panel player chart grouped by manual position group."""
    work = day_data.copy()
    order_map = {group: index for index, group in enumerate(REPORT_POSITION_ORDER)}
    work["_position_order"] = work["position_group"].map(order_map).fillna(len(order_map)).astype(int)
    work = work.sort_values(["_position_order", "player_name"], kind="stable").reset_index(drop=True)

    if work.empty:
        return None

    position_values, labels, group_bounds = [], [], []
    cursor = 0.0
    visible_groups = [group for group in REPORT_POSITION_ORDER if group in set(work["position_group"].astype(str))]
    visible_groups += [
        group for group in work["position_group"].dropna().astype(str).unique().tolist()
        if group not in visible_groups
    ]
    for group_index, group in enumerate(visible_groups):
        subset = work[work["position_group"].astype(str) == str(group)]
        if subset.empty:
            continue
        start = cursor
        for _, row in subset.iterrows():
            position_values.append(cursor)
            labels.append(_player_axis_label(row.get("player_name", "")))
            cursor += 1.0
        group_bounds.append((group, start, cursor - 1.0, group_index))
        cursor += 0.85

    if not position_values:
        return None

    plot = work.copy()
    plot["_x"] = position_values
    hsr_values = pd.to_numeric(plot.get("hsr_m"), errors="coerce").fillna(0)
    accel_values = pd.to_numeric(plot.get("accelerations"), errors="coerce").fillna(0)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.55, 6.65),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1], "hspace": 0.42},
    )
    fig.patch.set_facecolor("white")

    status_colors = {
        "Hit": "#16A34A",
        "Miss": "#DC2626",
        "Pending": "#94A3B8",
        "No target": "#94A3B8",
        "No data": "#94A3B8",
    }
    hsr_colors = [status_colors.get(value, "#94A3B8") for value in plot["target_status"].astype(str).tolist()]

    # Light group bands make the positional groupings easy to scan without clutter.
    group_shades = ["#F8FAFC", "#FFFFFF"]
    for group, start_x, end_x, group_index in group_bounds:
        for axis in axes:
            axis.axvspan(start_x - 0.48, end_x + 0.48, color=group_shades[group_index % 2], zorder=0)

    hsr_bars = axes[0].bar(
        plot["_x"], hsr_values, width=0.72, color=hsr_colors,
        edgecolor="white", linewidth=1.05, zorder=3,
    )
    accel_bars = axes[1].bar(
        plot["_x"], accel_values, width=0.72, color="#0B2F5B",
        edgecolor="white", linewidth=1.05, zorder=3,
    )

    axes[0].set_ylabel("HSR (m)", fontweight="bold", color="#0F172A")
    axes[0].set_title("High-speed running", loc="left", fontsize=11.5, fontweight="bold", pad=21, color="#0F172A")
    axes[1].set_ylabel("Accelerations", fontweight="bold", color="#0F172A")
    axes[1].set_title("Accelerations", loc="left", fontsize=11.5, fontweight="bold", pad=10, color="#0F172A")

    max_hsr = max(float(hsr_values.max()), 1.0)
    max_accels = max(float(accel_values.max()), 1.0)
    axes[0].set_ylim(0, max_hsr * 1.34 + 6)
    axes[1].set_ylim(0, max_accels * 1.25 + 2)

    for bar, value in zip(hsr_bars, hsr_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + max_hsr * 0.035 + 1,
            f"{float(value):.0f}",
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#334155", zorder=5,
        )
    for bar, value in zip(accel_bars, accel_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + max_accels * 0.05 + 0.4,
            f"{float(value):.0f}",
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#334155", zorder=5,
        )

    for group, start_x, end_x, _ in group_bounds:
        center = (start_x + end_x) / 2
        target_text = (
            format_hsr_target(group, day_type)
            .replace(" OR ", "\nOR ")
            .replace(" AND ", "\nAND ")
        )
        axes[0].text(
            center,
            axes[0].get_ylim()[1] * 1.02,
            f"{group}\n{target_text}",
            ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#334155",
        )
        target = hsr_target_for(group, day_type)
        if target:
            axes[0].hlines(
                target["hsr_value"], start_x - 0.42, end_x + 0.42,
                color="#64748B", linestyle=(0, (4, 3)), linewidth=1.15, zorder=4,
            )
        for axis in axes:
            axis.axvline(end_x + 0.42, color="#CBD5E1", linewidth=0.85, zorder=2)

    axes[0].set_xticks(position_values)
    axes[0].set_xticklabels(labels, rotation=0, ha="center", fontsize=6.2, color="#334155")
    axes[0].tick_params(axis="x", pad=7, bottom=False, labelbottom=True)

    axes[1].set_xticks(position_values)
    axes[1].set_xticklabels(labels, rotation=0, ha="center", fontsize=6.5, color="#334155")
    axes[1].tick_params(axis="x", pad=8)
    axes[1].set_xlabel("")

    for axis in axes:
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.75, zorder=1)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#CBD5E1")
        axis.spines["bottom"].set_color("#CBD5E1")
        axis.tick_params(colors="#475569", labelsize=7.5)

    fig.suptitle(
        f"{pd.Timestamp(report_date).strftime('%A, %b %-d, %Y')} | {day_type} day",
        x=0.075, y=0.995, ha="left", fontsize=14, fontweight="bold", color="#041E42",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.18)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=190, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _pdf_paragraph(value, style):
    text = "" if value is None else str(value)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(text, style)


def _status_cell_color(status):
    return {
        "Hit": colors.HexColor("#EAF8EF"),
        "Miss": colors.HexColor("#FDEBEC"),
        "Pending": colors.HexColor("#FEF3C7"),
        "No target": colors.HexColor("#F1F5F9"),
        "No data": colors.HexColor("#F1F5F9"),
    }.get(str(status), colors.white)


def _pdf_metric_tile(label, value, accent_hex, value_style, label_style, tile_width=1.96 * inch):
    tile = Table(
        [[Paragraph(str(value), value_style)], [Paragraph(str(label), label_style)]],
        colWidths=[tile_width],
        rowHeights=[0.31 * inch, 0.25 * inch],
    )
    tile.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FBFCFE")),
        ("LINEABOVE", (0, 0), (-1, 0), 3.0, colors.HexColor(accent_hex)),
        ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor("#D8E0EA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tile


def create_single_day_pdf_target_report(player_day_summary, report_date, day_type):
    """Build a polished PDF report covering exactly one selected date."""
    if player_day_summary is None or player_day_summary.empty:
        raise ValueError("No player workload rows are available for the selected report date.")

    report_stamp = pd.to_datetime(report_date, errors="coerce")
    if pd.isna(report_stamp):
        raise ValueError("Choose a valid report date.")
    report_stamp = report_stamp.normalize()

    work = player_day_summary.copy()
    work["day_date"] = pd.to_datetime(work.get("day_date"), errors="coerce").dt.normalize()
    work = work[work["day_date"].eq(report_stamp)].copy()
    if work.empty:
        raise ValueError("No included FCL player workload rows were found for the selected report date.")

    # The chosen day type is authoritative for every player in this single-day report.
    work["day_type"] = day_type
    work["target"] = work.apply(
        lambda row: format_hsr_target(row.get("position_group"), day_type), axis=1
    )
    work["target_status"] = work.apply(
        lambda row: evaluate_hsr_target(
            row.get("hsr_m"), row.get("accelerations"), row.get("position_group"), day_type
        ), axis=1
    )

    order_map = {group: index for index, group in enumerate(REPORT_POSITION_ORDER)}
    work["_position_order"] = work["position_group"].map(order_map).fillna(len(order_map)).astype(int)
    work = work.sort_values(["_position_order", "player_name"], kind="stable").drop(columns=["_position_order"]).reset_index(drop=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_day_type = re.sub(r"[^A-Za-z0-9]+", "", str(day_type)) or "Day"
    path = os.path.join("/tmp", f"FCL_Daily_HSR_Report_{report_stamp.strftime('%Y%m%d')}_{safe_day_type}_{stamp}.pdf")

    doc = SimpleDocTemplate(
        path,
        pagesize=landscape(letter),
        leftMargin=0.38 * inch,
        rightMargin=0.38 * inch,
        topMargin=0.34 * inch,
        bottomMargin=0.33 * inch,
    )
    styles = getSampleStyleSheet()
    header_title_style = ParagraphStyle(
        "ReportHeaderTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19.5,
        leading=22, textColor=colors.white, alignment=TA_LEFT,
    )
    header_kicker_style = ParagraphStyle(
        "ReportHeaderKicker", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.8,
        leading=9, textColor=colors.HexColor("#BFDBFE"), alignment=TA_LEFT,
    )
    header_copy_style = ParagraphStyle(
        "ReportHeaderCopy", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.4,
        leading=10.5, textColor=colors.HexColor("#D9E7F5"), alignment=TA_LEFT,
    )
    badge_label_style = ParagraphStyle(
        "ReportBadgeLabel", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8.5,
        leading=10, textColor=colors.white, alignment=TA_CENTER,
    )
    badge_date_style = ParagraphStyle(
        "ReportBadgeDate", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8.0,
        leading=10, textColor=colors.white, alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "ReportHeading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
        leading=14, textColor=colors.HexColor(C_NAVY), spaceBefore=1, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.2,
        leading=8.9, textColor=colors.HexColor(C_TEXT), alignment=TA_LEFT,
    )
    body_center_style = ParagraphStyle(
        "ReportBodyCenter", parent=body_style, alignment=TA_CENTER,
    )
    header_style = ParagraphStyle(
        "ReportTableHeader", parent=body_style, fontName="Helvetica-Bold", fontSize=7.0,
        leading=8.2, textColor=colors.white, alignment=TA_CENTER,
    )
    metric_value_style = ParagraphStyle(
        "MetricValue", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=16,
        leading=18, textColor=colors.HexColor(C_NAVY), alignment=TA_LEFT,
    )
    metric_label_style = ParagraphStyle(
        "MetricLabel", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.0,
        leading=8, textColor=colors.HexColor("#64748B"), alignment=TA_LEFT,
    )
    target_note_title_style = ParagraphStyle(
        "TargetNoteTitle", parent=body_style, fontName="Helvetica-Bold", fontSize=9.2,
        leading=11, textColor=colors.HexColor(C_NAVY), alignment=TA_LEFT,
    )
    target_note_body_style = ParagraphStyle(
        "TargetNoteBody", parent=body_style, fontSize=7.8,
        leading=10.0, textColor=colors.HexColor("#475569"), alignment=TA_LEFT,
    )
    status_hit_style = ParagraphStyle(
        "StatusHit", parent=body_center_style, fontName="Helvetica-Bold", textColor=colors.HexColor("#166534"),
    )
    status_miss_style = ParagraphStyle(
        "StatusMiss", parent=body_center_style, fontName="Helvetica-Bold", textColor=colors.HexColor("#B91C1C"),
    )
    status_neutral_style = ParagraphStyle(
        "StatusNeutral", parent=body_center_style, fontName="Helvetica-Bold", textColor=colors.HexColor("#475569"),
    )

    player_count = int(work["player_name"].nunique())
    avg_hsr = float(pd.to_numeric(work["hsr_m"], errors="coerce").mean()) if player_count else np.nan
    avg_accels = float(pd.to_numeric(work["accelerations"], errors="coerce").mean()) if player_count else np.nan
    report_date_label = report_stamp.strftime("%A, %b %-d, %Y")
    day_badge_color = C_RED if str(day_type).lower() == "high" else C_BLUE

    header_left = [
        Paragraph("FCL GPS LOAD", header_kicker_style),
        Paragraph("Daily GPS load report", header_title_style),
    ]
    header_right = [
        Paragraph(f"{str(day_type).upper()} DAY", badge_label_style),
        Spacer(1, 0.04 * inch),
        Paragraph(report_date_label, badge_date_style),
    ]
    report_header = Table([[header_left, header_right]], colWidths=[7.7 * inch, 2.54 * inch], hAlign="LEFT")
    report_header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(C_NAVY)),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor(day_badge_color)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 18),
        ("RIGHTPADDING", (0, 0), (0, 0), 16),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (1, 0), (1, 0), 11),
        ("RIGHTPADDING", (1, 0), (1, 0), 11),
        ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor(C_NAVY)),
    ]))

    metric_tile_width = 3.37 * inch
    metric_tiles = [
        _pdf_metric_tile("Players included", player_count, C_NAVY, metric_value_style, metric_label_style, metric_tile_width),
        _pdf_metric_tile("Avg HSR / player", "-" if pd.isna(avg_hsr) else f"{avg_hsr:.1f} m", C_BLUE, metric_value_style, metric_label_style, metric_tile_width),
        _pdf_metric_tile("Avg accels / player", "-" if pd.isna(avg_accels) else f"{avg_accels:.1f}", C_BLUE, metric_value_style, metric_label_style, metric_tile_width),
    ]
    metrics_table = Table([metric_tiles], colWidths=[3.42 * inch] * 3, hAlign="LEFT")
    metrics_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    detail_headers = ["Position group", "Player", "HSR", "Accelerations", "Target result"]
    detail_data = [[_pdf_paragraph(value, header_style) for value in detail_headers]]
    for _, row in work.iterrows():
        status = str(row.get("target_status", "No target"))
        if status == "Hit":
            status_style = status_hit_style
        elif status == "Miss":
            status_style = status_miss_style
        else:
            status_style = status_neutral_style
        target_result = format_target_result(
            row.get("position_group", "Unassigned"),
            day_type,
            status,
        )
        detail_data.append([
            _pdf_paragraph(row.get("position_group", "Unassigned"), body_style),
            _pdf_paragraph(row.get("player_name", ""), body_style),
            _pdf_paragraph(f"{float(row.get('hsr_m', 0)):.1f} m", body_center_style),
            _pdf_paragraph(f"{float(row.get('accelerations', 0)):.0f}", body_center_style),
            _pdf_paragraph(target_result, status_style),
        ])
    detail_table = LongTable(
        detail_data,
        colWidths=[1.55 * inch, 3.20 * inch, 1.20 * inch, 1.40 * inch, 1.65 * inch],
        repeatRows=1,
        hAlign="LEFT",
    )
    detail_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(C_NAVY)),
        ("GRID", (0, 0), (-1, -1), 0.28, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.0),
    ]
    target_status_col = len(detail_headers) - 1
    for row_index in range(1, len(detail_data)):
        if row_index % 2 == 0:
            detail_style.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F8FAFC")))
        status = str(work.iloc[row_index - 1].get("target_status", "No target"))
        detail_style.append(("BACKGROUND", (target_status_col, row_index), (target_status_col, row_index), _status_cell_color(status)))
    detail_table.setStyle(TableStyle(detail_style))

    footer_style = ParagraphStyle(
        "ReportFooter", parent=body_style, fontSize=6.7, leading=8.5, textColor=colors.HexColor("#94A3B8"), alignment=TA_LEFT,
    )
    profile_title_style = ParagraphStyle(
        "ProfileTitle", parent=heading_style, fontSize=14, leading=17, textColor=colors.HexColor(C_NAVY), spaceAfter=3,
    )
    profile_copy_style = ParagraphStyle(
        "ProfileCopy", parent=body_style, fontSize=8.0, leading=10, textColor=colors.HexColor("#64748B"),
    )

    story = [
        report_header,
        Spacer(1, 0.16 * inch),
        metrics_table,
        Spacer(1, 0.17 * inch),
        Paragraph(f"Included players ({player_count})", heading_style),
        detail_table,
        Spacer(1, 0.08 * inch),
        PageBreak(),
        Paragraph("Player load profile", profile_title_style),
        Spacer(1, 0.03 * inch),
    ]
    chart_buffer = build_pdf_position_chart(work, report_stamp.strftime("%Y-%m-%d"), day_type)
    if chart_buffer is not None:
        story.append(ReportLabImage(chart_buffer, width=10.15 * inch, height=6.38 * inch))
    else:
        story.append(Paragraph("No player workload rows were available for this date.", body_style))

    doc.build(story)
    return path

def is_non_field_block(drill_name):
    raw = str(drill_name or "").strip().lower()
    display = clean_drill_name(drill_name).strip().lower()
    return any(
        raw.startswith(prefix) or display.startswith(prefix)
        for prefix in NON_FIELD_PREFIXES
    )


def filter_dashboard_rows(data, exclude_non_field):
    """Apply the optional non-field filter to already FCL data."""
    if data.empty:
        return data.copy()

    output = data.copy()

    # Protect the dashboard against a stale browser store created by an older app version.
    if "is_excluded" in output.columns:
        output = output[~output["is_excluded"].fillna(False).astype(bool)].copy()

    if "exclude_non_field" in (exclude_non_field or []):
        output = output[
            ~output["drill_name"].map(is_non_field_block)
        ].copy()

    output = output[output["duration_min"].fillna(0) >= MIN_DURATION_MIN].copy()
    return output


def build_block_summary(data):
    """
    One row per inferred team drill block.

    A block is drill name + repeat occurrence. This prevents the first and second
    appearances of the same drill name from being merged into one long block.
    Metrics are medians across athlete rows so the chart represents a typical athlete,
    not a cumulative team total.
    """
    if data.empty:
        return pd.DataFrame()

    group_columns = [
        "date",
        "session_id",
        "session_name",
        "block_key",
        "drill_name",
        "drill_name_display",
    ]
    available = [column for column in group_columns if column in data.columns]

    blocks = (
        data.groupby(available, dropna=False, as_index=False)
        .agg(
            player_count=("player_name", "nunique"),
            median_sequence=("drill_sequence", "median"),
            median_duration_min=("duration_min", "median"),
            median_hsr_m=("hsr_distance_m", "median"),
            median_accels=("n_accelerations", "median"),
            median_distance_m=("total_distance_m", "median"),
            median_hmld_m=("hmld_m", "median"),
            median_hsr_per_min=("hsr_per_min", "median"),
            median_accels_per_min=("accels_per_min", "median"),
            p90_top_speed_ms=("top_speed_ms", lambda values: values.quantile(0.90)),
            rows=("player_name", "size"),
        )
        .sort_values(["date", "session_name", "median_sequence", "block_key"])
        .reset_index(drop=True)
    )

    metric_columns = [
        "median_sequence",
        "median_duration_min",
        "median_hsr_m",
        "median_accels",
        "median_distance_m",
        "median_hmld_m",
        "median_hsr_per_min",
        "median_accels_per_min",
        "p90_top_speed_ms",
    ]
    for column in metric_columns:
        blocks[column] = pd.to_numeric(blocks[column], errors="coerce")

    blocks["median_duration_min"] = blocks["median_duration_min"].clip(lower=0)
    return blocks


def percentile_values(reference, values):
    """Empirical 0-100 percentile values with safe handling of missing inputs."""
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().sort_values().to_numpy()
    vals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
    result = np.full(len(vals), np.nan)

    if len(ref) == 0:
        return result

    valid = ~np.isnan(vals)
    result[valid] = np.searchsorted(ref, vals[valid], side="right") / len(ref) * 100
    return result


def add_intensity_score(blocks, reference_blocks, hsr_weight_pct):
    """Blend HSR/min and accelerations/min percentile ranks into a transparent score."""
    if blocks.empty:
        return blocks

    output = blocks.copy()
    hsr_weight = max(0, min(100, float(hsr_weight_pct))) / 100
    accel_weight = 1 - hsr_weight

    output["hsr_intensity_pct"] = percentile_values(
        reference_blocks["median_hsr_per_min"],
        output["median_hsr_per_min"],
    )
    output["accel_intensity_pct"] = percentile_values(
        reference_blocks["median_accels_per_min"],
        output["median_accels_per_min"],
    )
    output["intensity_score"] = (
        hsr_weight * output["hsr_intensity_pct"].fillna(0)
        + accel_weight * output["accel_intensity_pct"].fillna(0)
    ).round(1)

    output["intensity_band"] = pd.cut(
        output["intensity_score"],
        bins=[-np.inf, 35, 65, 85, np.inf],
        labels=["Low", "Moderate", "High", "Very High"],
        include_lowest=True,
    ).astype(str)

    output = output.sort_values(["median_sequence", "block_key"]).reset_index(drop=True)
    output["display_duration_min"] = output["median_duration_min"].fillna(0).clip(lower=0.25)
    output["timeline_start_min"] = output["display_duration_min"].cumsum().shift(fill_value=0)
    output["timeline_end_min"] = output["timeline_start_min"] + output["display_duration_min"]
    output["timeline_label"] = np.where(
        output["drill_name_display"].str.len() > 18,
        output["drill_name_display"].str.slice(0, 17) + "…",
        output["drill_name_display"],
    )
    return output


def build_player_daily_summary(data):
    """Create a typical-player daily summary without summing athletes into team totals."""
    if data.empty:
        return pd.DataFrame()

    player_day = (
        data.groupby(["date", "session_id", "session_name", "player_name"], as_index=False)
        .agg(
            player_hsr_m=("hsr_distance_m", "sum"),
            player_accels=("n_accelerations", "sum"),
            player_duration_min=("duration_min", "sum"),
            player_distance_m=("total_distance_m", "sum"),
            player_hmld_m=("hmld_m", "sum"),
        )
    )
    player_day["player_hsr_per_min"] = player_day.apply(
        lambda row: safe_divide(row["player_hsr_m"], row["player_duration_min"]),
        axis=1,
    )
    player_day["player_accels_per_min"] = player_day.apply(
        lambda row: safe_divide(row["player_accels"], row["player_duration_min"]),
        axis=1,
    )

    summary = (
        player_day.groupby(["date", "session_id", "session_name"], as_index=False)
        .agg(
            players=("player_name", "nunique"),
            typical_hsr_m=("player_hsr_m", "median"),
            typical_accels=("player_accels", "median"),
            typical_duration_min=("player_duration_min", "median"),
            typical_distance_m=("player_distance_m", "median"),
            typical_hmld_m=("player_hmld_m", "median"),
            typical_hsr_per_min=("player_hsr_per_min", "median"),
            typical_accels_per_min=("player_accels_per_min", "median"),
        )
        .sort_values(["date", "session_name"])
        .reset_index(drop=True)
    )

    return summary



def group_percentiles(reference, values):
    """Return 0-100 percentile ranks for a player metric within the selected group."""
    return percentile_values(reference, values)


def flag_from_percentile(percentile, high_cutoff, comparison_ready):
    """Use plain-language load separation labels rather than overclaiming an injury-risk outlier."""
    try:
        percentile = float(percentile)
    except Exception:
        return "Unavailable"

    if not comparison_ready or math.isnan(percentile):
        return "Comparison limited"
    if percentile >= float(high_cutoff):
        return "Higher than group"
    if percentile <= 100 - float(high_cutoff):
        return "Lower than group"
    return "Within group"


def relative_change_from_median(values, median_value):
    """Percent difference from group median; safely returns NaN where a ratio is not meaningful."""
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    try:
        median_value = float(median_value)
    except Exception:
        median_value = np.nan
    if pd.isna(median_value) or abs(median_value) < 1e-9:
        return pd.Series(np.nan, index=series.index)
    return ((series - median_value) / median_value * 100).round(1)


def build_player_session_summary(data, hsr_weight_pct, high_cutoff=85):
    """
    Summarize each player's filtered selected-session workload and compare it only
    with teammates in that exact session/filter combination.

    Load score = blended within-group percentile of total HSR and total
    accelerations. Rate score = the same blend for HSR/min and accelerations/min.
    The default flagging boundary is 85th / 15th percentile, and flags are
    intentionally disabled when fewer than five players are available.
    """
    if data.empty:
        return pd.DataFrame()

    group_columns = ["date", "session_id", "session_name", "player_id", "player_name"]
    present_group_columns = [column for column in group_columns if column in data.columns]
    if "player_name" not in present_group_columns:
        return pd.DataFrame()

    player_summary = (
        data.groupby(present_group_columns, dropna=False, as_index=False)
        .agg(
            player_hsr_m=("hsr_distance_m", "sum"),
            player_accels=("n_accelerations", "sum"),
            player_duration_min=("duration_min", "sum"),
            player_distance_m=("total_distance_m", "sum"),
            player_hmld_m=("hmld_m", "sum"),
            player_blocks=("block_key", "nunique"),
        )
    )

    for column in [
        "player_hsr_m", "player_accels", "player_duration_min",
        "player_distance_m", "player_hmld_m", "player_blocks",
    ]:
        player_summary[column] = pd.to_numeric(player_summary[column], errors="coerce")

    player_summary["player_hsr_per_min"] = player_summary.apply(
        lambda row: safe_divide(row["player_hsr_m"], row["player_duration_min"]),
        axis=1,
    )
    player_summary["player_accels_per_min"] = player_summary.apply(
        lambda row: safe_divide(row["player_accels"], row["player_duration_min"]),
        axis=1,
    )

    hsr_weight = max(0, min(100, float(hsr_weight_pct))) / 100
    accel_weight = 1 - hsr_weight
    player_count = int(player_summary["player_name"].nunique())
    comparison_ready = player_count >= 5

    # Percentiles are within this group only — never versus the wider date-range reference.
    player_summary["hsr_group_pct"] = group_percentiles(
        player_summary["player_hsr_m"], player_summary["player_hsr_m"],
    )
    player_summary["accel_group_pct"] = group_percentiles(
        player_summary["player_accels"], player_summary["player_accels"],
    )
    player_summary["hsr_rate_group_pct"] = group_percentiles(
        player_summary["player_hsr_per_min"], player_summary["player_hsr_per_min"],
    )
    player_summary["accel_rate_group_pct"] = group_percentiles(
        player_summary["player_accels_per_min"], player_summary["player_accels_per_min"],
    )

    player_summary["group_load_score"] = (
        hsr_weight * player_summary["hsr_group_pct"].fillna(0)
        + accel_weight * player_summary["accel_group_pct"].fillna(0)
    ).round(1)
    player_summary["group_rate_score"] = (
        hsr_weight * player_summary["hsr_rate_group_pct"].fillna(0)
        + accel_weight * player_summary["accel_rate_group_pct"].fillna(0)
    ).round(1)

    group_hsr_median = player_summary["player_hsr_m"].median()
    group_accel_median = player_summary["player_accels"].median()
    player_summary["hsr_vs_group_median_pct"] = relative_change_from_median(
        player_summary["player_hsr_m"], group_hsr_median,
    )
    player_summary["accels_vs_group_median_pct"] = relative_change_from_median(
        player_summary["player_accels"], group_accel_median,
    )
    player_summary["group_players"] = player_count
    player_summary["comparison_ready"] = comparison_ready
    player_summary["load_flag"] = player_summary["group_load_score"].map(
        lambda value: flag_from_percentile(value, high_cutoff, comparison_ready)
    )
    player_summary["rate_flag"] = player_summary["group_rate_score"].map(
        lambda value: flag_from_percentile(value, high_cutoff, comparison_ready)
    )
    player_summary["hsr_flag"] = player_summary["hsr_group_pct"].map(
        lambda value: flag_from_percentile(value, high_cutoff, comparison_ready)
    )
    player_summary["accel_flag"] = player_summary["accel_group_pct"].map(
        lambda value: flag_from_percentile(value, high_cutoff, comparison_ready)
    )
    player_summary["flag_cutoff"] = int(high_cutoff)

    return player_summary.sort_values(
        ["group_load_score", "player_hsr_m", "player_accels", "player_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def build_player_comparison_figure(player_summary, high_cutoff):
    """Horizontal player-load view with high / low group separation visually highlighted."""
    if player_summary.empty:
        return empty_figure("No player workload rows match the selected filters.")

    if not bool(player_summary["comparison_ready"].iloc[0]):
        return empty_figure(
            "Player comparison needs at least 5 players after the active filters."
        )

    ordered = player_summary.sort_values("group_load_score", ascending=True).copy()
    figure = go.Figure()
    trace_order = [
        ("Lower than group", "Lower workload", C_BLUE),
        ("Within group", "Within group", C_NAVY),
        ("Higher than group", "Higher workload", C_RED),
    ]

    for flag, label, color in trace_order:
        subset = ordered[ordered["load_flag"] == flag].copy()
        if subset.empty:
            continue

        custom = np.column_stack([
            subset["player_hsr_m"].fillna(0).round(1),
            subset["hsr_group_pct"].fillna(0).round(0),
            subset["hsr_vs_group_median_pct"].fillna(0).round(1),
            subset["player_accels"].fillna(0).round(1),
            subset["accel_group_pct"].fillna(0).round(0),
            subset["accels_vs_group_median_pct"].fillna(0).round(1),
            subset["player_duration_min"].fillna(0).round(1),
            subset["player_hsr_per_min"].fillna(0).round(2),
            subset["player_accels_per_min"].fillna(0).round(2),
            subset["group_rate_score"].fillna(0).round(1),
            subset["rate_flag"].astype(str),
        ])

        figure.add_trace(go.Bar(
            name=label,
            orientation="h",
            y=subset["player_name"],
            x=subset["group_load_score"],
            marker={"color": color},
            customdata=custom,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Group load score: %{x:.1f}<br>"
                "HSR: %{customdata[0]:.1f} m (%{customdata[1]:.0f}th pct; %{customdata[2]:+.1f}% vs median)<br>"
                "Accelerations: %{customdata[3]:.0f} (%{customdata[4]:.0f}th pct; %{customdata[5]:+.1f}% vs median)<br>"
                "Filtered duration: %{customdata[6]:.1f} min<br>"
                "HSR/min: %{customdata[7]:.2f}; Accels/min: %{customdata[8]:.2f}<br>"
                "Rate score: %{customdata[9]:.1f} — %{customdata[10]}<extra></extra>"
            ),
        ))

    group_size = int(ordered["group_players"].iloc[0])
    figure.update_layout(
        barmode="overlay",
        template="plotly_white",
        paper_bgcolor=C_WHITE,
        plot_bgcolor=C_WHITE,
        height=max(360, min(900, 42 * len(ordered) + 135)),
        margin={"l": 140, "r": 25, "t": 26, "b": 54},
        legend={"orientation": "h", "y": 1.10, "x": 0},
        hoverlabel={"bgcolor": C_WHITE, "font": {"color": C_TEXT}},
        xaxis={
            "title": (
                "Within-group total load score (0–100; "
                "blend of HSR and accelerations percentile ranks)"
            ),
            "range": [0, 105],
            "gridcolor": "#EAEFF5",
            "zeroline": False,
        },
        yaxis={"title": "", "showgrid": False},
    )
    figure.add_vline(x=50, line_dash="dot", line_color="#94A3B8", line_width=1)
    figure.add_vline(x=high_cutoff, line_dash="dot", line_color=C_RED, line_width=1)
    figure.add_vline(x=100 - high_cutoff, line_dash="dot", line_color=C_BLUE, line_width=1)
    figure.add_annotation(
        x=50,
        y=1.04,
        xref="x",
        yref="paper",
        text=f"Group median area · n = {group_size}",
        showarrow=False,
        font={"size": 10, "color": C_MUTED},
    )
    return figure

def add_daily_intensity_score(daily_summary, hsr_weight_pct):
    if daily_summary.empty:
        return daily_summary

    output = daily_summary.copy()
    hsr_weight = max(0, min(100, float(hsr_weight_pct))) / 100
    accel_weight = 1 - hsr_weight
    output["hsr_intensity_pct"] = percentile_values(
        output["typical_hsr_per_min"],
        output["typical_hsr_per_min"],
    )
    output["accel_intensity_pct"] = percentile_values(
        output["typical_accels_per_min"],
        output["typical_accels_per_min"],
    )
    output["daily_intensity_score"] = (
        hsr_weight * output["hsr_intensity_pct"].fillna(0)
        + accel_weight * output["accel_intensity_pct"].fillna(0)
    ).round(1)
    return output


def empty_figure(message):
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": C_MUTED},
    )
    figure.update_layout(
        height=410,
        template="plotly_white",
        paper_bgcolor=C_WHITE,
        plot_bgcolor=C_WHITE,
        margin={"l": 40, "r": 25, "t": 25, "b": 40},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def build_timeline_figure(blocks, hsr_weight_pct):
    if blocks.empty:
        return empty_figure("No chartable drill blocks match the current filters.")

    figure = go.Figure()
    custom = np.column_stack([
        blocks["drill_name_display"].astype(str),
        blocks["intensity_band"].astype(str),
        blocks["median_duration_min"].fillna(0).round(2),
        blocks["median_hsr_m"].fillna(0).round(1),
        blocks["median_accels"].fillna(0).round(1),
        blocks["median_hsr_per_min"].fillna(0).round(2),
        blocks["median_accels_per_min"].fillna(0).round(2),
        blocks["player_count"].fillna(0).astype(int),
    ])

    figure.add_trace(go.Bar(
        name="FCL field drills",
        x=blocks["display_duration_min"],
        base=blocks["timeline_start_min"],
        y=blocks["intensity_score"],
        marker={"color": C_NAVY},
        customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Band: %{customdata[1]}<br>"
            "Typical duration: %{customdata[2]:.1f} min<br>"
            "Typical HSR: %{customdata[3]:.1f} m<br>"
            "Typical accelerations: %{customdata[4]:.1f}<br>"
            "HSR/min: %{customdata[5]:.2f}<br>"
            "Accels/min: %{customdata[6]:.2f}<br>"
            "Players represented: %{customdata[7]}<extra></extra>"
        ),
    ))

    # Label only blocks long enough to keep the time-line readable.
    labels = blocks[blocks["display_duration_min"] >= 2.0]
    if not labels.empty:
        figure.add_trace(go.Scatter(
            x=(labels["timeline_start_min"] + labels["display_duration_min"] / 2),
            y=labels["intensity_score"] + 3,
            mode="text",
            text=labels["timeline_label"],
            textposition="top center",
            textfont={"size": 10, "color": C_TEXT},
            hoverinfo="skip",
            showlegend=False,
            cliponaxis=False,
        ))

    figure.update_layout(
        barmode="overlay",
        template="plotly_white",
        paper_bgcolor=C_WHITE,
        plot_bgcolor=C_WHITE,
        height=490,
        margin={"l": 58, "r": 25, "t": 40, "b": 64},
        showlegend=False,
        hoverlabel={"bgcolor": C_WHITE, "font": {"color": C_TEXT}},
        xaxis={
            "title": "Estimated session timeline (minutes; drill order inferred from the API export)",
            "rangemode": "tozero",
            "showgrid": True,
            "gridcolor": "#EAEFF5",
            "zeroline": False,
        },
        yaxis={
            "title": f"Relative intensity (0–100; {int(hsr_weight_pct)}% HSR/min / {100-int(hsr_weight_pct)}% Accel/min)",
            "range": [0, max(105, float(blocks["intensity_score"].max()) + 15)],
            "showgrid": True,
            "gridcolor": "#EAEFF5",
            "zeroline": False,
        },
    )
    figure.add_hline(y=35, line_dash="dot", line_color="#CBD5E1", line_width=1)
    figure.add_hline(y=65, line_dash="dot", line_color="#CBD5E1", line_width=1)
    figure.add_hline(y=85, line_dash="dot", line_color="#CBD5E1", line_width=1)
    return figure


def build_daily_overview_figure(daily_summary, selected_date, selected_session):
    if daily_summary.empty:
        return empty_figure("No daily summaries match the current filters.")

    labels = daily_summary.apply(
        lambda row: (
            row["date"]
            if not row["session_name"]
            else f'{row["date"]} — {row["session_name"]}'
        ),
        axis=1,
    )
    selected_mask = (
        (daily_summary["date"].astype(str) == str(selected_date))
        & (daily_summary["session_name"].astype(str) == str(selected_session))
    )
    colors = np.where(selected_mask, C_RED, C_NAVY)

    custom = np.column_stack([
        daily_summary["typical_hsr_m"].fillna(0).round(1),
        daily_summary["typical_accels"].fillna(0).round(1),
        daily_summary["typical_duration_min"].fillna(0).round(1),
        daily_summary["typical_hsr_per_min"].fillna(0).round(2),
        daily_summary["typical_accels_per_min"].fillna(0).round(2),
        daily_summary["players"].fillna(0).astype(int),
    ])

    figure = go.Figure(go.Bar(
        x=labels,
        y=daily_summary["daily_intensity_score"],
        marker={"color": colors},
        customdata=custom,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Daily intensity: %{y:.1f}<br>"
            "Typical HSR: %{customdata[0]:.1f} m<br>"
            "Typical accelerations: %{customdata[1]:.1f}<br>"
            "Typical duration: %{customdata[2]:.1f} min<br>"
            "HSR/min: %{customdata[3]:.2f}<br>"
            "Accels/min: %{customdata[4]:.2f}<br>"
            "Players represented: %{customdata[5]}<extra></extra>"
        ),
    ))
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=C_WHITE,
        plot_bgcolor=C_WHITE,
        height=330,
        margin={"l": 55, "r": 20, "t": 20, "b": 90},
        xaxis={"tickangle": -35, "showgrid": False},
        yaxis={
            "title": "Relative day intensity (0–100)",
            "range": [0, 105],
            "gridcolor": "#EAEFF5",
            "zeroline": False,
        },
        showlegend=False,
    )
    return figure



def resolve_analysis_period(range_mode, selected_week_start=None, custom_start=None, custom_end=None):
    """Return normalized inclusive period start/end timestamps for the analysis controls."""
    if range_mode == "custom":
        start = pd.to_datetime(custom_start, errors="coerce")
        end = pd.to_datetime(custom_end, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            return None, None
        start, end = start.normalize(), end.normalize()
        if end < start:
            return None, None
        return start, end

    start = pd.to_datetime(selected_week_start, errors="coerce")
    if pd.isna(start):
        return None, None
    start = start.normalize()
    return start, start + pd.Timedelta(days=6)


def format_analysis_period(period_start, period_end):
    if period_start is None or period_end is None:
        return "Choose an analysis period"
    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end)
    if start == end:
        return start.strftime("%A, %b %-d, %Y")
    if start.year == end.year:
        return f"{start.strftime('%b %-d')} – {end.strftime('%b %-d, %Y')}"
    return f"{start.strftime('%b %-d, %Y')} – {end.strftime('%b %-d, %Y')}"


def build_selected_week_daily_summary(data, selected_players, period_start, period_end):
    """
    Build athlete-first FCL averages for every calendar day in the selected period.

    Each athlete's filtered work is totaled within a day before the group mean is
    calculated. This avoids overweighting athletes who happen to have more drill
    rows, blocks, or sessions logged on a day.
    """
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected_players = [] if selected_players is None else list(selected_players)
    if not selected_players or period_start is None or period_end is None:
        return pd.DataFrame(), pd.DataFrame()

    period_start = pd.to_datetime(period_start, errors="coerce")
    period_end = pd.to_datetime(period_end, errors="coerce")
    if pd.isna(period_start) or pd.isna(period_end):
        return pd.DataFrame(), pd.DataFrame()
    period_start, period_end = period_start.normalize(), period_end.normalize()
    if period_end < period_start:
        return pd.DataFrame(), pd.DataFrame()

    use = data.copy()
    use["_day_dt"] = pd.to_datetime(use["date"], errors="coerce").dt.normalize()
    use = use[use["_day_dt"].notna()].copy()
    use = use[
        use["player_name"].astype(str).isin({str(name) for name in selected_players})
        & use["_day_dt"].between(period_start, period_end)
    ].copy()

    numeric_cols = [
        "hsr_distance_m", "n_accelerations", "duration_min", "total_distance_m",
        "hmld_m", "sprint_distance_m", "mechanical_load", "top_speed_ms", "max_accel_ms2",
    ]
    for column in numeric_cols:
        if column not in use.columns:
            use[column] = np.nan
        use[column] = pd.to_numeric(use[column], errors="coerce")

    calendar = pd.DataFrame({"day_date": pd.date_range(period_start, period_end, freq="D")})
    if use.empty:
        calendar["players_included"] = 0
        calendar["avg_player_sessions"] = np.nan
        for metric in WEEKLY_METRICS:
            calendar[metric] = np.nan
        calendar["day_label"] = calendar["day_date"].dt.strftime("%a %b %-d")
        return calendar, pd.DataFrame()

    player_day = (
        use.groupby(["_day_dt", "player_name"], as_index=False)
        .agg(
            daily_hsr_m=("hsr_distance_m", "sum"),
            daily_accels=("n_accelerations", "sum"),
            daily_duration_min=("duration_min", "sum"),
            daily_total_distance_m=("total_distance_m", "sum"),
            daily_hmld_m=("hmld_m", "sum"),
            daily_sprint_distance_m=("sprint_distance_m", "sum"),
            daily_mechanical_load=("mechanical_load", "sum"),
            daily_top_speed_ms=("top_speed_ms", "max"),
            daily_max_accel_ms2=("max_accel_ms2", "max"),
            sessions=("session_id", "nunique"),
        )
    )
    player_day["daily_hsr_per_min"] = player_day.apply(
        lambda row: safe_divide(row["daily_hsr_m"], row["daily_duration_min"]), axis=1
    )
    player_day["daily_accels_per_min"] = player_day.apply(
        lambda row: safe_divide(row["daily_accels"], row["daily_duration_min"]), axis=1
    )

    daily = (
        player_day.groupby("_day_dt", as_index=False)
        .agg(
            players_included=("player_name", "nunique"),
            avg_player_sessions=("sessions", "mean"),
            avg_player_hsr_m=("daily_hsr_m", "mean"),
            avg_player_accels=("daily_accels", "mean"),
            avg_player_duration_min=("daily_duration_min", "mean"),
            avg_player_total_distance_m=("daily_total_distance_m", "mean"),
            avg_player_hmld_m=("daily_hmld_m", "mean"),
            avg_player_sprint_distance_m=("daily_sprint_distance_m", "mean"),
            avg_player_mechanical_load=("daily_mechanical_load", "mean"),
            avg_player_hsr_per_min=("daily_hsr_per_min", "mean"),
            avg_player_accels_per_min=("daily_accels_per_min", "mean"),
            avg_player_top_speed_ms=("daily_top_speed_ms", "mean"),
            avg_player_max_accel_ms2=("daily_max_accel_ms2", "mean"),
        )
        .rename(columns={"_day_dt": "day_date"})
    )

    result = calendar.merge(daily, on="day_date", how="left")
    result["players_included"] = result["players_included"].fillna(0).astype(int)
    result["day_label"] = result["day_date"].dt.strftime("%a %b %-d")
    return result, player_day

def build_selected_week_daily_figure(day_summary, selected_metrics):
    """Render one polished bar-chart panel per selected metric across Monday–Sunday."""
    selected_metrics = [
        metric for metric in (selected_metrics or [])
        if metric in WEEKLY_METRICS and metric in day_summary.columns
    ]
    if day_summary.empty:
        return empty_figure("Choose a valid analysis period and one or more FCL players to view daily averages.")
    if not selected_metrics:
        return empty_figure("Choose one or more metrics to graph.")

    # Keep the selected Monday–Sunday order as a categorical axis so every day label
    # stays visible directly underneath its corresponding bar.
    plot_df = day_summary.copy()
    plot_df["day_label"] = plot_df["day_date"].dt.strftime("%a<br>%b %-d")
    day_order = plot_df["day_label"].tolist()

    rows = len(selected_metrics)
    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.10, 0.18 / max(rows, 1)),
        subplot_titles=[WEEKLY_METRICS[metric]["label"] for metric in selected_metrics],
    )

    palette = [C_RED, C_NAVY, C_BLUE, C_GREEN, C_AMBER, "#7C3AED", "#0F766E", "#BE123C", "#334155", "#9333EA", "#0369A1"]
    no_data_color = "#E2E8F0"
    for row_number, metric in enumerate(selected_metrics, start=1):
        config = WEEKLY_METRICS[metric]
        values = pd.to_numeric(plot_df[metric], errors="coerce")
        represented = plot_df["players_included"].fillna(0).astype(int)
        colors = [
            palette[(row_number - 1) % len(palette)] if n_players > 0 else no_data_color
            for n_players in represented
        ]
        custom = np.column_stack([
            represented,
            pd.to_numeric(plot_df["avg_player_sessions"], errors="coerce").round(1),
            plot_df["day_date"].dt.strftime("%A, %b %-d"),
        ])
        digits = int(config["digits"])
        value_template = f"%{{y:,.{digits}f}}"

        figure.add_trace(
            go.Bar(
                x=plot_df["day_label"],
                y=values.fillna(0),
                name=config["label"],
                marker={"color": colors, "line": {"color": "#FFFFFF", "width": 1.5}},
                width=0.68,
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[2]}</b><br>"
                    f"{config['label']}: {value_template}<br>"
                    "Selected players represented: %{customdata[0]}<br>"
                    "Avg sessions per represented player: %{customdata[1]:.1f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row_number,
            col=1,
        )
        figure.update_yaxes(
            title_text=config["axis"],
            gridcolor="#EAEFF5",
            zeroline=False,
            rangemode="tozero",
            row=row_number,
            col=1,
        )

    # Show the weekday/date label underneath every bar and preserve all seven days,
    # including days with no selected-player data.
    figure.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=day_order,
        tickmode="array",
        tickvals=day_order,
        ticktext=day_order,
        title_text="Day of week",
        showgrid=False,
        tickfont={"size": 12, "color": C_TEXT},
        row=rows,
        col=1,
    )
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=C_WHITE,
        plot_bgcolor=C_WHITE,
        height=max(365, 230 * rows),
        margin={"l": 78, "r": 28, "t": 50, "b": 68},
        bargap=0.30,
        hoverlabel={"bgcolor": C_WHITE, "font": {"color": C_TEXT}},
    )
    return figure


def build_selected_week_daily_table(day_summary, selected_metrics):
    """Create the day-by-day table for the currently selected week and metrics."""
    if day_summary.empty:
        return [], []

    selected_metrics = [
        metric for metric in (selected_metrics or [])
        if metric in WEEKLY_METRICS and metric in day_summary.columns
    ]
    out = day_summary.copy()
    out["day"] = out["day_date"].dt.strftime("%a %b %-d")
    columns = [
        {"name": "Day", "id": "day"},
        {"name": "Players", "id": "players_included"},
        {"name": "Avg sessions/player", "id": "avg_player_sessions"},
    ]
    ids = ["day", "players_included", "avg_player_sessions"]

    for metric in selected_metrics:
        config = WEEKLY_METRICS[metric]
        columns.append({"name": config["label"], "id": metric})
        ids.append(metric)

    show = out[[column for column in ids if column in out.columns]].copy()
    for column in show.columns:
        if column != "day":
            digits = int(WEEKLY_METRICS.get(column, {"digits": 1})["digits"]) if column in WEEKLY_METRICS else 1
            show[column] = pd.to_numeric(show[column], errors="coerce").round(digits)
    return columns, clean_for_json(show)



def build_selected_week_drill_summary(data, selected_players, period_start, period_end, selected_drills=None):
    """
    Drill-level selected-period view. Each athlete's work is totaled within a
    date + drill before averages are calculated across selected athletes.
    """
    if data.empty:
        return pd.DataFrame()

    selected_players = [] if selected_players is None else [str(value) for value in selected_players]
    if not selected_players or period_start is None or period_end is None:
        return pd.DataFrame()

    period_start = pd.to_datetime(period_start, errors="coerce")
    period_end = pd.to_datetime(period_end, errors="coerce")
    if pd.isna(period_start) or pd.isna(period_end):
        return pd.DataFrame()
    period_start, period_end = period_start.normalize(), period_end.normalize()
    if period_end < period_start:
        return pd.DataFrame()

    selected_drills = [] if selected_drills is None else [str(value) for value in selected_drills]
    use = data.copy()
    use["_day_dt"] = pd.to_datetime(use["date"], errors="coerce").dt.normalize()
    use = use[
        use["_day_dt"].notna()
        & use["_day_dt"].between(period_start, period_end)
        & use["player_name"].astype(str).isin(set(selected_players))
    ].copy()
    if selected_drills:
        use = use[use["drill_name_display"].astype(str).isin(set(selected_drills))].copy()
    if use.empty:
        return pd.DataFrame()

    numeric_cols = [
        "hsr_distance_m", "n_accelerations", "duration_min", "total_distance_m",
        "hmld_m", "sprint_distance_m", "mechanical_load", "top_speed_ms", "max_accel_ms2",
    ]
    for column in numeric_cols:
        if column not in use.columns:
            use[column] = np.nan
        use[column] = pd.to_numeric(use[column], errors="coerce")

    player_drill = (
        use.groupby(["_day_dt", "player_name", "drill_name_display"], as_index=False)
        .agg(
            daily_hsr_m=("hsr_distance_m", "sum"),
            daily_accels=("n_accelerations", "sum"),
            daily_duration_min=("duration_min", "sum"),
            daily_total_distance_m=("total_distance_m", "sum"),
            daily_hmld_m=("hmld_m", "sum"),
            daily_sprint_distance_m=("sprint_distance_m", "sum"),
            daily_mechanical_load=("mechanical_load", "sum"),
            daily_top_speed_ms=("top_speed_ms", "max"),
            daily_max_accel_ms2=("max_accel_ms2", "max"),
        )
    )
    player_drill["daily_hsr_per_min"] = player_drill.apply(
        lambda row: safe_divide(row["daily_hsr_m"], row["daily_duration_min"]), axis=1
    )
    player_drill["daily_accels_per_min"] = player_drill.apply(
        lambda row: safe_divide(row["daily_accels"], row["daily_duration_min"]), axis=1
    )

    summary = (
        player_drill.groupby(["_day_dt", "drill_name_display"], as_index=False)
        .agg(
            players_included=("player_name", "nunique"),
            avg_player_hsr_m=("daily_hsr_m", "mean"),
            avg_player_accels=("daily_accels", "mean"),
            avg_player_duration_min=("daily_duration_min", "mean"),
            avg_player_total_distance_m=("daily_total_distance_m", "mean"),
            avg_player_hmld_m=("daily_hmld_m", "mean"),
            avg_player_sprint_distance_m=("daily_sprint_distance_m", "mean"),
            avg_player_mechanical_load=("daily_mechanical_load", "mean"),
            avg_player_hsr_per_min=("daily_hsr_per_min", "mean"),
            avg_player_accels_per_min=("daily_accels_per_min", "mean"),
            avg_player_top_speed_ms=("daily_top_speed_ms", "mean"),
            avg_player_max_accel_ms2=("daily_max_accel_ms2", "mean"),
        )
        .rename(columns={"_day_dt": "day_date"})
        .sort_values(["day_date", "drill_name_display"])
        .reset_index(drop=True)
    )
    summary["day_label"] = summary["day_date"].dt.strftime("%a %b %-d")
    return summary

def build_week_drill_figure(drill_summary, selected_metrics, period_start, period_end):
    """Render a drill-by-day heatmap across the selected analysis period."""
    selected_metrics = [
        metric for metric in (selected_metrics or [])
        if metric in WEEKLY_METRICS and metric in drill_summary.columns
    ]
    if drill_summary.empty:
        return empty_figure("No drill-level work matches the selected dates, players, and drill filter.")
    if not selected_metrics:
        return empty_figure("Choose one or more metrics to graph.")

    period_start = pd.to_datetime(period_start, errors="coerce")
    period_end = pd.to_datetime(period_end, errors="coerce")
    if pd.isna(period_start) or pd.isna(period_end) or period_end < period_start:
        return empty_figure("Choose a valid analysis period to view drill-level exposure.")
    dates = pd.date_range(period_start.normalize(), period_end.normalize(), freq="D")

    order = (
        drill_summary.groupby("drill_name_display", as_index=False)
        .agg(first_seen=("day_date", "min"), total_hsr=("avg_player_hsr_m", "sum"))
        .sort_values(["first_seen", "total_hsr", "drill_name_display"], ascending=[True, False, True])
    )
    drills = order["drill_name_display"].astype(str).tolist()
    rows = len(selected_metrics)
    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.09, 0.16 / max(rows, 1)),
        subplot_titles=[WEEKLY_METRICS[metric]["label"] for metric in selected_metrics],
    )
    colorscales = ["Blues", "Reds", "Purples", "Greens", "Oranges", "Teal"]

    for row_number, metric in enumerate(selected_metrics, start=1):
        config = WEEKLY_METRICS[metric]
        pivot = (
            drill_summary.pivot_table(index="drill_name_display", columns="day_date", values=metric, aggfunc="mean")
            .reindex(index=drills, columns=dates)
        )
        player_pivot = (
            drill_summary.pivot_table(index="drill_name_display", columns="day_date", values="players_included", aggfunc="max")
            .reindex(index=drills, columns=dates)
        )
        figure.add_trace(
            go.Heatmap(
                z=pivot.to_numpy(),
                x=dates,
                y=drills,
                customdata=player_pivot.to_numpy(),
                colorscale=colorscales[(row_number - 1) % len(colorscales)],
                colorbar={"title": config["axis"], "len": max(0.25, 0.80 / rows), "y": 1 - ((row_number - 0.5) / rows)},
                hovertemplate=(
                    "<b>%{y}</b><br>%{x|%A, %b %d}<br>"
                    f"{config['label']}: %{{z:,.{int(config['digits'])}f}}<br>"
                    "Players represented: %{customdata}<extra></extra>"
                ),
                hoverongaps=False,
                showscale=True,
            ),
            row=row_number,
            col=1,
        )
        figure.update_yaxes(
            categoryorder="array",
            categoryarray=drills[::-1],
            tickfont={"size": 10},
            row=row_number,
            col=1,
        )

    figure.update_xaxes(
        tickformat="%a<br>%b %d",
        dtick=86400000,
        showgrid=False,
        row=rows,
        col=1,
    )
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=C_WHITE,
        plot_bgcolor=C_WHITE,
        height=max(420, min(1250, 190 * rows + 24 * len(drills) * rows)),
        margin={"l": 165, "r": 88, "t": 58, "b": 52},
        hoverlabel={"bgcolor": C_WHITE, "font": {"color": C_TEXT}},
    )
    return figure


def build_week_drill_table(drill_summary, selected_metrics):
    if drill_summary.empty:
        return [], []
    selected_metrics = [
        metric for metric in (selected_metrics or [])
        if metric in WEEKLY_METRICS and metric in drill_summary.columns
    ]
    out = drill_summary.copy()
    out["day"] = out["day_date"].dt.strftime("%a %b %-d")
    columns = [
        {"name": "Day", "id": "day"},
        {"name": "Drill", "id": "drill_name_display"},
        {"name": "Players", "id": "players_included"},
    ]
    ids = ["day", "drill_name_display", "players_included"]
    for metric in selected_metrics:
        columns.append({"name": WEEKLY_METRICS[metric]["label"], "id": metric})
        ids.append(metric)
    show = out[[column for column in ids if column in out.columns]].copy()
    for column in show.columns:
        if column not in {"day", "drill_name_display"}:
            digits = int(WEEKLY_METRICS.get(column, {"digits": 0})["digits"]) if column in WEEKLY_METRICS else 0
            show[column] = pd.to_numeric(show[column], errors="coerce").round(digits)
    return columns, clean_for_json(show)


def fmt(value, digits=1, fallback="—"):
    try:
        if pd.isna(value):
            return fallback
        return f"{float(value):,.{digits}f}"
    except Exception:
        return fallback



# ── STREAMLIT UI ─────────────────────────────────────────────────────────────
TABLE_COLUMN_LABELS = {
    "timeline_start": "Start", "timeline_end": "End", "drill_name_display": "Drill",
    "intensity_score": "Intensity", "intensity_band": "Band", "player_count": "Players",
    "median_duration_min": "Min", "median_hsr_m": "HSR m", "median_accels": "Accel",
    "median_hsr_per_min": "HSR/min", "median_accels_per_min": "Accel/min",
    "p90_top_speed_ms": "P90 speed m/s",
    "player_name": "Player", "load_flag": "Load flag", "group_load_score": "Load score",
    "player_hsr_m": "HSR m", "hsr_group_pct": "HSR pct", "hsr_vs_group_median_pct": "HSR vs med",
    "player_accels": "Accels", "accel_group_pct": "Accel pct", "accels_vs_group_median_pct": "Accel vs med",
    "player_duration_min": "Field min", "player_hsr_per_min": "HSR/min",
    "player_accels_per_min": "Accel/min", "rate_flag": "Rate flag",
}


def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {C_BG}; }}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
        .hero {{
            background: linear-gradient(135deg, {C_NAVY} 0%, #0A2A5B 100%);
            border-bottom: 4px solid {C_RED}; border-radius: 0 0 18px 18px;
            padding: 24px 28px; margin: -1rem -1rem 1.2rem -1rem; color: white;
        }}
        .hero-kicker {{ color: #FCA5A5; font-size: 0.72rem; font-weight: 800; letter-spacing: .16em; }}
        .hero-title {{ font-size: 2rem; font-weight: 800; margin-top: .15rem; }}
        .hero-copy {{ color: #CBD5E1; font-size: .9rem; margin-top: .25rem; }}
        .section-kicker {{ color: {C_RED}; font-weight: 800; letter-spacing: .14em; font-size: .68rem; text-transform: uppercase; }}
        div[data-testid="stMetric"] {{ background: white; border: 1px solid {C_BORDER}; border-radius: 12px; padding: 10px 12px; }}
        div[data-testid="stPlotlyChart"] {{ background: white; border-radius: 14px; }}
        .small-muted {{ color: {C_MUTED}; font-size: .82rem; line-height: 1.5; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    defaults = {
        "raw_data": pd.DataFrame(),
        "roster_df": pd.DataFrame(columns=["player_name", "position_raw", "position_group"]),
        "pull_status": "Choose a date range, then pull FCL session data.",
        "pull_status_kind": "info",
        "roster_status": "",
        "roster_editor_version": 0,
        "pdf_bytes": None,
        "pdf_filename": None,
        "pdf_status": "",
        "pull_start": date.today() - timedelta(weeks=4),
        "pull_end": date.today(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def status_message(message, kind="info"):
    fn = {"success": st.success, "warning": st.warning, "error": st.error}.get(kind, st.info)
    fn(message)


def current_roster_rows():
    roster = st.session_state.get("roster_df")
    if roster is None or roster.empty:
        return []
    return roster.to_dict("records")


def display_dataframe(df, *, height=None):
    if df is None or df.empty:
        st.caption("No rows to display.")
        return
    out = df.rename(columns={c: TABLE_COLUMN_LABELS.get(c, c) for c in df.columns})
    st.dataframe(out, use_container_width=True, hide_index=True, height=height)


def setup_pull_section():
    with st.container(border=True):
        st.markdown('<div class="section-kicker">Data pull</div>', unsafe_allow_html=True)
        button_cols = st.columns([1, 1, 1, 1, 2.3])
        quicks = [
            ("2 weeks", timedelta(weeks=2)), ("4 weeks", timedelta(weeks=4)),
            ("8 weeks", timedelta(weeks=8)), ("3 months", timedelta(days=90)),
        ]
        for col, (label, delta) in zip(button_cols[:4], quicks):
            if col.button(label, use_container_width=True):
                st.session_state.pull_start = date.today() - delta
                st.session_state.pull_end = date.today()
                st.rerun()

        date_cols = st.columns([1.2, 1.2, 1.35, 1])
        start_date = date_cols[0].date_input("Start date", key="pull_start", max_value=date.today())
        end_date = date_cols[1].date_input("End date", key="pull_end", max_value=date.today())
        sync_sheet = date_cols[2].toggle("Append raw pull to Google Sheet", value=False)
        pull_clicked = date_cols[3].button("Pull FCL sessions", type="primary", use_container_width=True)

        if pull_clicked:
            if not API_KEY:
                st.session_state.pull_status = "STATSPORTS_API_KEY is missing from Streamlit Secrets."
                st.session_state.pull_status_kind = "error"
            elif end_date < start_date:
                st.session_state.pull_status = "The end date must be on or after the start date."
                st.session_state.pull_status_kind = "error"
            else:
                total_days = (end_date - start_date).days + 1
                try:
                    with st.spinner(f"Pulling {total_days} day(s) of FCL STATSports data..."):
                        raw = pull_range(start_date.isoformat(), end_date.isoformat())
                    st.session_state.raw_data = raw
                    if raw.empty:
                        st.session_state.roster_df = pd.DataFrame(columns=["player_name", "position_raw", "position_group"])
                        st.session_state.pull_status = (
                            "Pull complete — no FCL drill rows were identified for this range. "
                            "Only sessions identified as FCL/Florida Complex are accepted."
                        )
                        st.session_state.pull_status_kind = "warning"
                    else:
                        existing = current_roster_rows()
                        st.session_state.roster_df = pd.DataFrame(build_roster_assignment_rows(raw, existing))
                        rows = len(raw)
                        players = raw["player_name"].nunique()
                        sessions = raw[["date", "session_name"]].drop_duplicates().shape[0]
                        sheet_note = "Google Sheet sync not requested."
                        kind = "success"
                        if sync_sheet:
                            try:
                                result = write_to_sheets_append_only(raw)
                                sheet_note = (
                                    f"Sheet sync: {result['appended']:,} appended; "
                                    f"{result['already_present']:,} already present; 0 existing rows changed."
                                )
                            except Exception as exc:
                                sheet_note = f"Raw data pulled, but Google Sheet sync failed: {exc}"
                                kind = "warning"
                        st.session_state.pull_status = (
                            f"FCL-only pull complete — {rows:,} raw rows · {players} players · "
                            f"{sessions} sessions · {total_days} days. {sheet_note}"
                        )
                        st.session_state.pull_status_kind = kind
                except Exception as exc:
                    st.session_state.pull_status = f"Pull failed: {exc}"
                    st.session_state.pull_status_kind = "error"

        status_message(st.session_state.pull_status, st.session_state.pull_status_kind)


def render_roster_tab(raw):
    st.subheader("Player-by-player roster tagging")
    st.caption(
        "Assign each athlete a workload group. On Streamlit Cloud, saved tags are written to the "
        f"'{ROSTER_SHEET_TAB_NAME}' tab in the configured Google Sheet."
    )
    roster = st.session_state.roster_df.copy()
    if roster.empty:
        st.info("Pull FCL data first.")
        return

    total = len(roster)
    tagged = int((~roster["position_group"].isin(["", "Unassigned"])).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Players tagged", f"{tagged} / {total}")
    m2.metric("Still unassigned", total - tagged)
    m3.metric("Tagged %", f"{(100 * tagged / total if total else 0):.0f}%")

    st.markdown("#### Quick assignment")
    c1, c2, c3 = st.columns([1.5, 1, 0.8])
    names = roster["player_name"].astype(str).tolist()
    quick_player = c1.selectbox("Player", names, key="quick_roster_player")
    current_group = roster.loc[roster["player_name"].eq(quick_player), "position_group"].iloc[0]
    quick_group = c2.selectbox(
        "Position group", POSITION_GROUPS,
        index=POSITION_GROUPS.index(current_group) if current_group in POSITION_GROUPS else POSITION_GROUPS.index("Unassigned"),
        key="quick_roster_group",
    )
    if c3.button("Save player tag", type="primary", use_container_width=True):
        roster.loc[roster["player_name"].eq(quick_player), "position_group"] = quick_group
        st.session_state.roster_df = roster
        try:
            count, location = save_position_groups(roster.to_dict("records"))
            st.session_state.roster_status = f"Saved {quick_player} as {quick_group}. {count} tags saved to {location}."
            st.session_state.roster_editor_version += 1
            st.rerun()
        except Exception as exc:
            st.session_state.roster_status = f"Tag updated in this session, but persistence failed: {exc}"

    st.markdown("#### Roster review")
    edited = st.data_editor(
        roster,
        use_container_width=True,
        hide_index=True,
        disabled=["player_name", "position_raw"],
        column_config={
            "player_name": st.column_config.TextColumn("Player"),
            "position_raw": st.column_config.TextColumn("API position"),
            "position_group": st.column_config.SelectboxColumn("Position group", options=POSITION_GROUPS, required=True),
        },
        key=f"roster_editor_{st.session_state.roster_editor_version}",
    )
    st.session_state.roster_df = edited.copy()
    if st.button("Save all roster edits", key="save_all_roster"):
        try:
            count, location = save_position_groups(edited.to_dict("records"))
            st.session_state.roster_status = f"Saved {count} player position-group tags to {location}."
        except Exception as exc:
            st.session_state.roster_status = f"Roster edits are active in this session, but persistence failed: {exc}"

    if st.session_state.roster_status:
        st.success(st.session_state.roster_status)


def practice_controls(raw):
    days = sorted(raw["date"].dropna().astype(str).unique().tolist(), reverse=True)
    if not days:
        return None
    c1, c2, c3, c4 = st.columns([1, 1.7, 1.25, 1.2])
    selected_day = c1.selectbox("Day", days, key="practice_day")
    subset = raw[raw["date"].astype(str).eq(str(selected_day))]
    sessions = subset[["session_id", "session_name"]].drop_duplicates().sort_values("session_name")
    session_ids = sessions["session_id"].astype(str).tolist()
    label_map = {str(r.session_id): (str(r.session_name) if str(r.session_name) else "(Unnamed session)") for r in sessions.itertuples()}
    selected_session_id = c2.selectbox(
        "Session", session_ids, format_func=lambda x: label_map.get(str(x), str(x)), key="practice_session"
    ) if session_ids else None
    hsr_weight = c3.slider("Intensity blend — HSR %", 0, 100, 60, 5, key="hsr_weight")
    cutoff = c4.selectbox(
        "High / low flag threshold", [75, 85, 90], index=1,
        format_func=lambda x: f"{x}th / {100-x}th percentile", key="flag_cutoff"
    )
    st.caption(f"Intensity score = {hsr_weight}% HSR/min + {100-hsr_weight}% accelerations/min.")
    return selected_day, selected_session_id, hsr_weight, cutoff


def render_practice_tab(raw, exclude_non_field):
    st.subheader("Practice view")
    controls = practice_controls(raw)
    if controls is None:
        st.info("No dates are available.")
        return
    selected_day, selected_session_id, hsr_weight, cutoff = controls

    filtered_all = filter_dashboard_rows(raw, ["exclude_non_field"] if exclude_non_field else [])
    if filtered_all.empty:
        st.warning("No rows remain after the active filters.")
        return

    all_blocks = build_block_summary(filtered_all)
    day_rows = filtered_all[
        filtered_all["date"].astype(str).eq(str(selected_day))
        & filtered_all["session_id"].astype(str).eq(str(selected_session_id))
    ].copy()
    day_blocks = add_intensity_score(build_block_summary(day_rows), all_blocks, hsr_weight)
    player_summary = build_player_session_summary(day_rows, hsr_weight, high_cutoff=int(cutoff))
    daily_summary = add_daily_intensity_score(build_player_daily_summary(filtered_all), hsr_weight)

    session_name = ""
    if not day_rows.empty:
        session_name = str(day_rows["session_name"].iloc[0] or "")
    title = f"{selected_day}" + (f" — {session_name}" if session_name else "")
    st.markdown(f"### {title}")

    selected_daily = daily_summary[
        daily_summary["date"].astype(str).eq(str(selected_day))
        & daily_summary["session_id"].astype(str).eq(str(selected_session_id))
    ]
    daily_row = selected_daily.iloc[0] if not selected_daily.empty else pd.Series(dtype=float)
    comparison_ready = not player_summary.empty and bool(player_summary["comparison_ready"].iloc[0])
    high_count = int((player_summary["load_flag"] == "Higher than group").sum()) if comparison_ready else 0
    low_count = int((player_summary["load_flag"] == "Lower than group").sum()) if comparison_ready else 0
    group_count = int(player_summary["group_players"].iloc[0]) if not player_summary.empty else 0

    metric_values = [
        ("Day intensity", fmt(daily_row.get("daily_intensity_score", np.nan), 0)),
        ("Typical-player HSR", f"{fmt(daily_row.get('typical_hsr_m', np.nan), 0)} m"),
        ("Typical-player accelerations", fmt(daily_row.get("typical_accels", np.nan), 0)),
        ("Typical field duration", f"{fmt(daily_row.get('typical_duration_min', np.nan), 0)} min"),
        ("HSR / min", fmt(daily_row.get("typical_hsr_per_min", np.nan), 2)),
        ("Accelerations / min", fmt(daily_row.get("typical_accels_per_min", np.nan), 2)),
        ("FCL group separation", f"{high_count} high / {low_count} low" if comparison_ready else "Limited"),
    ]
    cols = st.columns(len(metric_values))
    for col, (label, value) in zip(cols, metric_values):
        col.metric(label, value)

    st.caption(
        f"Block height = {hsr_weight}% HSR/min + {100-hsr_weight}% accelerations/min, scored as percentiles "
        "across included blocks in the pulled range. Block width = typical block duration."
    )
    st.plotly_chart(build_timeline_figure(day_blocks, hsr_weight), use_container_width=True, config={"displayModeBar": False})

    st.markdown("#### Practice comparison")
    st.caption("The selected session is highlighted in red so you can compare it with the rest of the pulled range.")
    st.plotly_chart(
        build_daily_overview_figure(daily_summary, selected_day, session_name),
        use_container_width=True, config={"displayModeBar": False},
    )

    st.markdown("#### Player load separation")
    if comparison_ready:
        st.caption(
            f"Each player is compared only with the {group_count} included players in this session. "
            f"Flags use the {cutoff}th / {100-cutoff}th percentile boundaries. These are workload-separation flags, not injury-risk labels."
        )
    else:
        st.caption(f"Only {group_count} player(s) remain. High/low flags turn on with at least 5 players.")
    st.plotly_chart(build_player_comparison_figure(player_summary, int(cutoff)), use_container_width=True, config={"displayModeBar": False})

    if not player_summary.empty:
        player_columns = [
            "player_name", "load_flag", "group_load_score", "player_hsr_m", "hsr_group_pct",
            "hsr_vs_group_median_pct", "player_accels", "accel_group_pct", "accels_vs_group_median_pct",
            "player_duration_min", "player_hsr_per_min", "player_accels_per_min", "rate_flag",
        ]
        display_players = player_summary[[c for c in player_columns if c in player_summary.columns]].copy()
        for col in display_players.columns:
            if col not in {"player_name", "load_flag", "rate_flag"}:
                display_players[col] = pd.to_numeric(display_players[col], errors="coerce").round(2)
        display_dataframe(display_players, height=430)

    st.markdown("#### Practice block details")
    if day_blocks.empty:
        st.caption("No block rows match this session.")
    else:
        display_blocks = day_blocks.copy()
        display_blocks["timeline_start"] = display_blocks["timeline_start_min"].map(lambda x: f"{x:.1f}")
        display_blocks["timeline_end"] = display_blocks["timeline_end_min"].map(lambda x: f"{x:.1f}")
        block_cols = [
            "timeline_start", "timeline_end", "drill_name_display", "intensity_score", "intensity_band",
            "player_count", "median_duration_min", "median_hsr_m", "median_accels",
            "median_hsr_per_min", "median_accels_per_min", "p90_top_speed_ms",
        ]
        display_blocks = display_blocks[[c for c in block_cols if c in display_blocks.columns]].copy()
        for col in display_blocks.columns:
            if col not in {"timeline_start", "timeline_end", "drill_name_display", "intensity_band"}:
                display_blocks[col] = pd.to_numeric(display_blocks[col], errors="coerce").round(2)
        display_dataframe(display_blocks, height=500)


def render_load_analysis_tab(raw, exclude_non_field):
    st.subheader("Load analysis")
    filtered_all = filter_dashboard_rows(raw, ["exclude_non_field"] if exclude_non_field else [])
    if filtered_all.empty:
        st.warning("No rows remain after the active filters.")
        return

    day_values = pd.to_datetime(filtered_all["date"], errors="coerce").dropna().dt.normalize()
    week_starts = sorted((day_values - pd.to_timedelta(day_values.dt.weekday, unit="D")).unique(), reverse=True)
    week_options = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in week_starts]

    c1, c2, c3 = st.columns([1, 1.5, 1])
    range_mode_label = c1.radio("Analysis period", ["Selected week", "Custom dates"], horizontal=True)
    range_mode = "week" if range_mode_label == "Selected week" else "custom"
    selected_week = None
    custom_start = custom_end = None
    if range_mode == "week":
        selected_week = c2.selectbox(
            "Week", week_options,
            format_func=lambda x: f"Week of {pd.Timestamp(x).strftime('%b %-d, %Y')} — {(pd.Timestamp(x)+pd.Timedelta(days=6)).strftime('%b %-d, %Y')}",
        ) if week_options else None
    else:
        min_day, max_day = day_values.min().date(), day_values.max().date()
        selected_range = c2.date_input("Custom dates", value=(max(min_day, max_day - timedelta(days=6)), max_day), min_value=min_day, max_value=max_day)
        if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
            custom_start, custom_end = selected_range[0].isoformat(), selected_range[1].isoformat()
    view_label = c3.radio("View", ["Daily team load", "Drill by drill"], horizontal=True)
    view_mode = "daily" if view_label == "Daily team load" else "drill"

    period_start, period_end = resolve_analysis_period(range_mode, selected_week, custom_start, custom_end)
    if period_start is None or period_end is None:
        st.info("Choose a valid analysis period.")
        return

    roster_map = roster_group_lookup(current_roster_rows())
    present_groups = [g for g in POSITION_GROUPS if g in set(roster_map.values())]
    c4, c5 = st.columns([1, 1.6])
    selected_groups = c4.multiselect("Position group", present_groups, default=present_groups)
    active_groups = set(selected_groups or present_groups)
    all_players = sorted(filtered_all["player_name"].dropna().astype(str).unique().tolist(), key=str.lower)
    available_players = [p for p in all_players if roster_map.get(p, "Unassigned") in active_groups]
    selected_players = c5.multiselect("Players", available_players, default=available_players)

    c6, c7 = st.columns([1.4, 1.4])
    metric_options = list(WEEKLY_METRICS)
    selected_metrics = c6.multiselect(
        "Metrics", metric_options, default=WEEKLY_DEFAULT_METRICS,
        format_func=lambda x: WEEKLY_METRICS[x]["label"],
    )

    use = filtered_all.copy()
    use["_day_dt"] = pd.to_datetime(use["date"], errors="coerce").dt.normalize()
    use = use[use["_day_dt"].between(period_start, period_end)]
    if selected_players:
        use = use[use["player_name"].astype(str).isin(set(selected_players))]
    available_drills = sorted(use.get("drill_name_display", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(), key=str.lower)
    selected_drills = c7.multiselect("Drills for drill-by-drill view", available_drills, default=[])

    if not selected_metrics:
        st.info("Select at least one metric.")
        return

    analysis_period_label = format_analysis_period(period_start, period_end)
    daily_summary, _ = build_selected_week_daily_summary(filtered_all, selected_players, period_start, period_end)
    drill_summary = build_selected_week_drill_summary(filtered_all, selected_players, period_start, period_end, selected_drills)

    if view_mode == "drill":
        columns, table_data = build_week_drill_table(drill_summary, selected_metrics)
        figure = build_week_drill_figure(drill_summary, selected_metrics, period_start, period_end)
        rows = len(drill_summary)
        copy = (
            f"{analysis_period_label}. Each cell is the average selected-player exposure for that drill on that day. "
            "Hover for exact workload and player count."
        )
    else:
        columns, table_data = build_selected_week_daily_table(daily_summary, selected_metrics)
        figure = build_selected_week_daily_figure(daily_summary, selected_metrics)
        rows = len(daily_summary)
        copy = (
            f"{analysis_period_label}. Athlete totals are calculated first, then averaged across selected players represented that day."
        )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Analysis period", analysis_period_label)
    k2.metric("Selected players", len(selected_players))
    k3.metric("Position groups", len(selected_groups))
    k4.metric("Drills included", len(selected_drills) if view_mode == "drill" and selected_drills else "All")
    k5.metric("Analysis rows", rows)
    st.caption(copy)
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

    if table_data:
        table_df = pd.DataFrame(table_data)
        rename = {item["id"]: item["name"] for item in columns if "id" in item and "name" in item}
        display_dataframe(table_df.rename(columns=rename), height=520)
    else:
        st.caption("No table rows match the active analysis filters.")


def render_pdf_tab(raw, exclude_non_field):
    st.subheader("One-day PDF report")
    st.caption(
        "Choose one date, mark it High or Low, and select the players to include. "
        "Page 1 is the player load summary; page 2 is the position-group HSR and acceleration profile."
    )
    st.info(
        "High targets: Infielders HSR >80 m OR ≥10 accels; Outfielders HSR >100 m OR ≥10 accels; "
        "Catchers HSR >50 m. Low targets: Infielders HSR <30 m AND <8 accels; other target groups HSR <30 m."
    )

    filtered = filter_dashboard_rows(raw, ["exclude_non_field"] if exclude_non_field else [])
    dates = sorted(filtered["date"].dropna().astype(str).unique().tolist(), reverse=True)
    if not dates:
        st.warning("No report dates are available.")
        return

    c1, c2 = st.columns([1, 1])
    report_date = c1.selectbox("Report date", dates)
    day_type = c2.radio("Classify this day", ["High", "Low"], horizontal=True)

    day_rows = filtered[filtered["date"].astype(str).eq(report_date)]
    names = sorted({str(x).strip() for x in day_rows["player_name"].dropna().tolist() if str(x).strip()}, key=str.lower)
    selected_players = st.multiselect(
        "Players to include", names, default=names, key=f"report_players_{report_date}"
    )

    if st.button("Create one-day PDF report", type="primary"):
        if not selected_players:
            st.warning("Select at least one player.")
        else:
            try:
                use = filtered[filtered["player_name"].astype(str).isin(set(selected_players))].copy()
                day_type_rows = [{"date": report_date, "day_type": day_type}]
                summary = build_pdf_player_day_summary(
                    use, current_roster_rows(), day_type_rows, report_date, report_date
                )
                if summary.empty:
                    raise ValueError("No included FCL player workload rows were found for the selected report date.")
                path = create_single_day_pdf_target_report(summary, report_date, day_type)
                pdf_bytes = Path(path).read_bytes()
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
                st.session_state.pdf_bytes = pdf_bytes
                st.session_state.pdf_filename = f"FCL_Daily_HSR_Report_{report_date}_{day_type}.pdf"
                eligible = int(summary["target_status"].isin(["Hit", "Miss"]).sum())
                hits = int(summary["target_status"].eq("Hit").sum())
                st.session_state.pdf_status = (
                    f"Created {report_date} · {day_type} day · {len(summary)} selected players · "
                    f"{eligible} targetable · {hits} hits"
                )
            except Exception as exc:
                st.session_state.pdf_bytes = None
                st.session_state.pdf_filename = None
                st.session_state.pdf_status = f"PDF creation failed: {exc}"

    if st.session_state.pdf_status:
        if st.session_state.pdf_bytes:
            st.success(st.session_state.pdf_status)
            st.download_button(
                "Download PDF",
                data=st.session_state.pdf_bytes,
                file_name=st.session_state.pdf_filename or "FCL_Daily_HSR_Report.pdf",
                mime="application/pdf",
                use_container_width=False,
            )
        else:
            st.error(st.session_state.pdf_status)


def main():
    inject_css()
    init_state()
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">FCL • GPS LOAD</div>
          <div class="hero-title">Workload Dashboard</div>
          <div class="hero-copy">Day structure, player separation, position-group workload review, and one-day target reporting.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not API_KEY:
        st.warning("STATSPORTS_API_KEY is not configured. The app UI will load, but data pulls will not work until you add it to Streamlit Secrets.")
    if not GOOGLE_SHEET_ID:
        st.warning("GOOGLE_SHEET_ID is not configured. Data pulls still work, but Google Sheet sync/shared roster persistence will be unavailable.")

    setup_pull_section()
    raw = st.session_state.raw_data
    if raw is None or raw.empty:
        st.info("Pull FCL data above to populate the dashboard.")
        return

    with st.container(border=True):
        exclude_non_field = st.toggle(
            "Exclude Entire Session, lift, cages, ISD, activation, dynamic, and warm-up blocks",
            value=True,
            help="This filter applies to the practice view, load analysis, and PDF report.",
        )

    practice_tab, analysis_tab, roster_tab, pdf_tab = st.tabs([
        "Practice", "Load analysis", "Roster tagging", "PDF report"
    ])
    with practice_tab:
        render_practice_tab(raw, exclude_non_field)
    with analysis_tab:
        render_load_analysis_tab(raw, exclude_non_field)
    with roster_tab:
        render_roster_tab(raw)
    with pdf_tab:
        render_pdf_tab(raw, exclude_non_field)


if __name__ == "__main__":
    main()
