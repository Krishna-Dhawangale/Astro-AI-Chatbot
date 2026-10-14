# ============================================================
# STAGE 8.20 — DASHA TIMING ENGINE
# ============================================================
#
# Purpose:
#   Convert existing Dasha hierarchy + domain Dasha evidence
#   into structured timing windows.
#
# IMPORTANT:
#   - Does NOT recalculate Vimshottari Dasha
#   - Does NOT calculate planetary positions
#   - Does NOT assign probabilities
#   - Does NOT make final predictions
#   - Uses already validated Dasha dates
# ============================================================


# ------------------------------------------------------------
# 1. NORMALIZE DASHA TIMING RECORD
# ------------------------------------------------------------

def build_dasha_timing_record(
    dasha_level,
    planet,
    start,
    end,
    domain,
    source
):
    """
    Create one normalized Dasha timing record.
    """

    return {
        "domain": domain,
        "dasha_level": dasha_level,
        "planet": planet,
        "start": start,
        "end": end,
        "source": source
    }


# ------------------------------------------------------------
# 2. EXTRACT CURRENT DASHA WINDOWS
# ------------------------------------------------------------

def extract_current_dasha_windows(
    current_dasha_hierarchy
):
    """
    Extract Mahadasha, Antardasha and
    Pratyantardasha timing windows.
    """

    if not isinstance(current_dasha_hierarchy, dict):
        raise TypeError(
            "current_dasha_hierarchy must be a dictionary."
        )

    current_dasha_windows = []

    for level in [
        "mahadasha",
        "antardasha",
        "pratyantardasha"
    ]:

        data = current_dasha_hierarchy.get(level)

        if not data:
            continue

        planet = data.get("planet")

        if not isinstance(planet, str):
            continue

        start = data.get("start")
        end = data.get("end")

        if start is None or end is None:
            continue

        current_dasha_windows.append({
            "dasha_level": level,
            "planet": planet,
            "start": start,
            "end": end
        })

    return current_dasha_windows


# ------------------------------------------------------------
# 3. DOMAIN DASHA TIMING
# ------------------------------------------------------------

def build_domain_dasha_timing(
    domain,
    domain_data,
    current_dasha_windows
):
    """
    Connect domain Dasha evidence to the
    corresponding current Dasha timing window.
    """

    if not isinstance(domain_data, dict):
        return []

    timing_records = []

    dasha_connections = domain_data.get(
        "dasha_connections",
        {}
    )

    if not isinstance(dasha_connections, dict):
        return []

    all_connections = (
        dasha_connections.get("primary", [])
        +
        dasha_connections.get("supporting", [])
    )

    for connection in all_connections:

        dasha_level = connection.get(
            "dasha_level"
        )

        planet = connection.get(
            "planet"
        )

        if not isinstance(dasha_level, str):
            continue

        if not isinstance(planet, str):
            continue

        # ----------------------------------------------------
        # Find matching active Dasha window
        # ----------------------------------------------------

        matching_window = None

        for window in current_dasha_windows:

            if (
                window["dasha_level"] == dasha_level
                and
                window["planet"] == planet
            ):
                matching_window = window
                break

        if matching_window is None:
            continue

        timing_records.append({

            "domain": domain,

            "dasha_level":
                dasha_level,

            "planet":
                planet,

            "start":
                matching_window["start"],

            "end":
                matching_window["end"],

            "natal_house":
                connection.get("natal_house"),

            "house_type":
                connection.get("house_type"),

            "timing_source":
                "domain_dasha_connection"
        })

    return timing_records


# ------------------------------------------------------------
# 4. BUILD TIMING FOR ALL DOMAINS
# ------------------------------------------------------------

def build_dasha_timing_by_domain(
    domain_evidence,
    current_dasha_windows
):
    """
    Build Dasha timing records for every
    available domain.
    """

    if not isinstance(domain_evidence, dict):
        raise TypeError(
            "domain_evidence must be a dictionary."
        )

    dasha_timing_by_domain = {}

    for domain, domain_data in domain_evidence.items():

        dasha_timing_by_domain[domain] = (
            build_domain_dasha_timing(
                domain=domain,
                domain_data=domain_data,
                current_dasha_windows=
                    current_dasha_windows
            )
        )

    return dasha_timing_by_domain


# ------------------------------------------------------------
# 5. BUILD ACTIVE DASHA TIMELINE
# ------------------------------------------------------------

def build_active_dasha_timeline(
    current_dasha_windows
):
    """
    Create a simplified active Dasha timeline.
    """

    active_dasha_timeline = []

    for window in current_dasha_windows:

        active_dasha_timeline.append({

            "dasha_level":
                window["dasha_level"],

            "planet":
                window["planet"],

            "start":
                window["start"],

            "end":
                window["end"]
        })

    return active_dasha_timeline


# ------------------------------------------------------------
# 6. BUILD DOMAIN TIMING SIGNALS
# ------------------------------------------------------------

def build_dasha_timing_signals(
    dasha_timing_by_domain
):
    """
    Convert domain timing records into
    structured activation signals.
    """

    dasha_timing_signals = {}

    for domain, records in (
        dasha_timing_by_domain.items()
    ):

        signals = []

        for record in records:

            signals.append({

                "domain":
                    domain,

                "signal_type":
                    "dasha_domain_activation",

                "dasha_level":
                    record["dasha_level"],

                "planet":
                    record["planet"],

                "start":
                    record["start"],

                "end":
                    record["end"],

                "natal_house":
                    record["natal_house"],

                "house_type":
                    record["house_type"]
            })

        dasha_timing_signals[domain] = signals

    return dasha_timing_signals


# ------------------------------------------------------------
# 7. UNIFIED STAGE 8.20 OBJECT
# ------------------------------------------------------------

def build_dasha_timing_analysis(
    current_dasha_hierarchy,
    domain_evidence
):
    """
    Main Stage 8.20 Dasha Timing Engine.

    Inputs:
        current_dasha_hierarchy
        domain_evidence

    Returns:
        Structured Dasha timing analysis.
    """

    # --------------------------------------------------------
    # Extract active Dasha windows
    # --------------------------------------------------------

    current_dasha_windows = (
        extract_current_dasha_windows(
            current_dasha_hierarchy
        )
    )

    # --------------------------------------------------------
    # Build domain timing
    # --------------------------------------------------------

    dasha_timing_by_domain = (
        build_dasha_timing_by_domain(
            domain_evidence=domain_evidence,
            current_dasha_windows=
                current_dasha_windows
        )
    )

    # --------------------------------------------------------
    # Build active timeline
    # --------------------------------------------------------

    active_dasha_timeline = (
        build_active_dasha_timeline(
            current_dasha_windows
        )
    )

    # --------------------------------------------------------
    # Build timing signals
    # --------------------------------------------------------

    dasha_timing_signals = (
        build_dasha_timing_signals(
            dasha_timing_by_domain
        )
    )

    # --------------------------------------------------------
    # Unified object
    # --------------------------------------------------------

    stage_8_20_dasha_timing = {

        "current_dasha_windows":
            current_dasha_windows,

        "active_dasha_timeline":
            active_dasha_timeline,

        "domain_timing":
            dasha_timing_by_domain,

        "timing_signals":
            dasha_timing_signals
    }

    return stage_8_20_dasha_timing


# ------------------------------------------------------------
# 8. VALIDATION
# ------------------------------------------------------------

def validate_dasha_timing_analysis(
    stage_8_20_dasha_timing,
    expected_domains=None
):
    """
    Validate Stage 8.20 output.

    Returns True if validation passes.
    """

    if expected_domains is None:

        expected_domains = {
            "career",
            "marriage",
            "finance",
            "education",
            "property"
        }

    assert isinstance(
        stage_8_20_dasha_timing,
        dict
    )

    # --------------------------------------------------------
    # Required top-level keys
    # --------------------------------------------------------

    required_keys = [
        "current_dasha_windows",
        "active_dasha_timeline",
        "domain_timing",
        "timing_signals"
    ]

    for key in required_keys:

        assert key in stage_8_20_dasha_timing, (
            f"Missing Stage 8.20 key: {key}"
        )

    # --------------------------------------------------------
    # Normally there are 3 active Dasha levels
    # --------------------------------------------------------

    current_dasha_windows = (
        stage_8_20_dasha_timing[
            "current_dasha_windows"
        ]
    )

    assert len(current_dasha_windows) == 3, (
        "Expected 3 active Dasha levels: "
        "Mahadasha, Antardasha and Pratyantardasha."
    )

    for window in current_dasha_windows:

        assert isinstance(
            window["dasha_level"],
            str
        )

        assert isinstance(
            window["planet"],
            str
        )

        assert window["start"] is not None

        assert window["end"] is not None

    # --------------------------------------------------------
    # Domain validation
    # --------------------------------------------------------

    dasha_timing_by_domain = (
        stage_8_20_dasha_timing[
            "domain_timing"
        ]
    )

    assert set(
        dasha_timing_by_domain.keys()
    ) >= expected_domains

    for domain, records in (
        dasha_timing_by_domain.items()
    ):

        assert isinstance(
            records,
            list
        )

        for record in records:

            assert record["domain"] == domain

            assert isinstance(
                record["dasha_level"],
                str
            )

            assert isinstance(
                record["planet"],
                str
            )

            assert record["start"] is not None

            assert record["end"] is not None

    return True


# ------------------------------------------------------------
# 9. CONVENIENCE FUNCTION
# ------------------------------------------------------------

def run_dasha_timing_engine(
    current_dasha_hierarchy,
    domain_evidence,
    expected_domains=None
):
    """
    Build and validate the complete Stage 8.20
    Dasha Timing Engine output.
    """

    result = build_dasha_timing_analysis(
        current_dasha_hierarchy=
            current_dasha_hierarchy,

        domain_evidence=
            domain_evidence
    )

    validate_dasha_timing_analysis(
        stage_8_20_dasha_timing=result,
        expected_domains=expected_domains
    )

    return result


def build_dasha_timing_evidence(current_dasha_hierarchy, active_dasha_placements=None):
    if isinstance(current_dasha_hierarchy, dict) and "mahadasha" in current_dasha_hierarchy:
        active_planets = []
        if isinstance(active_dasha_placements, list):
            active_planets = [p.get("planet") for p in active_dasha_placements if isinstance(p, dict) and "planet" in p]
        elif isinstance(current_dasha_hierarchy, dict):
            for level in ("mahadasha", "antardasha", "pratyantardasha"):
                p = current_dasha_hierarchy.get(level, {}).get("planet")
                if p and p not in active_planets:
                    active_planets.append(p)
        return {
            "mahadasha": current_dasha_hierarchy.get("mahadasha", {}),
            "antardasha": current_dasha_hierarchy.get("antardasha", {}),
            "pratyantardasha": current_dasha_hierarchy.get("pratyantardasha", {}),
            "active_planets": active_planets,
            "placements": active_dasha_placements or []
        }
    return run_dasha_timing_engine(current_dasha_hierarchy, active_dasha_placements)
