# ============================================================
# STAGE 8.21 — TRANSIT TIMING ENGINE
# ============================================================


def build_domain_transit_timing(
    domain,
    domain_data,
    transit_analysis
):
    """
    Convert domain transit evidence into normalized
    timing records.
    """

    records = []

    transit_connections = domain_data.get(
        "transit_connections",
        {}
    )

    primary = transit_connections.get(
        "primary",
        []
    )

    supporting = transit_connections.get(
        "supporting",
        []
    )

    for connection in primary:

        records.append({
            "domain": domain,
            "connection_type": "primary",
            "transit_planet":
                connection.get("transit_planet"),
            "transit_house":
                connection.get("transit_house"),
            "transit_rashi":
                connection.get("transit_rashi"),
            "transit_nakshatra":
                connection.get("transit_nakshatra"),
            "transit_pada":
                connection.get("transit_pada")
        })

    for connection in supporting:

        records.append({
            "domain": domain,
            "connection_type": "supporting",
            "transit_planet":
                connection.get("transit_planet"),
            "transit_house":
                connection.get("transit_house"),
            "transit_rashi":
                connection.get("transit_rashi"),
            "transit_nakshatra":
                connection.get("transit_nakshatra"),
            "transit_pada":
                connection.get("transit_pada")
        })

    return records


def build_transit_timing_analysis(
    transit_analysis,
    domain_evidence,
    stage_8_20_dasha_timing
):
    # --------------------------------------------------------
    # DOMAIN TRANSIT TIMING
    # --------------------------------------------------------

    transit_timing_by_domain = {}

    for domain, domain_data in domain_evidence.items():

        transit_timing_by_domain[domain] = (
            build_domain_transit_timing(
                domain=domain,
                domain_data=domain_data,
                transit_analysis=transit_analysis
            )
        )

    # --------------------------------------------------------
    # TRANSIT PLANET SNAPSHOT
    # --------------------------------------------------------

    transit_planet_snapshot = {}
    planets_dict = transit_analysis.get("transit_planets") or transit_analysis.get("planets", {})

    for planet, data in planets_dict.items():

        transit_planet_snapshot[planet] = {

            "longitude":
                data.get("longitude"),

            "rashi":
                data.get("rashi"),

            "rashi_degree":
                data.get("rashi_degree"),

            "nakshatra":
                data.get("nakshatra"),

            "pada":
                data.get("pada"),

            "transit_house":
                data.get("transit_house"),

            "speed":
                data.get("speed")
        }

    # --------------------------------------------------------
    # DIRECT TRANSIT → NATAL RELATIONSHIPS
    # --------------------------------------------------------

    direct_relationships = (
        transit_analysis.get(
            "direct_natal_planet_aspects",
            []
        )
    )

    # --------------------------------------------------------
    # DOMAIN-SPECIFIC TRANSIT RELATIONSHIPS
    # --------------------------------------------------------

    domain_transit_relationships = {}

    for domain in domain_evidence:

        relevant_records = []

        domain_transit_planets = {
            record.get("transit_planet")
            for record
            in transit_timing_by_domain[domain]
            if record.get("transit_planet")
        }

        for relationship in direct_relationships:

            transit_planet = relationship.get(
                "transit_planet"
            )

            if transit_planet in domain_transit_planets:

                relevant_records.append({
                    "transit_planet":
                        transit_planet,

                    "natal_planet":
                        relationship.get(
                            "natal_planet"
                        ),

                    "aspect":
                        relationship.get(
                            "aspect"
                        ),

                    "aspect_distance":
                        relationship.get(
                            "aspect_distance"
                        ),

                    "transit_house":
                        relationship.get(
                            "transit_house"
                        ),

                    "natal_house":
                        relationship.get(
                            "natal_house"
                        )
                })

        domain_transit_relationships[domain] = (
            relevant_records
        )

    # --------------------------------------------------------
    # DASHA + TRANSIT TIMING
    # --------------------------------------------------------

    dasha_timing_by_domain = (
        stage_8_20_dasha_timing[
            "domain_timing"
        ]
    )

    dasha_transit_timing = {}

    for domain in domain_evidence:

        dasha_records = dasha_timing_by_domain.get(
            domain,
            []
        )

        transit_records = transit_timing_by_domain.get(
            domain,
            []
        )

        combined_records = []

        for dasha_record in dasha_records:

            dasha_planet = dasha_record.get(
                "planet"
            )

            dasha_level = dasha_record.get(
                "dasha_level"
            )

            for transit_record in transit_records:

                transit_planet = transit_record.get(
                    "transit_planet"
                )

                combined_records.append({

                    "domain":
                        domain,

                    "dasha_level":
                        dasha_level,

                    "dasha_planet":
                        dasha_planet,

                    "dasha_start":
                        dasha_record.get(
                            "start"
                        ),

                    "dasha_end":
                        dasha_record.get(
                            "end"
                        ),

                    "transit_planet":
                        transit_planet,

                    "transit_house":
                        transit_record.get(
                            "transit_house"
                        ),

                    "connection_type":
                        transit_record.get(
                            "connection_type"
                        ),

                    "transit_rashi":
                        transit_record.get(
                            "transit_rashi"
                        )
                })

        dasha_transit_timing[domain] = (
            combined_records
        )

    # --------------------------------------------------------
    # TIMING SIGNALS
    # --------------------------------------------------------

    transit_timing_signals = {}

    for domain in domain_evidence:

        signals = []

        # Transit house activation
        for record in transit_timing_by_domain[domain]:

            signals.append({
                "signal_type":
                    "transit_house_activation",

                "domain":
                    domain,

                "connection_type":
                    record["connection_type"],

                "transit_planet":
                    record["transit_planet"],

                "transit_house":
                    record["transit_house"],

                "transit_rashi":
                    record["transit_rashi"]
            })

        # Transit → natal relationship
        for relationship in (
            domain_transit_relationships[domain]
        ):

            signals.append({
                "signal_type":
                    "transit_natal_relationship",

                "domain":
                    domain,

                **relationship
            })

        # Dasha + transit combination
        for record in dasha_transit_timing[domain]:

            signals.append({
                "signal_type":
                    "dasha_transit_overlap",

                **record
            })

        transit_timing_signals[domain] = signals

    # --------------------------------------------------------
    # UNIFIED STAGE 8.21 OBJECT
    # --------------------------------------------------------

    return {
        "transit_datetime":
            transit_analysis.get(
                "datetime_utc"
            ),

        "transit_planets":
            transit_planet_snapshot,

        "domain_transit_timing":
            transit_timing_by_domain,

        "domain_transit_relationships":
            domain_transit_relationships,

        "dasha_transit_timing":
            dasha_transit_timing,

        "timing_signals":
            transit_timing_signals
    }


def validate_transit_timing_analysis(
    stage_8_21_transit_timing,
    expected_domains=None
):
    if expected_domains is None:
        expected_domains = {
            "career",
            "marriage",
            "finance",
            "education",
            "property"
        }

    assert set(
        stage_8_21_transit_timing[
            "domain_transit_timing"
        ].keys()
    ) >= expected_domains

    assert set(
        stage_8_21_transit_timing[
            "domain_transit_relationships"
        ].keys()
    ) >= expected_domains

    assert set(
        stage_8_21_transit_timing[
            "dasha_transit_timing"
        ].keys()
    ) >= expected_domains

    assert set(
        stage_8_21_transit_timing[
            "timing_signals"
        ].keys()
    ) >= expected_domains

    transit_planet_snapshot = (
        stage_8_21_transit_timing[
            "transit_planets"
        ]
    )

    assert len(transit_planet_snapshot) > 0

    for planet, data in transit_planet_snapshot.items():

        assert isinstance(
            planet,
            str
        )

        assert data["longitude"] is not None
        assert data["rashi"] is not None
        assert data["transit_house"] is not None

    for domain, records in (
        stage_8_21_transit_timing[
            "domain_transit_timing"
        ].items()
    ):

        assert isinstance(
            records,
            list
        )

        for record in records:

            assert record["domain"] == domain

            assert record[
                "transit_planet"
            ] is not None

    for domain, signals in (
        stage_8_21_transit_timing[
            "timing_signals"
        ].items()
    ):

        assert isinstance(
            signals,
            list
        )

        for signal in signals:

            assert "signal_type" in signal

            assert signal["domain"] == domain

    return True


def build_transit_timing_evidence(transit_result, transit_aspects=None):
    if isinstance(transit_result, dict) and "planets" in transit_result:
        return {
            "planets": transit_result.get("planets", {}),
            "aspects": transit_aspects or [],
            "natal_aspects": transit_aspects or []
        }
    return build_transit_timing_analysis(transit_result, transit_aspects)
