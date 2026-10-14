"""Aircraft type, wake category, and who is operating the flight.

``AIRCRAFT_OPERATOR_flt`` is anonymised (a SHA-256 hash), but the operator is
still recoverable: ``CALLSIGN_flt`` carries the real ICAO airline prefix
("DLH400"), and ``FLIGHT_mvt`` carries a prefix too. The latter is the more
valuable of the two here -- it comes from the movement record, so it survives
the ~1.5% of rows where the NM flight join failed and every ``_flt`` column is
null at once (0.75% missing versus 1.33%).

Note that the ``FLIGHT_mvt`` prefix is *not* uniformly IATA: the delivered data
mixes IATA ("LH", "KL") and ICAO ("THY", "AFR", "BAW") codes depending on the
reporting airport, so it is kept as `carrier_flightno` rather than being
presented as something it is not. As a categorical that is harmless -- the two
encodings simply never collide -- but it means the column should not be joined
against an external airline table without normalising first.
"""

from __future__ import annotations

import pandas as pd

from prc.features.registry import FeatureContext, feature


def _cat(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("__NA__").astype("category")


@feature(
    group="aircraft",
    outputs=(
        "aircraft_type",
        "aircraft_type_flt",
        "aircraft_type_mismatch",
        "wake_category",
        "market_segment",
        "flight_type",
        "flight_rule_mvt",
        "flight_rule_flt",
        "operator_hash",
        "carrier_icao",
        "carrier_flightno",
        "carrier_missing",
    ),
    depends_on=("core_keys",),
    categorical=(
        "aircraft_type",
        "aircraft_type_flt",
        "wake_category",
        "market_segment",
        "flight_type",
        "flight_rule_mvt",
        "flight_rule_flt",
        "operator_hash",
        "carrier_icao",
        "carrier_flightno",
    ),
    doc="Aircraft type / wake category / operator identity.",
)
def aircraft_identity(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    dep = mvt.loc[ctx.dep_index]

    type_mvt = dep["AIRCRAFT_TYPE_mvt"].astype("string")
    type_flt = dep["AIRCRAFT_TYPE_flt"].astype("string")

    callsign = dep["CALLSIGN_flt"].astype("string")
    flight_no = dep["FLIGHT_mvt"].astype("string")
    carrier_icao = callsign.str.extract(r"^([A-Z]{3})", expand=False)
    carrier_flightno = flight_no.str.extract(r"^([A-Z0-9]{2,3}?)(?=\d)", expand=False)

    return pd.DataFrame(
        {
            "aircraft_type": _cat(type_mvt),
            "aircraft_type_flt": _cat(type_flt),
            "aircraft_type_mismatch": (type_mvt.notna() & type_flt.notna() & type_mvt.ne(type_flt)).astype("int8"),
            "wake_category": _cat(dep["WK_TBL_CAT_flt"]),
            "market_segment": _cat(dep["MARKET_SEGMENT_flt"]),
            "flight_type": _cat(dep["FLIGHT_TYPE_flt"]),
            "flight_rule_mvt": _cat(dep["FLIGHT_RULE_mvt"]),
            "flight_rule_flt": _cat(dep["FLIGHT_RULE_flt"]),
            "operator_hash": _cat(dep["AIRCRAFT_OPERATOR_flt"]),
            "carrier_icao": _cat(carrier_icao),
            "carrier_flightno": _cat(carrier_flightno),
            "carrier_missing": carrier_flightno.isna().astype("int8"),
        },
        index=ctx.dep_index,
    )


@feature(
    group="aircraft",
    outputs=("carrier_apt_freq", "carrier_apt_share", "type_apt_freq", "carrier_stand_concentration"),
    depends_on=("core_keys", "aircraft_identity", "stand_runway_keys"),
    doc="How established this carrier / aircraft type is at this airport.",
)
def operator_presence(mvt: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Home-carrier effects.

    A based carrier uses familiar stands and predictable routings; a rare
    visitor gets whatever is free. Counts only -- no target is touched.
    """
    ident = ctx.dep("aircraft_identity")
    keys = ctx.dep("stand_runway_keys")
    apt = mvt.loc[ctx.dep_index, "apt"].astype("string")

    carrier_apt = (apt + "_" + ident["carrier_flightno"].astype("string")).astype("string")
    type_apt = (apt + "_" + ident["aircraft_type"].astype("string")).astype("string")

    carrier_n = carrier_apt.map(carrier_apt.value_counts()).astype("float32")
    apt_n = apt.map(apt.value_counts()).astype("float32")
    type_n = type_apt.map(type_apt.value_counts()).astype("float32")

    # How concentrated this carrier's stand usage is at this airport: a based
    # carrier with a dedicated pier behaves differently from a scattered one.
    pair = (carrier_apt + "_" + keys["stand_prefix"].astype("string")).astype("string")
    pair_n = pair.map(pair.value_counts()).astype("float32")

    return pd.DataFrame(
        {
            "carrier_apt_freq": carrier_n,
            "carrier_apt_share": (carrier_n / apt_n).astype("float32"),
            "type_apt_freq": type_n,
            "carrier_stand_concentration": (pair_n / carrier_n).astype("float32"),
        },
        index=ctx.dep_index,
    )
