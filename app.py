import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import difflib

st.set_page_config(page_title="Farm Stress Dashboard", layout="wide")

# ============================================================
# 1. CANONICAL 60-ITEM FARM STRESS INVENTORY
# ============================================================
CANONICAL_ITEMS = [
    "Sudden and significant increase in debt load",
    "Significant production loss due to pests and disease",
    "Insufficient regular cash flow to meet financial obligations or for daily necessities",
    "Delay planting or harvesting due to weather",
    "Media distortion of farm situation",
    "Low commodity prices",
    "Significant changes in type or farming operations",
    "Meeting with loan officer for additional loan",
    "Purchase of major machinery or facility and livestock",
    "Not being considered part of the business by others",
    "Taking an off the farm job to meet expenses",
    "Prolonged bad weather",
    "Problems with weeds and insects",
    "Machinery breakdown at critical time",
    "Not enough time to spend with spouse",
    "Crop loss due to weather",
    "Illness during planting and harvesting",
    "Deciding when to sell",
    "Rising expenses",
    "Government policies and regulations",
    "Concerns about continued financial viability of the farm",
    "Having a loan request turned down",
    "Farming related accident",
    "Government free trade policies",
    "Government cheap food policies",
    "Metric conversion",
    "Breeding or reproductive difficulties with livestock",
    "No farm help or loss of help when needed",
    "Need to learn/adjust to new government policies and regulations",
    "Spouse does not have enough time for child rearing",
    "Increase workload at peak time",
    "Unplanned interruptions",
    "Use of hazardous chemicals in farm",
    "Dealing with sales people",
    "Long work hours",
    "Few vacations away from the farm",
    "Feeling isolated on the farm",
    "Having to travel long distances for services, shopping and health care",
    "Pressure of having too much to do in too little time period",
    "Keeping up with new technologies and products",
    "Worrying about market conditions",
    "Having to make decisions without all necessary information",
    "Being expected to work on the farm as well as manage the house",
    "Worrying about owing money",
    "Worrying about keeping the farm in the family",
    "Not seeing enough people",
    "Not having enough money or time for entertainment and recreation",
    "Death of a parent or member of immediate family",
    "Death of a friend",
    "Problem balancing work and family responsibilities",
    "Problem with relatives in farm operating agreement",
    "Divorce or marital separation",
    "Problems with partnership",
    "Daughter or son leaving home",
    "Trouble with parents or in laws",
    "Conflict with spouse over spending priorities",
    "Major decisions being made without any knowledge or input",
    "Having to wear too many hats eg farmer, spouse, mother",
    "Surface right negotiation",
    "Other recent events which had an impact on life",
]
assert len(CANONICAL_ITEMS) == 60, "Canonical item list must have exactly 60 items"

# ============================================================
# 2. 8 THEMATIC CATEGORIES (paper sections 4.2 - 4.9)
# ============================================================
CATEGORIES = {
    "Financial & Economic": [
        "Sudden and significant increase in debt load",
        "Insufficient regular cash flow to meet financial obligations or for daily necessities",
        "Low commodity prices",
        "Meeting with loan officer for additional loan",
        "Purchase of major machinery or facility and livestock",
        "Taking an off the farm job to meet expenses",
        "Rising expenses",
        "Concerns about continued financial viability of the farm",
        "Having a loan request turned down",
        "Government free trade policies",
        "Government cheap food policies",
        "Deciding when to sell",
        "Worrying about market conditions",
        "Worrying about owing money",
        "Worrying about keeping the farm in the family",
    ],
    "Crop Production & Environmental": [
        "Significant production loss due to pests and disease",
        "Delay planting or harvesting due to weather",
        "Prolonged bad weather",
        "Problems with weeds and insects",
        "Crop loss due to weather",
        "Breeding or reproductive difficulties with livestock",
        "Illness during planting and harvesting",
        "Use of hazardous chemicals in farm",
    ],
    "Workload & Time Management": [
        "Increase workload at peak time",
        "Unplanned interruptions",
        "Few vacations away from the farm",
        "Pressure of having too much to do in too little time period",
        "Being expected to work on the farm as well as manage the house",
        "Having to wear too many hats eg farmer, spouse, mother",
        "Long work hours",
    ],
    "Government, Technology & Information": [
        "Government policies and regulations",
        "Need to learn/adjust to new government policies and regulations",
        "Metric conversion",
        "Keeping up with new technologies and products",
        "Having to make decisions without all necessary information",
        "Media distortion of farm situation",
        "Dealing with sales people",
    ],
    "Family & Interpersonal": [
        "Not enough time to spend with spouse",
        "Spouse does not have enough time for child rearing",
        "Problem balancing work and family responsibilities",
        "Conflict with spouse over spending priorities",
        "Trouble with parents or in laws",
        "Daughter or son leaving home",
        "Major decisions being made without any knowledge or input",
        "Divorce or marital separation",
        "Death of a parent or member of immediate family",
        "Death of a friend",
    ],
    "Social Isolation & Community": [
        "Not being considered part of the business by others",
        "Feeling isolated on the farm",
        "Not seeing enough people",
        "Having to travel long distances for services, shopping and health care",
        "Not having enough money or time for entertainment and recreation",
    ],
    "Farm Management & Operational": [
        "Significant changes in type or farming operations",
        "No farm help or loss of help when needed",
        "Problems with partnership",
        "Problem with relatives in farm operating agreement",
        "Surface right negotiation",
        "Machinery breakdown at critical time",
        "Farming related accident",
    ],
    "Other Life Events": [
        "Other recent events which had an impact on life",
    ],
}

# ============================================================
# 3. 7 EFA-DERIVED FACTORS (Principal Axis Factoring, Varimax)
#    Items selected using loading >= 0.40 threshold from the
#    actual rotated factor matrix supplied by the researcher.
# ============================================================
FACTORS = {
    "Personal & Financial Concerns": [
        "Worrying about owing money",
        "Worrying about keeping the farm in the family",
        "Unplanned interruptions",
        "Not seeing enough people",
    ],
    "Role & Decision-Making Stress": [
        "Major decisions being made without any knowledge or input",
        "Having to wear too many hats eg farmer, spouse, mother",
        "Surface right negotiation",
        "Other recent events which had an impact on life",
    ],
    "Marital & Relationship Strain": [
        "Divorce or marital separation",
        "Problems with partnership",
    ],
    "Occupational & Financial Risks": [
        "Sudden and significant increase in debt load",
        "Use of hazardous chemicals in farm",
        "Dealing with sales people",
    ],
    "Work-Family & Seasonal Challenges": [
        "Not enough time to spend with spouse",
        "Crop loss due to weather",
        "Illness during planting and harvesting",
    ],
    "Farm Sustainability & Risk": [
        "Concerns about continued financial viability of the farm",
        "Having a loan request turned down",
        "Farming related accident",
    ],
    "Farm Management Challenges": [
        "Having to travel long distances for services, shopping and health care",
        "Pressure of having too much to do in too little time period",
        "Keeping up with new technologies and products",
    ],
}

# ============================================================
# 3b. REGRESSION-SIGNIFICANT FACTORS (from the full 20-factor
#    EFA solution). These are the 7 factors that came out as
#    statistically significant predictors of Overall Stress
#    Score in the researcher's multiple linear regression
#    (Model R Square = 0.990). Item groupings and standardized
#    beta weights are taken directly from her SPSS output --
#    this dashboard does not recompute the regression itself,
#    it applies her fixed weights to newly uploaded data.
#    NOTE: this is a different 7-factor grouping than FACTORS
#    above (the original paper's named EFA dimensions) -- some
#    factors overlap, others come from elsewhere in the 20.
# ============================================================
REGRESSION_FACTORS = {
    "Personal & Financial Concerns": [
        "Worrying about owing money",
        "Worrying about keeping the farm in the family",
        "Unplanned interruptions",
        "Not seeing enough people",
    ],
    "Role & Decision-Making Stress": [
        "Major decisions being made without any knowledge or input",
        "Having to wear too many hats eg farmer, spouse, mother",
        "Surface right negotiation",
        "Other recent events which had an impact on life",
    ],
    "Farm Management Challenges": [
        "Having to travel long distances for services, shopping and health care",
        "Pressure of having too much to do in too little time period",
        "Keeping up with new technologies and products",
    ],
    "Work-Family & Seasonal Challenges": [
        "Not enough time to spend with spouse",
        "Crop loss due to weather",
        "Illness during planting and harvesting",
    ],
    "Farming Demands & Environmental Challenges": [
        "Significant changes in type or farming operations",
        "Long work hours",
        "Problem balancing work and family responsibilities",
        "Prolonged bad weather",
    ],
    "Social Isolation & Leisure Deprivation": [
        "Feeling isolated on the farm",
        "Few vacations away from the farm",
    ],
    "Family Role & Economic Burden": [
        "Problem with relatives in farm operating agreement",
        "Being expected to work on the farm as well as manage the house",
    ],
}

# Standardized Beta coefficients from the researcher's multiple
# linear regression (DV: Overall Stress Score). Fixed reference
# values -- not recomputed per upload.
REGRESSION_BETAS = {
    "Personal & Financial Concerns": 0.251,
    "Role & Decision-Making Stress": 0.233,
    "Farm Management Challenges": 0.224,
    "Work-Family & Seasonal Challenges": 0.207,
    "Farming Demands & Environmental Challenges": 0.204,
    "Social Isolation & Leisure Deprivation": 0.223,
    "Family Role & Economic Burden": 0.216,
}

# ============================================================
# 4. SCALE MAPPING (text labels -> 1-5 numeric)
# ============================================================
SCALE_MAP = {
    "very low stress": 1,
    "low stress": 2,
    "medium stress": 3,
    "high stress": 4,
    "very high stress": 5,
}


def normalize_col(c):
    return " ".join(str(c).lower().strip().split())


def map_columns(df_columns):
    """Match uploaded column headers to the canonical 60 items.
    Pass 1: exact match after normalization (handles case/spacing only).
    Pass 2: high-confidence fuzzy match, restricted to columns of similar
    length, to fix typos without letting short/unrelated background
    columns (e.g. 'Farming Experience') collide with stress items
    (e.g. 'Rising expenses')."""
    norm_targets = {normalize_col(i): i for i in CANONICAL_ITEMS}
    mapping = {}
    used = set()

    # Pass 1: exact normalized match
    remaining_cols = []
    for col in df_columns:
        nc = normalize_col(col)
        if nc in norm_targets and norm_targets[nc] not in used:
            mapping[col] = norm_targets[nc]
            used.add(norm_targets[nc])
        else:
            remaining_cols.append(col)

    # Pass 2: high-confidence fuzzy match (typo tolerance only)
    for col in remaining_cols:
        nc = normalize_col(col)
        candidates = [t for t in norm_targets if norm_targets[t] not in used]
        # only compare against targets of similar length to avoid
        # short-string coincidental overlaps
        candidates = [t for t in candidates if abs(len(t) - len(nc)) <= 8]
        match = difflib.get_close_matches(nc, candidates, n=1, cutoff=0.85)
        if match:
            mapping[col] = norm_targets[match[0]]
            used.add(norm_targets[match[0]])
    return mapping


def to_numeric_score(series):
    """Convert a column of text labels or numbers into 1-5 numeric scores."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.lower().str.strip()
    return s.map(SCALE_MAP)


def classify_row(row):
    """Classify a farmer's overall stress level using the source
    inventory's own scoring guide: whichever score-band (1-2 / 3 / 4-5)
    contains the most of that farmer's answered items wins (plurality)."""
    vals = row.dropna()
    if len(vals) == 0:
        return None
    low = ((vals == 1) | (vals == 2)).sum()
    med = (vals == 3).sum()
    high = ((vals == 4) | (vals == 5)).sum()
    counts = {"Low": low, "Medium": med, "High": high}
    return max(counts, key=counts.get)


# Standard background/demographic columns expected alongside the 60
# stress items. New users can download a blank template with these
# headers pre-filled rather than needing to know the format by hand.
DEMOGRAPHIC_COLUMNS = [
    "Timestamp",
    "Name of respondent",
    "Village",
    "Mandal/Taluka",
    "District",
    "Age (in years)",
    "Marital Status",
    "Education Level",
    "Family Size",
    "Landholding Size",
    "Annual Family Income",
    "Main Source of Income",
    "Present Loan Status",
    "Source of Loan (if yes)",
    "Type of loan",
    "Type of Farming",
    "Role in Farming",
    "Farming Experience",
    "Daily Working Hours",
    "Care labour activities",
    "Self-rated Health",
    "Chronic Illness",
    "SHG / Community Participation",
    "Decision-making Power",
    "Major Stressors",
    "Identities",
]


def build_template_excel():
    """Build a blank Excel file (headers only, no data rows) containing
    the demographic columns plus all 60 canonical stress items, so a new
    user can download it, fill it in, and re-upload without needing to
    know the exact expected column format."""
    import io
    all_columns = DEMOGRAPHIC_COLUMNS + CANONICAL_ITEMS
    template_df = pd.DataFrame(columns=all_columns)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        template_df.to_excel(writer, index=False, sheet_name="Farmer Responses")
    buffer.seek(0)
    return buffer


# ============================================================
# APP LAYOUT
# ============================================================
st.markdown(
    "<p style='text-align:center; font-size:15px; margin-bottom:0;'>"
    "Vasantrao Naik Marathwada Krishi Vidyapeeth</p>"
    "<p style='text-align:center; font-size:14px; color:gray; margin-top:0;'>"
    "College of Community Science</p>",
    unsafe_allow_html=True,
)
st.title("Mental Stress Among Rural Female Farmers Dashboard")
st.caption(
    "Upload farmer survey responses (60-item Farm Stress Inventory) to see "
    "stress patterns across items, categories, and the empirically-derived "
    "stress factors influencing overall mental stress among rural female "
    "farmers."
)

with st.expander("New here? Download a blank template first"):
    st.write(
        "Not sure of the exact column format? Download this blank Excel "
        "file -- it has all the required column headers (background "
        "details + the 60 Farm Stress Inventory items) already in place "
        "with no data. Fill in one row per farmer and upload it below."
    )
    st.download_button(
        label="Download blank template (.xlsx)",
        data=build_template_excel(),
        file_name="farm_stress_inventory_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

uploaded = st.file_uploader(
    "Upload farmer response file (Excel or CSV)", type=["xlsx", "xls", "csv"]
)

if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.success(f"Loaded {len(df)} farmer responses with {len(df.columns)} columns.")

    # Drop fully-empty columns (e.g. trailing blank columns some exports include)
    non_empty_cols = [c for c in df.columns if df[c].notna().any()]
    df = df[non_empty_cols]

    col_map = map_columns(df.columns)
    matched_canonical = set(col_map.values())
    missing = [i for i in CANONICAL_ITEMS if i not in matched_canonical]

    with st.expander(
        f"Column matching details ({len(matched_canonical)}/60 items matched)"
    ):
        st.write(pd.DataFrame(
            {"Uploaded column": list(col_map.keys()),
             "Matched to inventory item": list(col_map.values())}
        ))
        if missing:
            st.warning(
                f"{len(missing)} inventory items were NOT found in the "
                f"uploaded file:\n\n" + "\n".join(f"- {m}" for m in missing)
            )

    if len(matched_canonical) < 60:
        st.error(
            "Not all 60 stress inventory items could be matched. Results "
            "below only use the matched items -- check column names if "
            "this looks wrong."
        )

    # Build numeric scored dataframe using canonical column names
    scored = pd.DataFrame(index=df.index)
    for orig_col, canon in col_map.items():
        scored[canon] = to_numeric_score(df[orig_col])

    st.divider()

    # -------------------------------------------------------
    # SECTION 1: Top stressors (item-level)
    # -------------------------------------------------------
    st.header("Top Stressors (Item-Level)")
    item_means = scored.mean().sort_values(ascending=False)
    item_stats = pd.DataFrame(
        {"Mean": item_means, "SD": scored.std()[item_means.index]}
    )
    top_n = st.slider("Show top N stressors", 5, 30, 10)
    st.bar_chart(item_means.head(top_n))
    st.dataframe(item_stats.head(top_n).style.format("{:.2f}"))

    st.divider()

    # -------------------------------------------------------
    # SECTION 2: Category breakdown
    # -------------------------------------------------------
    st.header("Stress by Category")
    cat_scores = {}
    for cat, items in CATEGORIES.items():
        valid_items = [i for i in items if i in scored.columns]
        if valid_items:
            cat_scores[cat] = scored[valid_items].mean(axis=1).mean()
    cat_df = pd.Series(cat_scores).sort_values(ascending=False)
    st.bar_chart(cat_df)

    st.divider()

    # -------------------------------------------------------
    # SECTION 3: 7-Factor scores
    # -------------------------------------------------------
    st.header("7-Factor Stress Profile (EFA-derived)")
    st.caption(
        "Based on Principal Axis Factoring with Varimax rotation from the "
        "source study. Each factor score is the average of its loading "
        "items (loading >= 0.40)."
    )
    factor_scores = {}
    for fac, items in FACTORS.items():
        valid_items = [i for i in items if i in scored.columns]
        if valid_items:
            factor_scores[fac] = scored[valid_items].mean(axis=1).mean()
    factor_df = pd.Series(factor_scores).sort_values(ascending=False)
    st.bar_chart(factor_df)
    st.dataframe(factor_df.rename("Mean Score").to_frame().style.format("{:.2f}"))

    st.divider()

    # -------------------------------------------------------
    # SECTION 3b: Relative Contribution to Overall Stress
    #   (using the researcher's regression beta weights)
    # -------------------------------------------------------
    st.header("Relative Contribution to Overall Stress")
    st.caption(
        "Based on the researcher's multiple linear regression "
        "(Overall Stress Score as dependent variable, 20 EFA-derived "
        "factors as predictors; Model R Square = 0.990). All 20 factors "
        "were statistically significant predictors (p < .001) -- these 7 "
        "are the ones with the highest standardized beta weights (strongest "
        "relative influence). Beta weights are fixed values from her "
        "analysis, applied here to this uploaded batch's factor scores -- "
        "they are not recomputed per upload."
    )
    st.info(
        "Note: because the overall score is itself built from the same "
        "60 items these factors are derived from, R Square this high "
        "reflects how much of the total score each factor's items "
        "account for, rather than an independent causal prediction. "
        "Treat this as a relative-weighting view, not a causal one."
    )

    reg_factor_scores = {}
    for fac, items in REGRESSION_FACTORS.items():
        valid_items = [i for i in items if i in scored.columns]
        if valid_items:
            reg_factor_scores[fac] = scored[valid_items].mean(axis=1).mean()

    if reg_factor_scores:
        weighted = {
            fac: reg_factor_scores[fac] * REGRESSION_BETAS[fac]
            for fac in reg_factor_scores
            if fac in REGRESSION_BETAS
        }
        total_weighted = sum(weighted.values())
        contribution_pct = (
            {fac: (v / total_weighted * 100) for fac, v in weighted.items()}
            if total_weighted > 0
            else {}
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("This batch's factor scores")
            st.bar_chart(pd.Series(reg_factor_scores).sort_values(ascending=False))
        with col2:
            st.subheader("Fixed beta weights (from thesis)")
            st.bar_chart(pd.Series(REGRESSION_BETAS).sort_values(ascending=False))

        st.subheader("Estimated relative contribution to overall stress (%)")
        contribution_series = pd.Series(contribution_pct).sort_values(ascending=False)
        st.bar_chart(contribution_series)
        st.dataframe(
            pd.DataFrame({
                "Mean Score (this batch)": pd.Series(reg_factor_scores),
                "Beta Weight (fixed)": pd.Series(REGRESSION_BETAS),
                "Relative Contribution (%)": contribution_series,
            }).sort_values("Relative Contribution (%)", ascending=False)
            .style.format({
                "Mean Score (this batch)": "{:.2f}",
                "Beta Weight (fixed)": "{:.3f}",
                "Relative Contribution (%)": "{:.1f}",
            })
        )
    else:
        st.info("None of the 7 regression-significant factors' items were found in this upload.")

    st.divider()

    # -------------------------------------------------------
    # SECTION 4: Overall stress classification
    # -------------------------------------------------------
    st.header("Overall Stress Classification")
    st.caption(
        "Based on the source inventory's own scoring guide: each farmer is "
        "classified by which score-band (1-2 / 3 / 4-5) contains the most "
        "of their answered items."
    )

    scored_valid = scored.dropna(how="all")
    classifications = scored_valid.apply(classify_row, axis=1).dropna()
    class_counts = classifications.value_counts()
    class_pct = (class_counts / class_counts.sum() * 100).round(1)

    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots()
        ax.pie(
            class_pct,
            labels=[f"{k} ({v}%)" for k, v in class_pct.items()],
            autopct=None,
        )
        ax.set_title("Overall Stress Distribution")
        st.pyplot(fig)
    with col2:
        st.dataframe(class_pct.rename("Percentage").to_frame())

    st.divider()

    demo_cols = [c for c in df.columns if c not in col_map]

    # -------------------------------------------------------
    # SECTION 5: Stress vs Background Factors (comparison)
    # -------------------------------------------------------
    st.header("Stress vs Background Factors")
    st.caption(
        "Compare average stress levels across different farmer background "
        "groups -- e.g. does landholding size, education level, or village "
        "relate to higher stress?"
    )

    # Per-farmer row-level stress measures to compare against
    row_overall = scored.mean(axis=1)

    row_cat_scores = {}
    for cat, items in CATEGORIES.items():
        valid_items = [i for i in items if i in scored.columns]
        if valid_items:
            row_cat_scores[f"Category: {cat}"] = scored[valid_items].mean(axis=1)

    row_factor_scores = {}
    for fac, items in FACTORS.items():
        valid_items = [i for i in items if i in scored.columns]
        if valid_items:
            row_factor_scores[f"Factor: {fac}"] = scored[valid_items].mean(axis=1)

    measure_options = {"Overall Stress Score (all items)": row_overall}
    measure_options.update(row_cat_scores)
    measure_options.update(row_factor_scores)

    # Candidate background columns: low-cardinality only, so grouping
    # produces a meaningful comparison rather than one bar per farmer
    compare_candidates = [
        c for c in demo_cols
        if 1 < df[c].nunique(dropna=True) < 20
    ]

    if compare_candidates:
        col1, col2 = st.columns(2)
        with col1:
            chosen_demo = st.selectbox("Background factor", compare_candidates)
        with col2:
            chosen_measure_label = st.selectbox(
                "Stress measure", list(measure_options.keys())
            )
        chosen_measure = measure_options[chosen_measure_label]

        compare_df = pd.DataFrame(
            {chosen_demo: df[chosen_demo], "Stress": chosen_measure}
        ).dropna()

        grouped = (
            compare_df.groupby(chosen_demo)["Stress"]
            .agg(["mean", "std", "count"])
            .sort_values("mean", ascending=False)
        )

        st.bar_chart(grouped["mean"])
        st.dataframe(
            grouped.rename(
                columns={"mean": "Mean Stress", "std": "SD", "count": "N"}
            ).style.format({"Mean Stress": "{:.2f}", "SD": "{:.2f}"})
        )

        with st.expander(
            "View as stress-level distribution (% High/Medium/Low) instead"
        ):
            dist_df = pd.DataFrame(
                {chosen_demo: df[chosen_demo], "Level": classifications}
            ).dropna()
            if len(dist_df) > 0:
                pct_table = (
                    pd.crosstab(dist_df[chosen_demo], dist_df["Level"], normalize="index")
                    * 100
                )
                st.bar_chart(pct_table)
                st.dataframe(pct_table.style.format("{:.1f}"))
            else:
                st.info("No overlapping data available for this comparison.")
    else:
        st.info(
            "No background columns with a comparable number of categories "
            "(2-19 unique values) were found to compare against."
        )

    st.divider()

    # -------------------------------------------------------
    # SECTION 6: Demographics (optional, best-effort)
    # -------------------------------------------------------
    with st.expander("Demographic / Background Summary"):
        shown = 0
        skipped_high_cardinality = []
        for c in demo_cols:
            if shown >= 15:
                break
            n_unique = df[c].nunique(dropna=True)
            # Show any low-cardinality column regardless of whether it's
            # text labels ("Married") or numeric codes (1, 2, 3...) --
            # both formats appear across different export styles.
            if 1 < n_unique < 20:
                st.write(f"**{c}**")
                st.bar_chart(df[c].value_counts().sort_index())
                shown += 1
            elif n_unique >= 20:
                skipped_high_cardinality.append(c)

        if shown == 0:
            st.info(
                "No demographic columns with a chartable number of "
                "categories (2-19 unique values) were found."
            )
        if skipped_high_cardinality:
            st.caption(
                "Columns skipped (too many unique values to chart): "
                + ", ".join(skipped_high_cardinality)
            )

else:
    st.info("Upload a farmer response Excel/CSV file to get started.")
    st.caption(
        "Expected: columns matching the 60-item Farm Stress Inventory "
        "(Very Low / Low / Medium / High / Very high stress labels, or "
        "numeric 1-5 scores)."
    )
