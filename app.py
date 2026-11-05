import io
import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Student Academic Performance Analytics",
    page_icon="🎓",
    layout="wide",
)

SAMPLE_FILE = "student-mat.csv"
STUDY_TIME_LABELS = {1: "1 - < 2 h", 2: "2 - 2 to 5 h", 3: "3 - 5 to 10 h", 4: "4 - > 10 h"}
CATEGORIES = ["Excellent", "Pass", "Fail"]
CATEGORY_COLORS = {"Excellent": "#2E7D32", "Pass": "#F9A825", "Fail": "#C62828"}
REQUIRED_COLUMNS = ["school", "sex", "studytime", "absences", "G1", "G2", "G3"]

DEFAULT_FILTERS = {
    "f_sex": ["F", "M"],
    "f_school": ["GP", "MS"],
    "f_studytime": [1, 2, 3, 4],
    "f_min_g3": 0,
}

def detect_separator(raw_bytes: bytes) -> str:
    first_line = raw_bytes.decode("utf-8", errors="ignore").splitlines()[0]
    return ";" if first_line.count(";") > first_line.count(",") else ","


@st.cache_data(show_spinner="Reading dataset ...")
def load_data(file_bytes: bytes, sep: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    df.columns = [str(c).strip() for c in df.columns]


    for col in ["G1", "G2", "G3", "absences", "studytime"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

   
    return df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])


def classify_grade(g3: float) -> str:
    """7. Excellent 15-20, Pass 10-14, Fail 0-9."""
    if pd.isna(g3):
        return "Fail"
    return "Excellent" if g3 >= 15 else "Pass" if g3 >= 10 else "Fail"


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """3. Filter by gender, school, study time and minimum final grade."""
    return df[
        df["sex"].isin(filters["f_sex"])
        & df["school"].isin(filters["f_school"])
        & df["studytime"].isin(filters["f_studytime"])
        & (df["G3"].fillna(0) >= filters["f_min_g3"])
    ]



st.sidebar.title("Student Academic Performance")
uploaded_file = st.sidebar.file_uploader(
    "Upload student performance CSV",
    type=["csv"],
    help="UCI student-mat.csv (semicolon separated). No file? The bundled sample is used.",
)

if uploaded_file is not None:
    raw_bytes, source = uploaded_file.getvalue(), f"Uploaded: **{uploaded_file.name}**"
elif os.path.exists(SAMPLE_FILE):
    raw_bytes, source = open(SAMPLE_FILE, "rb").read(), f"Bundled sample: **{SAMPLE_FILE}**"
else:
    st.title("Student Academic Performance Analytics Dashboard")
    st.info(
        "Upload **student-mat.csv** in the sidebar to begin.\n\n"
        "Source: UCI ML Repository - Student Performance "
        "(https://archive.ics.uci.edu/dataset/320/student+performance)"
    )
    st.stop()

sep = detect_separator(raw_bytes)
df_raw = load_data(raw_bytes, sep)

missing_cols = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
if missing_cols:
    st.error("Missing required column(s): " + ", ".join(f"`{c}`" for c in missing_cols))
    st.stop()


for key, value in DEFAULT_FILTERS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("applied_filters", dict(DEFAULT_FILTERS))


def reset_filters() -> None:
    """on_click callback: runs before the widgets are created, so keys may be reassigned."""
    for key, value in DEFAULT_FILTERS.items():
        st.session_state[key] = value
    st.session_state["applied_filters"] = dict(DEFAULT_FILTERS)


with st.sidebar.form("filter_form"):
    st.multiselect("Gender", sorted(df_raw["sex"].dropna().unique()), key="f_sex")
    st.multiselect("School", sorted(df_raw["school"].dropna().unique()), key="f_school")
    st.multiselect(
        "Study time",
        sorted(df_raw["studytime"].dropna().unique()),
        format_func=lambda v: STUDY_TIME_LABELS.get(v, str(v)),
        key="f_studytime",
    )
    st.slider("Minimum final grade (G3)", 0, int(df_raw["G3"].max()), key="f_min_g3")
    if st.form_submit_button("Apply Filters", type="primary", width="stretch"):
        st.session_state["applied_filters"] = {k: st.session_state[k] for k in DEFAULT_FILTERS}

st.sidebar.button("Reset Filters", width="stretch", on_click=reset_filters)


df = apply_filters(df_raw, st.session_state["applied_filters"]).copy()
df["Performance"] = df["G3"].apply(classify_grade)

st.title("Student Academic Performance Analytics Dashboard")
st.caption("UCI Student Performance dataset (Mathematics). Metrics and charts follow the sidebar filters.")

if df.empty:
    st.warning("No records match the selected filters. Use **Reset Filters** in the sidebar.")
    st.stop()

tab_overview, tab_charts, tab_class, tab_records = st.tabs(
    [" Overview", " Visualisations", " Classification", " Records & Export"]
)


with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total students", f"{len(df):,}")
    c2.metric("Average G3", f"{df['G3'].mean():.2f}")
    c3.metric("Highest G3", int(df["G3"].max()))
    c4.metric("Average absences", f"{df['absences'].mean():.2f}")
    c5.metric("Students with G3 ≥ 10", f"{int((df['G3'] >= 10).sum()):,}")

    st.subheader("First 10 rows")
    st.dataframe(df.drop(columns=["Performance"]).head(10), width="stretch")

    st.subheader("Dataset summary")
    a, b, c = st.columns(3)
    a.write(f"**Dimensions:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    b.write(f"**Missing values:** {int(df_raw.isna().sum().sum())}")
    c.write(f"**Duplicate rows:** {int(df_raw.duplicated().sum())}")

    with st.expander("Column information, missing values and duplicates"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Column": df_raw.columns,
                    "Data type": [str(t) for t in df_raw.dtypes],
                    "Missing": df_raw.isna().sum().values,
                    "Unique": df_raw.nunique().values,
                }
            ),
            width="stretch",
        )


with tab_charts:
    by_study = df.groupby("studytime", as_index=False).agg(avg_g3=("G3", "mean"))
    by_study["Study time"] = by_study["studytime"].map(STUDY_TIME_LABELS)
    fig = px.bar(
        by_study, x="Study time", y="avg_g3", text=by_study["avg_g3"].round(2),
        labels={"avg_g3": "Average G3"}, color="avg_g3", color_continuous_scale="Blues",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 20], coloraxis_showscale=False, height=380)
    st.subheader("Average G3 by weekly study time")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        gender = df["sex"].value_counts().rename_axis("Gender").reset_index(name="Students")
        gender["Gender"] = gender["Gender"].map({"F": "Female", "M": "Male"})
        fig = px.pie(
            gender, names="Gender", values="Students", hole=0.45, color="Gender",
            color_discrete_map={"Female": "#E91E63", "Male": "#1565C0"},
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=380)
        st.subheader("Students by gender")
        st.plotly_chart(fig, width="stretch")

    with right:
        fig = px.scatter(
            df, x="absences", y="G3", color="sex",
            color_discrete_map={"F": "#E91E63", "M": "#1565C0"},
            labels={"absences": "Absences", "G3": "Final grade (G3)", "sex": "Gender"},
        )
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
        fig.update_layout(height=380)
        st.subheader("Absences vs final grade")
        st.plotly_chart(fig, width="stretch")

    fig = px.histogram(
        df, x="G3", nbins=21, color="Performance",
        category_orders={"Performance": CATEGORIES}, color_discrete_map=CATEGORY_COLORS,
        labels={"G3": "Final grade (G3)"},
    )
    fig.update_layout(barmode="stack", height=400, xaxis=dict(dtick=1), legend_title="Performance")
    st.subheader("Distribution of final grades")
    st.plotly_chart(fig, width="stretch")


with tab_class:
    counts = df["Performance"].value_counts().reindex(CATEGORIES).fillna(0).astype(int)
    c1, c2, c3 = st.columns(3)
    c1.metric("Excellent (15–20)", f"{counts['Excellent']:,}")
    c2.metric("Pass (10–14)", f"{counts['Pass']:,}")
    c3.metric("Fail (0–9)", f"{counts['Fail']:,}")

    cat_df = counts.rename_axis("Performance").reset_index(name="Students")
    fig = px.bar(
        cat_df, x="Performance", y="Students", color="Performance", text="Students",
        category_orders={"Performance": CATEGORIES}, color_discrete_map=CATEGORY_COLORS,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, width="stretch")

with tab_records:
    st.subheader("Filtered records")
    st.caption(f"{len(df):,} of {len(df_raw):,} records shown.")
    st.dataframe(df, width="stretch", height=460)
    st.download_button(
        "⬇️ Download filtered records as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_student_performance.csv",
        mime="text/csv",
        type="primary",
    )
