import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ml_model import train_model, predict_performance

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EduTrack | Student Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. WHITE AND LIGHT GREEN THEME
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F7FAF7;
    color: #263B2D;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E4F3E7, #F5FAF5);
    border-right: 1px solid #D5E8D9;
}

h1, h2, h3 {
    color: #245C3A !important;
    font-weight: 700 !important;
}

p, label {
    color: #405747;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCEBDD;
    border-left: 5px solid #79B889;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(40, 90, 50, 0.05);
}

[data-testid="stMetricLabel"] {
    color: #587461 !important;
}

[data-testid="stMetricValue"] {
    color: #245C3A !important;
    font-weight: 800;
}

.stButton > button,
.stFormSubmitButton > button {
    background-color: #398653;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.55rem 1rem;
    font-weight: 600;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #286A40;
    color: white;
    border: none;
}

.stTextInput input,
.stNumberInput input {
    border: 1px solid #C9DFCE;
    border-radius: 8px;
    background-color: white;
}

[data-testid="stDataFrame"] {
    border: 1px solid #DCEBDD;
    border-radius: 10px;
    overflow: hidden;
}

.hero-card {
    background: linear-gradient(120deg, #E3F4E6, #FFFFFF);
    border: 1px solid #D1E8D5;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 25px;
}

.hero-card h1 {
    font-size: 30px;
    margin-bottom: 8px;
}

.hero-card p {
    color: #587461;
}

.section-label {
    color: #398653;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.form-card {
    background: #FFFFFF;
    padding: 22px;
    border: 1px solid #DCEBDD;
    border-radius: 14px;
    margin-bottom: 20px;
}

hr {
    border-color: #DCEBDD;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
    }

    .hero-card {
        padding: 18px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. INITIAL STUDENT DATA
# =========================================================

initial_data = {
    "Student_ID": [
        101, 102, 103, 104, 105,
        106, 107, 108, 109, 110,
        111, 112, 113, 114, 115
    ],

    "Student_Name": [
        "Arun", "Priya", "Kavin", "Divya", "Rahul",
        "Meena", "Ajay", "Kavya", "Vijay", "Anu",
        "Sanjay", "Nisha", "Hari", "Swetha", "Ravi"
    ],

    "Department": [
        "IT", "CSE", "IT", "ECE", "CSE",
        "IT", "ECE", "CSE", "IT", "ECE",
        "CSE", "IT", "ECE", "CSE", "IT"
    ],

    "Marks": [
        85, 92, 67, 74, 88,
        95, 56, 81, 45, 78,
        62, 90, 39, 84, 71
    ],

    "Attendance": [
        92, 96, 78, 85, 90,
        98, 72, 88, 65, 86,
        75, 94, 60, 91, 82
    ]
}

initial_df = pd.DataFrame(initial_data)


# =========================================================
# 4. DATA PREPARATION FUNCTIONS
# =========================================================

def prepare_data(dataframe):

    dataframe = dataframe.copy()

    dataframe = dataframe.drop_duplicates(
        subset=["Student_ID"]
    )

    dataframe = dataframe.dropna(
        subset=[
            "Student_Name",
            "Department",
            "Marks",
            "Attendance"
        ]
    )

    dataframe["Performance"] = dataframe["Marks"].apply(
        lambda marks:
        "Excellent" if marks >= 85
        else "Good" if marks >= 70
        else "Average" if marks >= 50
        else "Needs Improvement"
    )

    dataframe["Result"] = dataframe["Marks"].apply(
        lambda marks: "Pass" if marks >= 40 else "Fail"
    )

    return dataframe


# =========================================================
# 5. SESSION STATE
# =========================================================

# Keeps added students available while this Streamlit session
# remains active.

if "student_data" not in st.session_state:

    st.session_state.student_data = prepare_data(
        initial_df
    )

if "added_student_message" not in st.session_state:

    st.session_state.added_student_message = ""

# Always read the latest student data.

df = prepare_data(
    st.session_state.student_data
)


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;padding:15px 0;">
            <div style="font-size:42px;">🎓</div>
            <h2 style="margin:5px 0;">EduTrack</h2>
            <p style="font-size:12px;">STUDENT ANALYTICS</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Choose a page",
        [
            "🏠 Dashboard",
            "➕ Add Student",
            "👨‍🎓 Student Records",
            "📊 Performance Analysis",
            "🔍 Student Search",
            "📑 Summary Statistics",
            "🤖 ML Prediction"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Department Filter")

    departments = sorted(df["Department"].unique())

    selected_departments = st.multiselect(
        "Select departments",
        options=departments,
        default=departments
    )

    performance_levels = [
        "Excellent",
        "Good",
        "Average",
        "Needs Improvement"
    ]

    st.markdown("### Performance Filter")

    selected_performance = st.multiselect(
        "Select performance levels",
        options=performance_levels,
        default=performance_levels
    )

    st.divider()

    st.caption("EduTrack Dashboard")
    st.caption("Python • Streamlit • Pandas")


# =========================================================
# 7. APPLY FILTERS
# =========================================================

filtered_df = df[
    df["Department"].isin(selected_departments)
    & df["Performance"].isin(selected_performance)
]


# =========================================================
# 8. CHART COLORS
# =========================================================

GREEN = "#398653"
LIGHT_GREEN = "#A8D5B5"
PALE_GREEN = "#DFF0E3"


# =========================================================
# 9. ADD STUDENT PAGE
# =========================================================

if page == "➕ Add Student":

    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">STUDENT MANAGEMENT</div>
            <h1>➕ Add New Student</h1>
            <p>
                Register a student and enter their academic details.
                The new record will appear in the dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.added_student_message:

        st.success(
            st.session_state.added_student_message
        )

        st.session_state.added_student_message = ""

    st.markdown("### 📝 Student Information")

    with st.form("add_student_form"):

        col1, col2 = st.columns(2)

        with col1:

            student_id = st.number_input(
                "Student ID",
                min_value=1,
                step=1,
                value=101,
                help="Enter a unique student ID."
            )

            student_name = st.text_input(
                "Student Name",
                placeholder="Enter full name"
            )

            department = st.selectbox(
                "Department",
                ["IT", "CSE", "ECE"]
            )

        with col2:

            marks = st.number_input(
                "Marks",
                min_value=0,
                max_value=100,
                value=75,
                help="Enter marks out of 100."
            )

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0,
                max_value=100,
                value=85
            )

        st.markdown("---")

        submitted = st.form_submit_button(
            "➕ Add Student",
            use_container_width=True
        )

        if submitted:

            student_name = student_name.strip()

            if not student_name:

                st.error(
                    "Please enter the student's name."
                )

            elif int(student_id) in df["Student_ID"].values:

                st.error(
                    "This Student ID already exists. "
                    "Please enter a different ID."
                )

            else:

                new_student = pd.DataFrame([{
                    "Student_ID": int(student_id),
                    "Student_Name": student_name,
                    "Department": department,
                    "Marks": int(marks),
                    "Attendance": int(attendance)
                }])

                updated_df = pd.concat(
                    [
                        st.session_state.student_data,
                        new_student
                    ],
                    ignore_index=True
                )

                st.session_state.student_data = prepare_data(
                    updated_df
                )

                st.session_state.added_student_message = (
                    f"{student_name} was added successfully!"
                )

                st.rerun()

    st.markdown("### 💡 How it works")

    info1, info2, info3 = st.columns(3)

    with info1:

        st.markdown(
            """
            <div class="form-card">
                <h3>01. Enter Details</h3>
                <p>Add the student's ID, name, department,
                marks and attendance.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info2:

        st.markdown(
            """
            <div class="form-card">
                <h3>02. Automatic Analysis</h3>
                <p>The application calculates the performance
                category and pass/fail result.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info3:

        st.markdown(
            """
            <div class="form-card">
                <h3>03. View Records</h3>
                <p>Open Student Records or Dashboard to
                see the updated information.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 10. DASHBOARD PAGE
# =========================================================

elif page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">ACADEMIC OVERVIEW</div>
            <h1>Student Performance Dashboard</h1>
            <p>
                Monitor student marks, track attendance and
                explore academic performance in one place.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_students = len(filtered_df)

    average_marks = (
        filtered_df["Marks"].mean()
        if total_students else 0
    )

    pass_percentage = (
        (filtered_df["Result"] == "Pass").mean() * 100
        if total_students else 0
    )

    average_attendance = (
        filtered_df["Attendance"].mean()
        if total_students else 0
    )

    st.markdown("### 📌 Key Performance Indicators")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", total_students)

    c2.metric(
        "Average Marks",
        f"{average_marks:.1f}/100"
    )

    c3.metric(
        "Pass Percentage",
        f"{pass_percentage:.1f}%"
    )

    c4.metric(
        "Average Attendance",
        f"{average_attendance:.1f}%"
    )

    st.markdown("### 📈 Academic Overview")

    if not filtered_df.empty:

        left, right = st.columns(2)

        with left:

            st.markdown("#### Department Performance")

            department_average = (
                filtered_df.groupby("Department")["Marks"]
                .mean()
                .sort_values(ascending=False)
            )

            fig, ax = plt.subplots(figsize=(7, 4))

            ax.bar(
                department_average.index,
                department_average.values,
                color=LIGHT_GREEN,
                edgecolor=GREEN
            )

            ax.set_ylabel("Average Marks")
            ax.set_ylim(0, 100)
            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)

        with right:

            st.markdown("#### Performance Distribution")

            performance_counts = (
                filtered_df["Performance"].value_counts()
            )

            fig, ax = plt.subplots(figsize=(7, 4))

            ax.pie(
                performance_counts.values,
                labels=performance_counts.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=[
                    "#398653",
                    "#78B88A",
                    "#B8DDBF",
                    "#E1EFE4"
                ],
                wedgeprops={
                    "edgecolor": "white",
                    "linewidth": 2
                }
            )

            fig.patch.set_facecolor("white")

            st.pyplot(fig)
            plt.close(fig)

    else:

        st.info(
            "No students match the selected filters."
        )

    st.markdown("### 👨‍🎓 Recent Student Records")

    st.dataframe(
        filtered_df.head(5),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 11. STUDENT RECORDS PAGE
# =========================================================

elif page == "👨‍🎓 Student Records":

    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">STUDENT DATABASE</div>
            <h1>Student Records</h1>
            <p>View, filter and download student information.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(f"**Total matching records:** {len(filtered_df)}")

    selected_columns = st.multiselect(
        "Choose columns to display",
        options=df.columns.tolist(),
        default=[
            "Student_ID",
            "Student_Name",
            "Department",
            "Marks",
            "Attendance",
            "Performance",
            "Result"
        ]
    )

    if selected_columns:

        st.dataframe(
            filtered_df[selected_columns],
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered_df[selected_columns].to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Student Records",
            data=csv_data,
            file_name="student_records.csv",
            mime="text/csv"
        )

    else:

        st.info("Select at least one column.")

    st.markdown("### Marks Overview")

    if not filtered_df.empty:

        st.bar_chart(
            filtered_df.set_index("Student_Name")["Marks"],
            color=GREEN
        )


# =========================================================
# 12. PERFORMANCE ANALYSIS PAGE
# =========================================================

elif page == "📊 Performance Analysis":

    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">ACADEMIC INSIGHTS</div>
            <h1>Performance Analysis</h1>
            <p>Compare student marks, attendance and departments.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not filtered_df.empty:

        left, right = st.columns(2)

        with left:

            st.markdown("### Marks Comparison")

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.bar(
                filtered_df["Student_Name"],
                filtered_df["Marks"],
                color=LIGHT_GREEN,
                edgecolor=GREEN
            )

            ax.set_xlabel("Student")
            ax.set_ylabel("Marks")
            ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=60)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with right:

            st.markdown("### Attendance Comparison")

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.bar(
                filtered_df["Student_Name"],
                filtered_df["Attendance"],
                color=PALE_GREEN,
                edgecolor=GREEN
            )

            ax.set_xlabel("Student")
            ax.set_ylabel("Attendance (%)")
            ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=60)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("### Department-wise Performance")

        department_summary = (
            filtered_df.groupby("Department")
            .agg(
                Students=("Student_ID", "count"),
                Average_Marks=("Marks", "mean"),
                Average_Attendance=("Attendance", "mean")
            )
            .round(2)
            .sort_values(
                "Average_Marks",
                ascending=False
            )
        )

        st.dataframe(
            department_summary,
            use_container_width=True
        )

        st.markdown("### Performance Category Summary")

        performance_summary = (
            filtered_df.groupby("Performance")
            .agg(
                Student_Count=("Student_ID", "count"),
                Average_Marks=("Marks", "mean")
            )
            .round(2)
        )

        st.dataframe(
            performance_summary,
            use_container_width=True
        )

    else:

        st.warning("No matching records found.")


# =========================================================
# 13. STUDENT SEARCH PAGE
# =========================================================

elif page == "🔍 Student Search":

    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">QUICK LOOKUP</div>
            <h1>Find a Student</h1>
            <p>Search using a student's name or ID.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    search_type = st.radio(
        "Search by",
        ["Student Name", "Student ID"],
        horizontal=True
    )

    search_value = st.text_input(
        "Enter name or ID",
        placeholder="Example: Priya or 102"
    )

    if st.button("🔍 Search Student"):

        if not search_value.strip():

            st.warning("Please enter a name or ID.")

        elif search_type == "Student Name":

            results = filtered_df[
                filtered_df["Student_Name"].str.contains(
                    search_value.strip(),
                    case=False,
                    regex=False
                )
            ]

            if not results.empty:

                st.success(
                    f"Found {len(results)} matching record(s)."
                )

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.warning("No matching student found.")

        else:

            if search_value.strip().isdigit():

                results = filtered_df[
                    filtered_df["Student_ID"]
                    == int(search_value.strip())
                ]

                if not results.empty:

                    st.success("Student record found!")

                    st.dataframe(
                        results,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.warning("No matching student found.")

            else:

                st.warning("Please enter a numeric Student ID.")


# =========================================================
# 14. SUMMARY STATISTICS PAGE
# =========================================================

elif page == "📑 Summary Statistics":

    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">DATA INSIGHTS</div>
            <h1>Summary Statistics</h1>
            <p>Review statistical information from the dataset.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not filtered_df.empty:

        st.markdown("### Marks and Attendance Statistics")

        stats = filtered_df[
            ["Marks", "Attendance"]
        ].describe().T

        st.dataframe(
            stats.round(2),
            use_container_width=True
        )

        st.markdown("### 🏆 Top Performing Students")

        top_students = (
            filtered_df.sort_values(
                "Marks",
                ascending=False
            )
            .head(5)
        )

        st.dataframe(
            top_students[
                [
                    "Student_Name",
                    "Department",
                    "Marks",
                    "Attendance",
                    "Performance"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### ⚠️ Students Requiring Attention")

        attention_students = filtered_df[
            (filtered_df["Marks"] < 50)
            | (filtered_df["Attendance"] < 75)
        ]

        if not attention_students.empty:

            st.dataframe(
                attention_students[
                    [
                        "Student_Name",
                        "Department",
                        "Marks",
                        "Attendance",
                        "Performance"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No students meet the attention criteria."
            )

    else:

        st.info("No records available.")
# =========================================================
# 15. MACHINE LEARNING PREDICTION PAGE
# =========================================================

elif page == "🤖 ML Prediction":
    st.success("ML Prediction page opened successfully!")


    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">MACHINE LEARNING</div>
            <h1>🤖 Student Performance Prediction</h1>
            <p>
                Explore a prototype ML model using student
                marks and attendance.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Educational prototype only. The model learns "
        "performance categories derived from marks, so its "
        "predictions are not independent forecasts of "
        "future academic performance."
    )

    st.markdown("### Enter Student Details")

    col1, col2 = st.columns(2)

    with col1:
        input_marks = st.number_input(
            "Marks",
            min_value=0,
            max_value=100,
            value=75,
            key="ml_marks"
        )

    with col2:
        input_attendance = st.number_input(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=85,
            key="ml_attendance"
        )

    if st.button("🔮 Predict Performance"):

        try:
            model, accuracy, report = train_model(df)

            prediction, confidence = predict_performance(
                model,
                input_marks,
                input_attendance
            )

            st.markdown("### Prediction Result")

            st.success(
                f"Predicted Performance: {prediction}"
            )

            st.metric(
                "Held-out Test Accuracy",
                f"{accuracy * 100:.1f}%"
            )

            st.caption(
                "Test accuracy is calculated using a small "
                "hold-out sample. It is not a reliable estimate "
                "of performance on new students."
            )

            st.markdown("### Classification Report")

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.round(2),
                use_container_width=True
            )

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                f"Prediction could not be completed: {error}"
            )

# =========================================================
# 15. FOOTER
# =========================================================

st.divider()

st.caption(
    "🎓 EduTrack | Student Performance Analysis Dashboard"
)

st.caption(
    "Built using Python, Streamlit, Pandas and Matplotlib"
)