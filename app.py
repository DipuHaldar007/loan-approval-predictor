import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# DARK THEME + CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* =====================================================
       GLOBAL
    ===================================================== */

    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
    }

    .main {
        background-color: #0b1120;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       HEADER
    ===================================================== */

    .loan-header {
        background: linear-gradient(135deg, #111827, #1d4ed8);
        border: 1px solid #263b73;
        border-radius: 18px;
        padding: 30px 35px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }

    .loan-header-title {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 750;
        margin: 0;
    }

    .loan-header-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        margin-top: 8px;
        margin-bottom: 0;
    }


    /* =====================================================
       INFO BOX
    ===================================================== */

    .info-box {
        background-color: #111c32;
        border: 1px solid #1e40af;
        border-left: 5px solid #3b82f6;
        border-radius: 12px;
        padding: 15px 18px;
        margin-bottom: 20px;
    }

    .info-title {
        color: #93c5fd;
        font-weight: 700;
        font-size: 0.95rem;
    }

    .info-text {
        color: #cbd5e1;
        font-size: 0.88rem;
        margin-top: 4px;
    }


    /* =====================================================
       SECTION CARDS
    ===================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #111827 !important;
        border: 1px solid #263244 !important;
        border-radius: 16px !important;
        padding: 8px !important;
        margin-top: 18px !important;
        margin-bottom: 18px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.20);
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 3px;
    }

    .section-description {
        color: #94a3b8;
        font-size: 0.86rem;
        margin-bottom: 16px;
    }


    /* =====================================================
       WIDGET LABELS
       IMPORTANT: MAKES AGE / GENDER / ETC. VISIBLE
    ===================================================== */

    [data-testid="stWidgetLabel"] {
        color: #f8fafc !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stWidgetLabel"] label {
        color: #f8fafc !important;
    }


    /* =====================================================
       NUMBER INPUT
    ===================================================== */

    .stNumberInput input {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 9px !important;
    }

    .stNumberInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }


    /* =====================================================
       SELECTBOX
    ===================================================== */

    div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 9px !important;
    }

    div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }

    div[data-baseweb="select"] input {
        color: #f8fafc !important;
    }


    /* =====================================================
       SLIDER
    ===================================================== */

    [data-testid="stSlider"] {
        padding-top: 5px;
    }

    [data-testid="stSlider"] p {
        color: #f8fafc !important;
    }


    /* =====================================================
       HELP ICON
    ===================================================== */

    [data-testid="stTooltipIcon"] {
        color: #93c5fd !important;
    }


    /* =====================================================
       METRICS
    ===================================================== */

    [data-testid="stMetric"] {
        background-color: #111827 !important;
        border: 1px solid #263244 !important;
        border-radius: 12px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }


    /* =====================================================
       BUTTON
    ===================================================== */

    .stButton > button {
        width: 100%;
        min-height: 3.2rem;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 700;
    }


    /* =====================================================
       APPROVED RESULT
    ===================================================== */

    .approved-card {
        background-color: #052e1b;
        border: 1px solid #16a34a;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 20px;
    }

    .approved-title {
        color: #4ade80;
        font-size: 2rem;
        font-weight: 800;
    }

    .approved-text {
        color: #bbf7d0;
        font-size: 1rem;
        margin-top: 7px;
    }


    /* =====================================================
       DENIED RESULT
    ===================================================== */

    .denied-card {
        background-color: #3a0d12;
        border: 1px solid #dc2626;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 20px;
    }

    .denied-title {
        color: #f87171;
        font-size: 2rem;
        font-weight: 800;
    }

    .denied-text {
        color: #fecaca;
        font-size: 1rem;
        margin-top: 7px;
    }


    /* =====================================================
       EXPANDER
    ===================================================== */

    [data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid #263244 !important;
        border-radius: 12px !important;
    }


    /* =====================================================
       FOOTER
    ===================================================== */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 35px;
        padding-top: 18px;
        border-top: 1px solid #1e293b;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("tuned_loan_model.pkl")


try:
    model = load_model()

except Exception:

    st.error("⚠️ Unable to load the loan prediction model.")

    st.info(
        "Please make sure 'tuned_loan_model.pkl' is present "
        "in the same GitHub repository as app.py."
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="loan-header">'
    '<div class="loan-header-title">🏦 Loan Approval Predictor</div>'
    '<div class="loan-header-subtitle">'
    'AI-powered prediction of loan approval based on applicant '
    'financial and personal information.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INFORMATION BOX
# =========================================================

st.markdown(
    '<div class="info-box">'
    '<div class="info-title">💡 Enter Applicant Information</div>'
    '<div class="info-text">'
    'Fill in the details below and click '
    '<b>Predict Loan Approval</b> to get the model prediction.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PERSONAL INFORMATION
# =========================================================

with st.container(border=True):

    st.markdown(
        '<div class="section-title">👤 Personal Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Basic information about the loan applicant.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
            help="Applicant's age in years."
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col3:

        marital = st.selectbox(
            "Marital Status",
            ["Single", "Married"]
        )


# =========================================================
# EMPLOYMENT & EDUCATION
# =========================================================

with st.container(border=True):

    st.markdown(
        '<div class="section-title">💼 Employment & Education</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Information about the applicant\'s education and employment.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        education = st.selectbox(
            "Education Level",
            ["Not Graduated", "Graduated"]
        )

    with col2:

        employment = st.selectbox(
            "Employment Status",
            [
                "Salaried",
                "Self-employed",
                "Unemployed"
            ]
        )

    with col3:

        employer_cat = st.selectbox(
            "Employer Category",
            [
                "Government",
                "MNC",
                "Private",
                "Unemployed",
                "Other"
            ]
        )


# =========================================================
# FINANCIAL INFORMATION
# =========================================================

with st.container(border=True):

    st.markdown(
        '<div class="section-title">💰 Financial Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Applicant income, savings and existing financial obligations.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        app_income = st.number_input(
            "Applicant Income (₹ / month)",
            min_value=0,
            value=50000,
            step=1000,
            help="Applicant's monthly income in Indian Rupees."
        )

    with col2:

        coapp_income = st.number_input(
            "Coapplicant Income (₹ / month)",
            min_value=0,
            value=0,
            step=1000,
            help="Coapplicant's monthly income. Enter 0 if there is no coapplicant."
        )

    with col3:

        savings = st.number_input(
            "Savings (₹)",
            min_value=0,
            value=5000,
            step=1000,
            help="Applicant's savings amount."
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            value=0,
            step=1,
            help="Number of currently active loans."
        )

    with col2:

        dependents = st.number_input(
            "Dependents",
            min_value=0,
            value=0,
            step=1,
            help="Number of people financially dependent on the applicant."
        )

    with col3:

        credit_score = st.slider(
            "Credit Score (300–850)",
            min_value=300,
            max_value=850,
            value=500,
            step=1,
            help="Applicant's credit score."
        )


# =========================================================
# LOAN INFORMATION
# =========================================================

with st.container(border=True):

    st.markdown(
        '<div class="section-title">🏠 Loan Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Details about the requested loan.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=0,
            value=100000,
            step=1000,
            help="Total amount of loan being requested."
        )

    with col2:

        loan_term = st.number_input(
            "Loan Term (months)",
            min_value=12,
            value=36,
            step=12,
            help="Repayment period in months. Example: 60 months = 5 years."
        )

    with col3:

        loan_purpose = st.selectbox(
            "Loan Purpose",
            [
                "Car",
                "Education",
                "Home",
                "Personal",
                "Other"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        property_area = st.selectbox(
            "Property Area",
            [
                "Urban",
                "Semiurban",
                "Rural"
            ]
        )

    with col2:

        collateral = st.number_input(
            "Collateral Value (₹)",
            min_value=0,
            value=10000,
            step=1000,
            help="Estimated value of the collateral."
        )


# =========================================================
# DTI
# =========================================================

with st.container(border=True):

    st.markdown(
        '<div class="section-title">📊 Debt-to-Income Ratio</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Enter the applicant\'s DTI ratio as a percentage.'
        '</div>',
        unsafe_allow_html=True
    )

    dti_percent = st.slider(
        "DTI Ratio (%)",
        min_value=0,
        max_value=100,
        value=30,
        step=1,
        help=(
            "Example: 30% is internally converted to 0.30 "
            "for the machine learning model."
        )
    )

    # Convert percentage to the original model format
    dti_ratio = dti_percent / 100


# =========================================================
# APPLICATION SUMMARY
# =========================================================

st.markdown(
    '<div class="section-title" style="margin-top:30px;">'
    '📋 Application Summary'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Monthly Income",
        f"₹{app_income:,.0f}"
    )

with col2:

    st.metric(
        "Loan Amount",
        f"₹{loan_amount:,.0f}"
    )

with col3:

    st.metric(
        "Credit Score",
        f"{credit_score}"
    )

with col4:

    st.metric(
        "DTI Ratio",
        f"{dti_percent}%"
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    predict_button = st.button(
        "🔮 Predict Loan Approval",
        type="primary",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # =================================================
        # EXACT INPUT STRUCTURE
        # =================================================

        input_dict = {

            "Applicant_Income": app_income,

            "Coapplicant_Income": coapp_income,

            "Age": age,

            "Dependents": dependents,

            "Existing_Loans": existing_loans,

            "Savings": savings,

            "Collateral_Value": collateral,

            "Loan_Amount": loan_amount,

            "Loan_Term": loan_term,

            "Education_Level":
                1 if education == "Not Graduated" else 0,

            "Employment_Status_Salaried":
                1 if employment == "Salaried" else 0,

            "Employment_Status_Self-employed":
                1 if employment == "Self-employed" else 0,

            "Employment_Status_Unemployed":
                1 if employment == "Unemployed" else 0,

            "Marital_Status_Single":
                1 if marital == "Single" else 0,

            "Loan_Purpose_Car":
                1 if loan_purpose == "Car" else 0,

            "Loan_Purpose_Education":
                1 if loan_purpose == "Education" else 0,

            "Loan_Purpose_Home":
                1 if loan_purpose == "Home" else 0,

            "Loan_Purpose_Personal":
                1 if loan_purpose == "Personal" else 0,

            "Property_Area_Semiurban":
                1 if property_area == "Semiurban" else 0,

            "Property_Area_Urban":
                1 if property_area == "Urban" else 0,

            "Gender_Male":
                1 if gender == "Male" else 0,

            "Employer_Category_Government":
                1 if employer_cat == "Government" else 0,

            "Employer_Category_MNC":
                1 if employer_cat == "MNC" else 0,

            "Employer_Category_Private":
                1 if employer_cat == "Private" else 0,

            "Employer_Category_Unemployed":
                1 if employer_cat == "Unemployed" else 0,

            "Credit_Score_sq":
                credit_score ** 2,

            "DTI_Ratio_sq":
                dti_ratio ** 2,

            "Applicant_Income_log":
                np.log1p(app_income)
        }


        # =================================================
        # DATAFRAME
        # =================================================

        input_df = pd.DataFrame([input_dict])


        # =================================================
        # PREDICTION
        # =================================================

        prediction = model.predict(input_df)


        # =================================================
        # RESULT
        # =================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '🎯 Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )


        if prediction[0] == 1:

            st.markdown(
                '<div class="approved-card">'
                '<div class="approved-title">'
                '🎉 LOAN APPROVED'
                '</div>'
                '<div class="approved-text">'
                'The model predicts this application as '
                '<b>low-risk</b>.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.balloons()


        else:

            st.markdown(
                '<div class="denied-card">'
                '<div class="denied-title">'
                '🚫 LOAN DENIED'
                '</div>'
                '<div class="denied-text">'
                'The model predicts this application as '
                '<b>high-risk</b>.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )


        # =================================================
        # TECHNICAL DETAILS
        # =================================================

        with st.expander("🔧 View Model Input Details"):

            st.write(
                "These are the values sent to the machine learning model:"
            )

            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True
            )

            st.write(
                f"Predicted Class: `{prediction[0]}`"
            )


    except Exception as e:

        st.error(
            "❌ An error occurred while making the prediction."
        )

        with st.expander("Technical Error"):

            st.code(str(e))


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '🏦 Loan Approval Predictor '
    '&nbsp;•&nbsp; '
    'Machine Learning Project '
    '&nbsp;•&nbsp; '
    'Built with Python & Streamlit'
    '</div>',
    unsafe_allow_html=True
)