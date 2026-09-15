    from xmlrpc import client
    
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    import matplotlib.pyplot as plt
    import os
    from openai import OpenAI
    
    # =========================================================
    # PAGE SETTINGS
    # =========================================================
    
    st.set_page_config(
        page_title="Bank Marketing Analytics",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    
    # =========================================================
    # CUSTOM CSS
    # =========================================================
    
    st.markdown("""
    <style>
    
    /* =========================================
       GENERAL PAGE
       ========================================= */
    
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    header {
        visibility: hidden;
    }
    
    .stApp {
        background-color: #f4f7fb;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    
    /* =========================================
       SIDEBAR
       ========================================= */
    
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #071426 0%,
            #0b1f3a 55%,
            #102b4c 100%
        );
    }
    
    [data-testid="stSidebar"] * {
        color: white;
    }
    
    
    /* Sidebar radio buttons */
    
    [data-testid="stSidebar"] .stRadio label {
        color: #dbeafe !important;
    }
    
    
    <<<<<<< HEAD
    
    =======
    >>>>>>> 7d9bd05 (update Final Project with changes)
    /* Sidebar Brand */
    
    .sidebar-brand {
        padding: 20px 10px 10px 10px;
    }
    
    .brand-icon {
        font-size: 32px;
        margin-bottom: 5px;
    }
    
    .brand-title {
        font-size: 25px;
        font-weight: 800;
        color: white;
    }
    
    .brand-subtitle {
        font-size: 14px;
        color: #b8c7d9;
        margin-top: 5px;
    }
    
    .sidebar-nav-title {
        font-size: 15px;
        font-weight: 700;
        color: #dbeafe;
        margin: 10px 5px 8px 5px;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #29415f;
        margin: 10px 5px;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #dbeafe !important;
        font-size: 15px;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 8px;
    }
    
    /* =========================================
       MAIN TITLE
       ========================================= */
    
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #0b1f3a;
        margin-bottom: 5px;
    }
    
    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 25px;
    }
    
    
    /* =========================================
       SECTION TITLES
       ========================================= */
    
    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #0b1f3a;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    
    /* =========================================
       METRIC CARDS
       ========================================= */
    
    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #dbe5f0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 3px 10px rgba(11, 31, 58, 0.08);
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600;
    }
    
    [data-testid="stMetricValue"] {
        color: #0b1f3a !important;
        font-weight: 800;
    }
    
    
    /* =========================================
       BUTTONS
       ========================================= */
    
    .stButton > button {
        background: linear-gradient(
            90deg,
            #0b5ed7,
            #087f8c
        );
    
        color: white;
    
        border: none;
    
        border-radius: 10px;
    
        padding: 12px 20px;
    
        font-size: 17px;
    
        font-weight: 700;
    
        box-shadow: 0 4px 10px rgba(11, 94, 215, 0.25);
    
        transition: 0.2s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #084298,
            #05666f
        );
    
        color: white;
    
        transform: translateY(-2px);
    
        box-shadow: 0 6px 15px rgba(11, 94, 215, 0.35);
    }
    
    
    /* =========================================
       INPUT BOXES
       ========================================= */
    
    [data-baseweb="select"] > div {
        border-radius: 8px;
        border-color: #cbd5e1;
    }
    
    [data-testid="stNumberInput"] input {
        border-radius: 8px;
    }
    
    
    /* =========================================
       HEADERS
       ========================================= */
    
    h1, h2, h3 {
        color: #0b1f3a !important;
    }
    
    
    /* =========================================
       SUCCESS MESSAGE
       ========================================= */
    
    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    
    
    /* =========================================
       DIVIDERS
       ========================================= */
    
    hr {
        border-color: #dbe5f0;
    }
    
    
    /* =========================================
       DATAFRAME
       ========================================= */
    
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
    
    
    /* =========================================
       INFO BOX
       ========================================= */
    
    [data-testid="stAlert"][kind="info"] {
        border-left: 5px solid #0b5ed7;
    }
    
    
    /* =========================================
       FOOTER
       ========================================= */
    
    .small-text {
        color: #64748b;
        font-size: 14px;
    }
    
    </style>
    """, unsafe_allow_html=True)
    
    
    # =========================================================
    # LOAD FILES
    # =========================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    @st.cache_resource
    def load_model():
        return joblib.load(os.path.join(BASE_DIR, "bank_model.pkl"))
    
    
    @st.cache_resource
    def load_results():
        return joblib.load(os.path.join(BASE_DIR, "bank_results.pkl"))
    
    
    @st.cache_data
    def load_data():
        return pd.read_csv(
            os.path.join(BASE_DIR, "bank-additional-full.csv"),
            sep=";"
        )
    
    @st.cache_resource
    def load_openai():
        if "OPENAI_API_KEY" not in st.secrets:
            return None
    
        return OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"]
        )
    
    model = load_model()
    results = load_results()
    df = load_data()
    client = load_openai()
    
    # =========================================================
    # SIDEBAR
    # =========================================================
    
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                </div>
            <div style="
                text-align:center;
                padding:10px;
                font-size:25px;
                font-weight:bold;
            ">
            🏦 BankPrediction
            </div>
            """,
            unsafe_allow_html=True
        )
    
        st.markdown(
            "<hr>",
            unsafe_allow_html=True
        )
    
        st.markdown(
            '<div class="sidebar-nav-title">Navigation</div>',
            unsafe_allow_html=True
        )
    
        page = st.radio(
            "Navigation",
            [
                "🏠 Overview",
                "🔮 Make Prediction",
                "🤖 AI Assistant",
                "📊 Data Analysis",
                "🤖 Model Performance",
                "⭐ Feature Importance",
                "ℹ️ About Project"
            ],
            label_visibility="collapsed"
        )
    st.markdown("""<div style="
                text-align:center;
                color:#94a3b8;
                margin-bottom:25px;
            ">
            Bank Marketing Analytics
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("""<div style="
            font-size: 42px;
            font-weight: 800;
            color: #1e293b;
            margin-bottom: 5px;
        ">
        Bank Marketing Analytics
        </div>
        """,
        unsafe_allow_html=True
    )
    def generate_prediction_explanation(
        prediction,
        probability,
        customer_data,
        feature_importance
    ):
        if client is None:
            return "OpenAI API key is not configured."
    
        important_features = feature_importance.head(10).to_string(
            index=False
        )
    
        prompt = f"""
    You are an AI assistant inside a Bank Marketing Prediction System.
    
    The machine learning model is Logistic Regression.
    You must NOT change or invent the model prediction.
    
    Explain the existing prediction clearly and simply.
    
    Prediction:
    {prediction}
    
    Probability of subscribing:
    {probability:.2%}
    
    Customer information:
    {customer_data}
    
    Top model features:
    {important_features}
    
    Give the answer in this format:
    
    1. Prediction explanation
    2. Main factors
    3. Suggested marketing action
    
    Do not invent additional customer information.
    Do not claim that the prediction is guaranteed.
    """
    
        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )
    
        return response.output_text
    st.markdown("---")
    st.caption(
            "Machine Learning Project"
        )
    
    
    # =========================================================
    # OVERVIEW
    # =========================================================
    
    if page == "🏠 Overview":
    
        st.markdown(
            '<div class="main-title">'
            'Bank Marketing Analytics'
            '</div>',
            unsafe_allow_html=True
        )
    
        st.markdown(
            '<div class="subtitle">'
            'Machine Learning Dashboard for Term Deposit Subscription'
            '</div>',
            unsafe_allow_html=True
        )
    
        # ---------------------------------------------
        # DATA STATISTICS
        # ---------------------------------------------
    
        total_records = len(df)
    
        subscriptions = (
            df["y"] == "yes"
        ).sum()
    
        no_subscriptions = (
            df["y"] == "no"
        ).sum()
    
        subscription_rate = (
            subscriptions / total_records * 100
        )
    
        average_age = df["age"].mean()
    
        col1, col2, col3, col4, col5 = st.columns(5)
    
        with col1:
    
            st.metric(
                "Total Records",
                f"{total_records:,}"
            )
    
        with col2:
    
            st.metric(
                "Subscriptions",
                f"{subscriptions:,}"
            )
    
        with col3:
    
            st.metric(
                "No Subscription",
                f"{no_subscriptions:,}"
            )
    
        with col4:
    
            st.metric(
                "Subscription Rate",
                f"{subscription_rate:.2f}%"
            )
    
        with col5:
    
            st.metric(
                "Average Age",
                f"{average_age:.1f}"
            )
    
        st.divider()
    
        # ---------------------------------------------
        # CHARTS
        # ---------------------------------------------
    
        col1, col2 = st.columns(2)
    
        with col1:
    
            st.subheader(
                "Subscription Distribution"
            )
    
            target_counts = (
                df["y"]
                .value_counts()
            )
    
            st.bar_chart(
                target_counts
            )
    
        with col2:
    
            st.subheader(
                "Age Distribution"
            )
    
            age_counts = (
                df["age"]
                .value_counts()
                .sort_index()
            )
    
            st.line_chart(
                age_counts
            )
    
        st.divider()
    
        # ---------------------------------------------
        # MODEL RESULT
        # ---------------------------------------------
    
        st.subheader(
            "Model Performance"
        )
    
        col1, col2, col3, col4, col5 = st.columns(5)
    
        with col1:
    
            st.metric(
                "Accuracy",
                f"{results['accuracy']:.4f}"
            )
    
        with col2:
    
            st.metric(
                "Precision",
                f"{results['precision']:.4f}"
            )
    
        with col3:
    
            st.metric(
                "Recall",
                f"{results['recall']:.4f}"
            )
    
        with col4:
    
            st.metric(
                "F1 Score",
                f"{results['f1']:.4f}"
            )
    # =========================================================
    # MAKE PREDICTION
    # =========================================================
    
    elif page == "🔮 Make Prediction":
    
        st.markdown(
            '<div class="main-title">'
            'Customer Prediction'
            '</div>',
            unsafe_allow_html=True
        )
    
        st.markdown(
            '<div class="subtitle">'
            'Enter customer information to predict term deposit subscription.'
            '</div>',
            unsafe_allow_html=True
        )
    
    
        # ---------------------------------------------
        # CUSTOMER INFORMATION
        # ---------------------------------------------
    
        st.markdown(
            '<div class="section-title">'
            '👤 Customer Information'
            '</div>',
            unsafe_allow_html=True
        )
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
    
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30
            )
    
            job = st.selectbox(
                "Job",
                sorted(df["job"].unique())
            )
    
        with col2:
    
            marital = st.selectbox(
                "Marital Status",
                sorted(df["marital"].unique())
            )
    
            education = st.selectbox(
                "Education",
                sorted(df["education"].unique())
            )
    
        with col3:
    
            default = st.selectbox(
                "Credit in Default",
                sorted(df["default"].unique())
            )
    
            housing = st.selectbox(
                "Housing Loan",
                sorted(df["housing"].unique())
            )
    
            loan = st.selectbox(
                "Personal Loan",
                sorted(df["loan"].unique())
            )
    
        st.divider()
    
        # ---------------------------------------------
        # CAMPAIGN INFORMATION
        # ---------------------------------------------
    
        st.markdown(
            '<div class="section-title">'
            '📞 Campaign Information'
            '</div>',
            unsafe_allow_html=True
        )
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
    
            contact = st.selectbox(
                "Contact Method",
                sorted(df["contact"].unique())
            )
    
            month = st.selectbox(
                "Contact Month",
                [
                    "jan", "feb", "mar", "apr",
                    "may", "jun", "jul", "aug",
                    "sep", "oct", "nov", "dec"
                ]
            )
    
        with col2:
    
            day_of_week = st.selectbox(
                "Day of Week",
                [
                    "mon",
                    "tue",
                    "wed",
                    "thu",
                    "fri"
                ]
            )
    
            duration = st.number_input(
                "Call Duration (seconds)",
                min_value=0,
                value=300
            )
    
        with col3:
    
            campaign = st.number_input(
                "Contacts During Campaign",
                min_value=1,
                value=1
            )
    
            previous = st.number_input(
                "Previous Contacts",
                min_value=0,
                value=0
            )
    
            pdays = st.number_input(
                "Days Since Previous Contact",
                min_value=0,
                value=999
            )
    
        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            sorted(df["poutcome"].unique())
        )
    
        st.divider()
    
        # ---------------------------------------------
        # ECONOMIC INFORMATION
        # ---------------------------------------------
    
        st.markdown(
            '<div class="section-title">'
            '📈 Economic Information'
            '</div>',
            unsafe_allow_html=True
        )
    
        col1, col2, col3, col4, col5 = st.columns(5)
    
        with col1:
    
            emp_var_rate = st.number_input(
                "Employment Variation Rate",
                value=0.0,
                step=0.1
            )
    
        with col2:
    
            cons_price_idx = st.number_input(
                "Consumer Price Index",
                value=93.0,
                step=0.1
            )
    
        with col3:
    
            cons_conf_idx = st.number_input(
                "Consumer Confidence Index",
                value=-40.0,
                step=0.1
            )
    
        with col4:
    
            euribor3m = st.number_input(
                "Euribor 3 Month Rate",
                value=3.0,
                step=0.1
            )
    
        with col5:
    
            nr_employed = st.number_input(
                "Number of Employees",
                value=5000.0,
                step=100.0
            )
    
        st.divider()
    
        # ---------------------------------------------
        # PREDICT BUTTON
        # ---------------------------------------------
    
        if st.button(
            "🔮 Predict Customer",
            use_container_width=True
        ):
    
            input_data = pd.DataFrame([{
    
                "age": age,
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "month": month,
                "day_of_week": day_of_week,
                "duration": duration,
                "campaign": campaign,
                "pdays": pdays,
                "previous": previous,
                "poutcome": poutcome,
                "emp.var.rate": emp_var_rate,
                "cons.price.idx": cons_price_idx,
                "cons.conf.idx": cons_conf_idx,
                "euribor3m": euribor3m,
                "nr.employed": nr_employed
    
            }])
    
            prediction = model.predict(
                input_data
            )[0]
    
            probabilities = model.predict_proba(
                input_data
            )[0]
    
            probability_dict = dict(
                zip(
                    model.classes_,
                    probabilities
                )
            )
    
            yes_probability = probability_dict["yes"]
    
            no_probability = probability_dict["no"]
    
            st.markdown(
                '<div class="section-title">'
                '📋 Prediction Result'
                '</div>',
                unsafe_allow_html=True
            )
    
            if prediction == "yes":
    
                st.success(
                    "### ✅ Likely to Subscribe\n"
                    f"Subscription probability: "
                    f"{yes_probability * 100:.2f}%"
                )
    
            else:
    
                st.error(
                    "### ❌ Unlikely to Subscribe\n"
                    f"Subscription probability: "
                    f"{yes_probability * 100:.2f}%"
                )
    
            col1, col2 = st.columns(2)
    
            with col1:
    
                st.metric(
                    "Subscription Probability",
                    f"{yes_probability * 100:.2f}%"
                )
    
                st.progress(
                    float(yes_probability)
                )
    
            with col2:
    
                st.metric(
                    "No Subscription Probability",
                    f"{no_probability * 100:.2f}%"
                )
    
                st.progress(
                    float(no_probability)
                )
    
    # ---------------------------------------------
            # AI PREDICTION EXPLANATION
            # ---------------------------------------------
    
            st.markdown(
                '<div class="section-title">'
                '🤖 AI Prediction Explanation'
                '</div>',
                unsafe_allow_html=True
            )
    
            if client is None:
    
                st.info(
                    "AI explanation is unavailable because "
                    "the OpenAI API key has not been configured."
                )
    
            else:
    
                if st.button(
                    "✨ Explain This Prediction with AI",
                    use_container_width=True
                ):
    
                    customer_data = input_data.iloc[0].to_dict()
    
                    with st.spinner(
                        "Generating AI explanation..."
                    ):
    
                        explanation = generate_prediction_explanation(
                            prediction,
                            yes_probability,
                            customer_data,
                            results["feature_importance"]
                        )
    
                    st.info(explanation)
    
                    
    # =========================================================
    # AI ASSISTANT
    # =========================================================
    
    elif page == "🤖 AI Assistant":
    
        st.markdown(
            '<div class="main-title">AI Marketing Assistant</div>',
            unsafe_allow_html=True
        )
    
        st.markdown(
            """
            <div class="subtitle">
            Ask questions about the Bank Marketing prediction system,
            machine learning results, and marketing insights.
            </div>
            """,
            unsafe_allow_html=True
        )
    
        if client is None:
    
            st.warning(
                "OpenAI API key is not configured."
            )
    
        else:
    
            question = st.text_area(
                "Ask the AI Assistant",
                placeholder=(
                    "Example: Which features are most important?"
                ),
                height=120
            )
    
            if st.button("Ask AI"):
                if question.strip():
    
                    feature_info = (
                        results["feature_importance"]
                        .head(10)
                        .to_string(index=False)
                    )
    
                    prompt = f"""
    You are the AI assistant for a Bank Marketing
    Prediction Machine Learning project.
    
    Answer the question clearly and simply.
    
    Model:
    Logistic Regression
    
    Accuracy:
    {results["accuracy"]:.4f}
    
    Precision:
    {results["precision"]:.4f}
    
    Recall:
    {results["recall"]:.4f}
    
    F1 Score:
    {results["f1"]:.4f}
    
    Top features:
    {feature_info}
    
    Question:
    {question}
    
    Do not invent model results.
    Use only the provided project information.
    """
    
                        with st.spinner("AI is thinking..."):
        
                            response = client.responses.create(
                                model="gpt-5.6-luna",
                                input=prompt
                            )
        
                        st.markdown("### 💬 AI Answer")
        
                        st.write(
                            response.output_text
                        )
                    else:
                            st.warning(
                                "Please enter a question."
                            )
    
    
    # =========================================================
    # DATA ANALYSIS
    # =========================================================
    
        elif page == "📊 Data Analysis":
            
             st.markdown(
                    '<div class="main-title">'
                    'Data Analysis'
                    '</div>',
                    unsafe_allow_html=True
                )
            
        st.markdown(
                    '<div class="subtitle">'
                    'Exploratory analysis of the Bank Marketing dataset.'
                    '</div>',
                    unsafe_allow_html=True
                )
            
        st.subheader(
                    "Dataset Overview"
                )
            
        col1, col2, col3, col4 = st.columns(4)
            
        with col1:
            
                    st.metric(
                        "Rows",
                        f"{df.shape[0]:,}"
                    )
            
        with col2:
            
                    st.metric(
                        "Input Features",
                        df.shape[1] - 1
                    )
            
        with col3:
            
                    numerical = len(
                        df.select_dtypes(
                            include=np.number
                        ).columns
                    )
            
                    st.metric(
                        "Numerical Features",
                        numerical
                    )
            
        with col4:
            
                    categorical = len(
                        df.select_dtypes(
                            include="object"
                        ).columns
                    ) - 1
            
                    st.metric(
                        "Categorical Features",
                        categorical
                    )
            
        st.divider()
            
        col1, col2 = st.columns(2)
            
        with col1:
            
                    st.subheader(
                        "Subscription Distribution"
                    )
            
                    st.bar_chart(
                        df["y"].value_counts()
                    )
            
        with col2:
            
                    st.subheader(
                        "Age Distribution"
                    )
            
                    st.bar_chart(
                        df["age"].value_counts().sort_index()
                    )
            
        st.divider()
            
        st.subheader(
                    "Subscription by Job"
                )
            
        job_data = pd.crosstab(
                    df["job"],
                    df["y"]
                )
            
        st.bar_chart(
                    job_data
                )
            
        st.divider()
            
        st.subheader(
                    "Dataset Sample"
                )
            
        st.dataframe(
                    df.head(20),
                    use_container_width=True,
                    hide_index=True
                )
    
    
    # =========================================================
    # MODEL PERFORMANCE
    # =========================================================
        elif page == "🤖 Model Performance":
        
            st.markdown(
                    '<div class="main-title">'
                    'Model Performance'
                    '</div>',
                    unsafe_allow_html=True
                )
            
            st.markdown(
                    '<div class="subtitle">'
                    'Results directly loaded from the machine learning notebook.'
                    '</div>',
                    unsafe_allow_html=True
                )
    
        # ---------------------------------------------
        # METRICS FROM NOTEBOOK
        # ---------------------------------------------
    
     col1, col2, col3, col4, col5 = st.columns(5)
    
     with col1:
    
            st.metric(
                "Accuracy",
                f"{results['accuracy']:.4f}"
            )
    
     with col2:
    
            st.metric(
                "Precision",
                f"{results['precision']:.4f}"
            )
    
     with col3:
    
            st.metric(
                "Recall",
                f"{results['recall']:.4f}"
            )
    
     with col4:
    
            st.metric(
                "F1 Score",
                f"{results['f1']:.4f}"
            )
        st.divider()
    
        # ---------------------------------------------
        # CONFUSION MATRIX
        # ---------------------------------------------
    
     col1, col2 = st.columns(2)
    
     with col1:
    
            st.subheader(
                "Confusion Matrix"
            )
    
            cm = results[
                "confusion_matrix"
            ]
    
            fig, ax = plt.subplots(
                figsize=(5, 4)
            )
    
            ax.imshow(cm)
    
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
    
            ax.set_xticklabels(
                ["Predicted No", "Predicted Yes"]
            )
    
            ax.set_yticklabels(
                ["Actual No", "Actual Yes"]
            )
    
            ax.set_xlabel(
                "Prediction"
            )
    
            ax.set_ylabel(
                "Actual"
            )
    
            ax.set_title(
                "Confusion Matrix"
            )
    
            for i in range(2):
    
                for j in range(2):
    
                    ax.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center"
                    )
    
            st.pyplot(fig)
    
    # =========================================================
    # FEATURE IMPORTANCE
    # =========================================================
    
    elif page == "⭐ Feature Importance":
    
    st.markdown(
            '<div class="main-title">'
            'Feature Importance'
            '</div>',
            unsafe_allow_html=True
        )
    
    st.markdown(
            '<div class="subtitle">'
            'Feature importance directly loaded from the notebook.'
            '</div>',
            unsafe_allow_html=True
        )
    
    importance_df = results[
            "feature_importance"
        ].copy()
    
        # ---------------------------------------------
        # TOP 15
        # ---------------------------------------------
    
    top_features = (
            importance_df
            .head(15)
            .sort_values(
                "Importance"
            )
        )
    
    st.subheader(
            "Top 15 Important Features"
        )
    
    chart_data = (
            top_features[
                ["Feature", "Importance"]
            ]
            .set_index("Feature")
        )
    
    st.bar_chart(
            chart_data
        )
    
    st.divider()
    
    st.subheader(
            "Feature Importance Details"
        )
    
    display_df = (
            importance_df
            .head(15)
            .copy()
        )
    
    display_df["Importance"] = (
            display_df["Importance"]
            .round(4)
        )
        display_df["Coefficient"] = (
    display_df["Coefficient"] = (
            display_df["Coefficient"]
            .round(4)
        )
        st.dataframe(
    st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    st.info(
            "The feature importance values shown here "
            "are the same values calculated by the "
            "Logistic Regression model in the notebook."
        )
    
    
    # =========================================================
    # ABOUT PROJECT
    # =========================================================
    
    elif page == "ℹ️ About Project":
    
    st.markdown(
            '<div class="main-title">'
            'About the Project'
            '</div>',
            unsafe_allow_html=True
        )
    
    st.markdown(
            '<div class="subtitle">'
            'Bank Marketing Classification System'
            '</div>',
            unsafe_allow_html=True
        )
    
    st.divider()
    
        # ---------------------------------------------
        # PROJECT INFORMATION
        # ---------------------------------------------
    
    col1, col2 = st.columns(2)
    
    with col1:
    
            st.subheader(
                "🎯 Problem Definition"
            )
    
            st.write(
                "The objective of this project is to predict "
                "whether a bank customer will subscribe to "
                "a term deposit after a marketing campaign."
            )
    
            st.subheader(
                "🤖 Machine Learning Model"
            )
    
            st.write(
                "Logistic Regression is used as the "
                "classification algorithm."
            )
    
            st.subheader(
                "📚 Learning Type"
            )
    
            st.write(
                "This is a supervised machine learning "
                "classification problem."
            )
    
    with col2:
    
            st.subheader(
                "⚙️ Preprocessing"
            )
    
            st.write(
                "Numerical features are standardized using "
                "StandardScaler. Categorical features are "
                "converted using OneHotEncoder."
            )
    
            st.subheader(
                "📊 Evaluation Metrics"
            )
    
            st.write(
                "The model is evaluated using Accuracy, "
                "Precision, Recall, F1 Score, Confusion "
                "Matrix and ROC-AUC."
            )
    
            st.subheader(
                "🚀 Deployment"
            )
    
            st.write(
                "The trained model and its results are saved "
                "using Joblib and displayed through a "
                "Streamlit application."
            )
    
    st.divider()
    
        # ---------------------------------------------
        # PROJECT STATISTICS
        # ---------------------------------------------
    
    st.subheader(
            "Project Statistics"
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
    
            st.metric(
                "Dataset Records",
                f"{len(df):,}"
                )
    
    with col2:
    
            st.metric(
                "Input Features",
                df.shape[1] - 1
            )
    
    with col3:
    
            st.metric(
                "Model",
                "Logistic \
                Regression"
            )
    
    with col4:
    
            st.metric(
                "Task",
                "Classification"
            )
    
    st.divider()
    
    st.success(
            "Bank Marketing Machine Learning Project"
        )
    
    st.caption(
            "Educational and demonstration application"
        )
