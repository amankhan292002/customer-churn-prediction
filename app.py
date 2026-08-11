import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD MODEL AND PREPROCESSOR
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("final_random_forest.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor


final_model, preprocessor = load_model()


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("Customer Churn Prediction")

st.write(
    "Enter customer details below to predict whether "
    "the customer is likely to churn."
)


# ---------------------------------------------------------
# CUSTOMER INFORMATION
# ---------------------------------------------------------

st.header("Customer Information")


col1, col2 = st.columns(2)


with col1:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Does the customer have online security?",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Does the customer have online backup?",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Does the customer have device protection?",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Does the customer have technical support?",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


with col2:

    gender = st.selectbox(
        "What is the customer's gender?",
        [
            "Female",
            "Male"
        ]
    )

    senior_citizen = st.selectbox(
        "Is the customer a senior citizen?",
        [
            "No",
            "Yes"
        ]
    )

    partner = st.selectbox(
        "Does the customer have a partner?",
        [
            "No",
            "Yes"
        ]
    )

    dependents = st.selectbox(
        "Does the customer have dependents?",
        [
            "No",
            "Yes"
        ]
    )

    phone_service = st.selectbox(
        "Does the customer have phone service?",
        [
            "No",
            "Yes"
        ]
    )

    multiple_lines = st.selectbox(
        "Does the customer have multiple lines?",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    streaming_tv = st.selectbox(
        "Does the customer use streaming TV?",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Does the customer use streaming movies?",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    paperless_billing = st.selectbox(
        "Does the customer use paperless billing?",
        [
            "No",
            "Yes"
        ]
    )

    payment_method = st.selectbox(
        "How does the customer pay?",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# ---------------------------------------------------------
# ENGINEERED FEATURES
# ---------------------------------------------------------

if tenure <= 12:
    tenure_group = "New"
elif tenure <= 24:
    tenure_group = "Loyal"
else:
    tenure_group = "Long-term"


high_charges = 1 if monthly_charges > 70 else 0


# Count services
service_values = [
    phone_service,
    multiple_lines,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies
]

total_services = sum(
    1 for value in service_values
    if value == "Yes"
)


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if predict_button:

    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------------------------------

    input_data = pd.DataFrame({
        "gender": [gender],

        # SeniorCitizen is normally numeric in the dataset
        "SeniorCitizen": [
            1 if senior_citizen == "Yes" else 0
        ],

        "Partner": [partner],

        "Dependents": [dependents],

        "tenure": [tenure],

        "PhoneService": [phone_service],

        "MultipleLines": [multiple_lines],

        "InternetService": [internet_service],

        "OnlineSecurity": [online_security],

        "OnlineBackup": [online_backup],

        "DeviceProtection": [device_protection],

        "TechSupport": [tech_support],

        "StreamingTV": [streaming_tv],

        "StreamingMovies": [streaming_movies],

        "Contract": [contract],

        "PaperlessBilling": [paperless_billing],

        "PaymentMethod": [payment_method],

        "MonthlyCharges": [monthly_charges],

        "TotalCharges": [total_charges],

        "tenure_group": [tenure_group],

        "high_charges": [high_charges],

        "total_services": [total_services]
    })


    # -----------------------------------------------------
    # MAKE SURE COLUMNS ARE IN THE SAME ORDER
    # -----------------------------------------------------

    expected_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "tenure_group",
        "high_charges",
        "total_services"
    ]

    input_data = input_data[expected_columns]


    # -----------------------------------------------------
    # FIX NUMERIC COLUMNS ACCORDING TO PREPROCESSOR
    # -----------------------------------------------------

    # Find columns used by numeric transformers
    numeric_columns = []

    for transformer_name, transformer, columns in preprocessor.transformers_:

        if transformer_name == "num":

            if isinstance(columns, (list, tuple)):
                numeric_columns.extend(columns)

            else:
                try:
                    numeric_columns.extend(list(columns))
                except TypeError:
                    pass


    # Convert numeric columns safely
    for column in numeric_columns:

        if column in input_data.columns:

            # Handle Yes/No if any numeric column receives them
            if input_data[column].dtype == "object":

                input_data[column] = input_data[column].replace({
                    "Yes": 1,
                    "No": 0
                })

            input_data[column] = pd.to_numeric(
                input_data[column],
                errors="coerce"
            )


    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    try:

        input_encoded = preprocessor.transform(input_data)

    except Exception as e:

        st.error(
            "There was an error while preprocessing the input data."
        )

        st.exception(e)

        st.stop()


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    try:

        prediction = final_model.predict(input_encoded)[0]

    except Exception as e:

        st.error(
            "There was an error while making the prediction."
        )

        st.exception(e)

        st.stop()


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.divider()

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "Customer is likely to churn."
        )

    else:

        st.success(
            "Customer is unlikely to churn."
        )


