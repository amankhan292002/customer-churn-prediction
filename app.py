import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("final_random_forest.pkl")
    preprocessor = joblib.load("preprocessor.pkl")

    return model, preprocessor


final_model, preprocessor = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("Customer Churn Prediction")

st.write(
    "Enter customer details below to predict whether "
    "the customer is likely to churn."
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.subheader("Customer Information")


# -----------------------------
# LEFT COLUMN
# -----------------------------

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


# -----------------------------
# RIGHT COLUMN
# -----------------------------

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


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ====================================================
        # FEATURE ENGINEERING
        # ====================================================

        # Tenure group
        if tenure <= 12:
            tenure_group = "0-12"
        elif tenure <= 24:
            tenure_group = "13-24"
        elif tenure <= 48:
            tenure_group = "25-48"
        elif tenure <= 60:
            tenure_group = "49-60"
        else:
            tenure_group = "61+"


        # High charges
        if monthly_charges > 70:
            high_charges = 1
        else:
            high_charges = 0


        # Count subscribed services
        service_values = [
            phone_service,
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


        # ====================================================
        # CREATE INPUT DATAFRAME
        # ====================================================
        senior_citizen_value = 1 if senior_citizen == "Yes" else 0

        input_data = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [senior_citizen_value],

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


        # ====================================================
        # ENSURE CORRECT NUMERIC TYPES
        # ====================================================

        numeric_columns = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "high_charges",
            "total_services"
        ]

        for column in numeric_columns:

            input_data[column] = pd.to_numeric(
                input_data[column],
                errors="coerce"
            )


        # ====================================================
        # CHECK FOR MISSING VALUES
        # ====================================================

        if input_data.isnull().any().any():

            st.error(
                "Some numeric values could not be converted. "
                "Please check the input values."
            )

            st.stop()


        # ====================================================
        # PREPROCESS INPUT
        # ====================================================

        input_encoded = preprocessor.transform(input_data)


        # ====================================================
        # FORCE NUMERIC OUTPUT
        # ====================================================

        input_encoded = np.asarray(
            input_encoded,
            dtype=np.float64
        )


        # ====================================================
        # SHOW ENCODED DATA
        # ====================================================

        with st.expander("View Processed Input"):

            st.write(
                "Encoded shape:",
                input_encoded.shape
            )

            st.dataframe(
                pd.DataFrame(input_encoded)
            )


        # ====================================================
        # MAKE PREDICTION
        # ====================================================

        prediction = final_model.predict(input_encoded)[0]


        # ====================================================
        # PREDICTION PROBABILITY
        # ====================================================

        churn_probability = None

        if hasattr(final_model, "predict_proba"):

            probabilities = final_model.predict_proba(
                input_encoded
            )[0]

            # Probability of class 1
            if len(probabilities) > 1:
                churn_probability = probabilities[1]


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader("Prediction Result")


        if prediction == 1:

            st.error(
                "⚠️ Customer is likely to churn."
            )

        else:

            st.success(
                "✅ Customer is unlikely to churn."
            )


        # ====================================================
        # PROBABILITY
        # ====================================================

        if churn_probability is not None:

            st.write(
                f"**Churn Probability: "
                f"{churn_probability * 100:.2f}%**"
            )

            st.progress(
                float(churn_probability)
            )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.divider()

        st.subheader("SHAP Explanation")

        try:

            # Create SHAP explainer
            explainer = shap.TreeExplainer(
                final_model
            )

            shap_values = explainer.shap_values(
                input_encoded
            )


            # Get feature names
            try:

                feature_names = (
                    preprocessor
                    .get_feature_names_out()
                )

            except Exception:

                feature_names = [
                    f"Feature {i}"
                    for i in range(
                        input_encoded.shape[1]
                    )
                ]


            # =================================================
            # HANDLE DIFFERENT SHAP OUTPUT FORMATS
            # =================================================

            if isinstance(shap_values, list):

                # Binary classification:
                # class 1 = churn
                if len(shap_values) > 1:

                    values = shap_values[1][0]

                else:

                    values = shap_values[0][0]

            else:

                shap_array = np.asarray(
                    shap_values
                )

                if shap_array.ndim == 3:

                    # Shape:
                    # (samples, features, classes)

                    if shap_array.shape[2] > 1:

                        values = shap_array[
                            0, :, 1
                        ]

                    else:

                        values = shap_array[
                            0, :, 0
                        ]

                elif shap_array.ndim == 2:

                    values = shap_array[0]

                else:

                    values = shap_array


            # =================================================
            # CREATE SHAP DATAFRAME
            # =================================================

            shap_df = pd.DataFrame({

                "Feature": feature_names,

                "SHAP Value": values

            })


            # Make sure lengths match
            min_length = min(
                len(shap_df),
                len(values)
            )

            shap_df = shap_df.iloc[
                :min_length
            ]


            # Absolute importance
            shap_df["Importance"] = (
                shap_df["SHAP Value"]
                .abs()
            )


            # Top 10 features
            shap_df = shap_df.sort_values(
                "Importance",
                ascending=False
            ).head(10)

            
            # =================================================
            # DISPLAY SHAP TABLE
            # =================================================

            st.write(
                "Top factors influencing this prediction:"
            )

            st.dataframe(
                shap_df[
                    [
                        "Feature",
                        "SHAP Value"
                    ]
                ],
                use_container_width=True
            )


            # =================================================
            # SHAP BAR CHART
            # =================================================

            chart_df = shap_df.sort_values(
                "SHAP Value"
            )


            fig, ax = plt.subplots(
        figsize=(8, 4),
        constrained_layout=True
    )

            ax.barh(
                chart_df["Feature"],
                chart_df["SHAP Value"]
            )

            ax.axvline(
                0,
                linewidth=1
            )

            ax.set_xlabel("SHAP Value", fontsize=10)
        ax.set_ylabel("Feature", fontsize=10)
    
            ax.set_title(
        "Top Features Influencing Churn Prediction",
        fontsize=12
    )
    
            ax.tick_params(axis="both", labelsize=8)
    

            st.pyplot(fig)

            plt.close(fig)


            st.caption(
                "Positive SHAP values push the prediction "
                "towards churn, while negative values push "
                "the prediction towards non-churn."
            )


        except Exception as shap_error:

            # SHAP failure should NOT break prediction

            st.warning(
                "SHAP explanation could not be generated."
            )

            with st.expander(
                "SHAP Technical Details"
            ):

                st.write(shap_error)


    # ========================================================
    # GENERAL ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "There was an error while making the prediction."
        )

        st.exception(e)
