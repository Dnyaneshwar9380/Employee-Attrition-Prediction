import streamlit as st
import pickle
import pandas as pd
from catboost import CatBoostClassifier

# Load model and unique values
with open('model_and_key_components.pkl', 'rb') as file:
    saved_components = pickle.load(file)

model = saved_components['model']
unique_values = saved_components['unique_values']

# Streamlit App
def main():

    st.title("Employee Attrition Prediction App")
    st.sidebar.title("Model Settings")

    # Sidebar
    with st.sidebar.expander("View Unique Values"):
        st.write("Unique values for each feature:")
        for column, values in unique_values.items():
            st.write(f"{column}: {values}")

    st.write("This app predicts employee attrition using a trained CatBoost model.")

    # User Inputs
    age = st.slider("Age", 18, 70, 30)

    department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )

    environment_satisfaction = st.slider(
        "Environment Satisfaction",
        1, 4, 2
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1, 4, 2
    )

    monthly_income = st.slider(
        "Monthly Income",
        1000, 20000, 5000
    )

    num_companies_worked = st.slider(
        "Number of Companies Worked",
        0, 10, 2
    )

    over_time = st.selectbox(
        "Over Time",
        [True, False]
    )

    percent_salary_hike = st.slider(
        "Percent Salary Hike",
        10, 25, 15
    )

    relationship_satisfaction = st.slider(
        "Relationship Satisfaction",
        1, 4, 2
    )

    training_times_last_year = st.slider(
        "Training Times Last Year",
        0, 6, 2
    )

    work_life_balance = st.slider(
        "Work Life Balance",
        1, 4, 2
    )

    years_since_last_promotion = st.slider(
        "Years Since Last Promotion",
        0, 15, 3
    )

    years_with_curr_manager = st.slider(
        "Years With Current Manager",
        0, 15, 3
    )

    # Create DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Department': [department],
        'EnvironmentSatisfaction': [environment_satisfaction],
        'JobRole': [job_role],
        'JobSatisfaction': [job_satisfaction],
        'MonthlyIncome': [monthly_income],
        'NumCompaniesWorked': [num_companies_worked],
        'OverTime': [over_time],
        'PercentSalaryHike': [percent_salary_hike],
        'RelationshipSatisfaction': [relationship_satisfaction],
        'TrainingTimesLastYear': [training_times_last_year],
        'WorkLifeBalance': [work_life_balance],
        'YearsSinceLastPromotion': [years_since_last_promotion],
        'YearsWithCurrManager': [years_with_curr_manager]
    })

    # Predict Button
    if st.button("Predict Attrition"):

        try:
            # Prediction
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[:, 1]

            # Result
            if prediction[0] == 0:
                st.success("Employee is predicted to stay (Attrition = No)")
            else:
                st.error("Employee is predicted to leave (Attrition = Yes)")

                # Suggestions
                st.subheader("Suggestions for retaining the employee:")
                st.markdown("- Invest in employee training and development.")
                st.markdown("- Improve work-life balance policies.")
                st.markdown("- Provide career growth opportunities.")
                st.markdown("- Introduce mentorship programs.")
                st.markdown("- Improve employee engagement activities.")
                st.markdown("- Offer better employee benefits.")
                st.markdown("- Focus on employee satisfaction.")

            # Probability
            st.write(f"Probability of Attrition: {probability[0]:.2f}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# Run App
if __name__ == "__main__":
    main()
