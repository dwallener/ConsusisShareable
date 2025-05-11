import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# --- App Title ---
st.title("Sleep Apnea Economic Impact Model (Age 40-60)")

# --- Sidebar Inputs ---
st.sidebar.header("Model Parameters")

population_size = st.sidebar.number_input("Population Size", 10_000, 1_000_000, 100_000, step=10_000)
osa_prevalence = st.sidebar.slider("OSA Prevalence (%)", 5, 50, 25) / 100
current_diagnosis_rate = st.sidebar.slider("Current Diagnosis Rate (%)", 0, 100, 20) / 100
target_diagnosis_rate = st.sidebar.slider("Target Diagnosis Rate (%)", 0, 100, 50) / 100
treatment_adherence = st.sidebar.slider("Treatment Adherence (%)", 0, 100, 60) / 100

cost_diagnosis = st.sidebar.number_input("Diagnosis Cost ($)", 100, 5000, 1000, step=100)
cost_treatment = st.sidebar.number_input("Annual Treatment Cost ($)", 500, 5000, 1500, step=100)
discount_rate = st.sidebar.slider("Discount Rate (%)", 0, 10, 3) / 100

years = np.arange(1, 21)

# --- Data Source Selection ---
data_source = st.sidebar.radio("Comorbidity Data Source", ["Manual Entry", "Load Monte Carlo JSON"])

if data_source == "Manual Entry":
    st.sidebar.subheader("Manual Comorbidity Prevalence (%)")
    comorbidity_prevalence = {
        'Hypertension': st.sidebar.slider("Hypertension (%)", 0, 100, 30) / 100,
        'Diabetes': st.sidebar.slider("Diabetes (%)", 0, 100, 15) / 100,
        'Cardiovascular Event': st.sidebar.slider("Cardiovascular Event (%)", 0, 100, 10) / 100,
        'Depression': st.sidebar.slider("Depression (%)", 0, 100, 25) / 100
    }

    st.sidebar.subheader("Manual Joint Probabilities (%)")
    joint_prevalence = {
        'Hypertension + Diabetes': st.sidebar.slider("Hypertension + Diabetes (%)", 0, 100, 12) / 100,
        'Hypertension + Cardio': st.sidebar.slider("Hypertension + Cardio (%)", 0, 100, 8) / 100,
        'Diabetes + Cardio': st.sidebar.slider("Diabetes + Cardio (%)", 0, 100, 5) / 100,
        'All Three': st.sidebar.slider("Hypertension + Diabetes + Cardio (%)", 0, 100, 3) / 100
    }
else:
    uploaded_file = st.sidebar.file_uploader("Upload Monte Carlo JSON", type="json")
    if uploaded_file:
        mc_data = json.load(uploaded_file)
        prevalence_raw = mc_data.get('joint_prevalence', {})
        
        # Extract single-condition prevalences safely
        comorbidity_prevalence = {
            'Hypertension': prevalence_raw.get('hypertension', 0),
            'Diabetes': prevalence_raw.get('diabetes', 0),
            'Cardiovascular Event': prevalence_raw.get('cardio', 0),
            'Depression': 0.25  # Assume static if not modeled in Monte Carlo
        }

        joint_prevalence = {
            'Hypertension + Diabetes': prevalence_raw.get('hypertension_diabetes', 0),
            'Hypertension + Cardio': prevalence_raw.get('hypertension_cardio', 0),
            'Diabetes + Cardio': prevalence_raw.get('diabetes_cardio', 0),
            'All Three': prevalence_raw.get('hypertension_diabetes_cardio', 0)
        }
    else:
        st.warning("Please upload a Monte Carlo JSON file to proceed.")
        st.stop()

comorbidity_costs = {
    'Hypertension': 2000,
    'Diabetes': 12000,
    'Cardiovascular Event': 30000,
    'Depression': 5000
}

risk_reduction = {
    'Hypertension': 0.40,
    'Diabetes': 0.25,
    'Cardiovascular Event': 0.50,
    'Depression': 0.30
}

def discounted_value(value, year, rate=discount_rate):
    return value / ((1 + rate) ** year)

def calculate_savings():
    untreated_osa = population_size * osa_prevalence * (1 - current_diagnosis_rate)
    improved_treated_osa = population_size * osa_prevalence * (target_diagnosis_rate - current_diagnosis_rate) * treatment_adherence

    total_savings = 0
    annual_data = []

    for year in years:
        yearly_savings = 0

        for condition in comorbidity_costs:
            prevalence = comorbidity_prevalence.get(condition, 0)
            baseline_cases = untreated_osa * prevalence
            reduced_cases = improved_treated_osa * prevalence * risk_reduction[condition]
            cost_avoided = reduced_cases * comorbidity_costs[condition]
            yearly_savings += discounted_value(cost_avoided, year)

        # Joint comorbidity adjustment (if provided)
        for joint_key, joint_val in joint_prevalence.items():
            cost_estimate = 0
            if "Hypertension" in joint_key:
                cost_estimate += comorbidity_costs['Hypertension']
            if "Diabetes" in joint_key:
                cost_estimate += comorbidity_costs['Diabetes']
            if "Cardio" in joint_key:
                cost_estimate += comorbidity_costs['Cardiovascular Event']
            cost_avoided = improved_treated_osa * joint_val * cost_estimate * np.mean(list(risk_reduction.values()))
            yearly_savings += discounted_value(cost_avoided, year)

        treatment_cost = improved_treated_osa * (cost_treatment + cost_diagnosis if year == 1 else cost_treatment)
        net_savings = yearly_savings - discounted_value(treatment_cost, year)
        
        total_savings += net_savings
        annual_data.append({
            'Year': year,
            'Cost Avoided': yearly_savings,
            'Treatment Cost': discounted_value(treatment_cost, year),
            'Net Savings': net_savings
        })

    df = pd.DataFrame(annual_data)
    return total_savings, df

# --- Run Model ---
total_savings, results_df = calculate_savings()

# --- Results Display ---
st.subheader("Total Net Present Value (NPV) Savings")
st.success(f"${total_savings:,.2f} over 20 years")
st.info(f"**Results are based on a modeled population of {population_size:,} members aged 40–60.**")

st.subheader("Annual Breakdown")
st.dataframe(results_df.style.format({"Cost Avoided": "${:,.0f}", "Treatment Cost": "${:,.0f}", "Net Savings": "${:,.0f}"}))

# --- Plot ---
st.subheader("Net Savings Over Time")
fig, ax = plt.subplots()
ax.plot(results_df['Year'], results_df['Net Savings'].cumsum(), marker='o')
ax.set_xlabel("Year")
ax.set_ylabel("Cumulative Net Savings ($)")
ax.grid(True)
st.pyplot(fig)

