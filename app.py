import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

MODEL_FILE = 'employee_model.pkl'
DATA_FILE = 'employees.csv'
ENCODED_FILE = 'employees_encoded.csv'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

@st.cache_data
def load_data():
    if os.path.exists(ENCODED_FILE):
        return pd.read_csv(ENCODED_FILE), pd.read_csv(DATA_FILE)
    return None, None

def get_risk_color(probability):
    if probability > 0.7:
        return "🔴 High Risk"
    elif probability > 0.4:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

def get_status_text(probability):
    if probability > 0.5:
        return "Likely to Leave"
    else:
        return "Likely to Stay"

def main():
    st.title("👥 Employee Attrition Prediction Dashboard")
    st.markdown("---")
    
    model_data = load_model()
    df_encoded, df_original = load_data()
    
    if model_data is None:
        st.error("Model not found! Please run `train.py` first to train the model.")
        return
    
    model, feature_cols, label_encoders = model_data
    
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df_encoded)
    attrition_count = int(df_encoded['Attrition'].sum())
    attrition_rate = attrition_count / total * 100
    active_count = total - attrition_count
    
    with col1:
        st.metric("Total Employees", total)
    
    with col2:
        st.metric("Attrition Count", attrition_count)
    
    with col3:
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    
    with col4:
        st.metric("Active Employees", active_count)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📋 Employee List", "🔍 Search & Predict", "📊 Analytics"])
    
    with tab1:
        st.subheader("All Employees with Predicted Churn Risk")
        
        df_display = df_encoded.copy()
        df_display['EmployeeID'] = range(1, len(df_display) + 1)
        
        predictions = model.predict_proba(df_display[feature_cols])[:, 1]
        df_display['Churn_Probability'] = predictions
        df_display['Risk_Level'] = df_display['Churn_Probability'].apply(
            lambda x: "High" if x > 0.7 else "Medium" if x > 0.4 else "Low"
        )
        df_display['Status'] = df_display['Churn_Probability'].apply(get_status_text)
        
        decoders = {col: {i: label for i, label in enumerate(le.classes_)} 
                   for col, le in label_encoders.items()}
        
        for col in ['Department', 'JobRole', 'EducationField', 'BusinessTravel', 
                    'MaritalStatus', 'Gender', 'OverTime']:
            if col in df_display.columns and col in decoders:
                df_display[col] = df_display[col].map(decoders[col])
        
        col_filter = st.columns(1)[0]
        with col_filter:
            filter_option = st.selectbox(
                "Filter by Risk Level:",
                ["All", "High Risk (>70%)", "Medium Risk (40-70%)", "Low Risk (<40%)"]
            )
        
        df_filtered = df_display.copy()
        if filter_option == "High Risk (>70%)":
            df_filtered = df_filtered[df_filtered['Churn_Probability'] > 0.7]
        elif filter_option == "Medium Risk (40-70%)":
            df_filtered = df_filtered[(df_filtered['Churn_Probability'] > 0.4) & 
                                      (df_filtered['Churn_Probability'] <= 0.7)]
        elif filter_option == "Low Risk (<40%)":
            df_filtered = df_filtered[df_filtered['Churn_Probability'] <= 0.4]
        
        display_cols = ['EmployeeID', 'Age', 'Department', 'JobRole', 'MonthlyIncome', 
                       'YearsAtCompany', 'Churn_Probability', 'Risk_Level', 'Status']
        available_cols = [c for c in display_cols if c in df_filtered.columns]
        
        st.dataframe(
            df_filtered[available_cols].sort_values('Churn_Probability', ascending=False),
            use_container_width=True,
            height=500
        )
        
        st.info(f"Showing {len(df_filtered)} of {len(df_display)} employees")
        
        csv = df_display[available_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download Full Report",
            data=csv,
            file_name="employee_attrition_report.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("🔍 Search & Predict for Individual Employee")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            search_id = st.number_input("Employee ID", min_value=1, max_value=len(df_encoded), value=1, step=1)
        
        if 1 <= search_id <= len(df_encoded):
            emp_encoded = df_encoded.iloc[search_id - 1]
            emp_original = df_original.iloc[search_id - 1] if df_original is not None else emp_encoded
            
            churn_prob = model.predict_proba(emp_encoded[feature_cols].values.reshape(1, -1))[0][1]
            
            with col2:
                st.markdown(f"### {get_risk_color(churn_prob)} - Churn Probability: {churn_prob:.1%}")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("#### Employee Details")
                for col in ['Age', 'Department', 'JobRole', 'MaritalStatus', 'Gender', 'OverTime', 'Education', 'EducationField']:
                    val = emp_original.get(col, 'N/A')
                    if col in label_encoders and isinstance(val, (int, float)):
                        val = label_encoders[col].classes_[int(val)]
                    st.write(f"**{col}:** {val}")
                
                income = emp_original.get('MonthlyIncome', 0)
                st.write(f"**Monthly Income:** ${income:,}")
            
            with col_right:
                st.markdown("#### Work History")
                for col in ['YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                           'TotalWorkingYears', 'NumCompaniesWorked', 'DistanceFromHome',
                           'BusinessTravel', 'WorkLifeBalance']:
                    val = emp_original.get(col, 0)
                    if col in label_encoders and isinstance(val, (int, float)):
                        val = label_encoders[col].classes_[int(val)]
                    st.write(f"**{col}:** {val}")
            
            st.markdown("---")
            st.markdown("#### Top Risk Factors")
            
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)
            
            for idx, row in importance_df.iterrows():
                feature = row['Feature']
                value = emp_encoded.get(feature, 0)
                if feature in label_encoders:
                    value = label_encoders[feature].classes_[int(value)]
                
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(row['Importance'] / importance_df['Importance'].max())
                    st.write(f"**{feature}:** {value}")
                with col_b:
                    st.write(f"Impact: {int(row['Importance'])}")
    
    with tab3:
        st.subheader("📊 Attrition Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Department-wise Attrition")
            dept_data = df_original.groupby('Department')['Attrition'].apply(
                lambda x: (x == 'Yes').sum() if x.dtype == 'object' else int(x.sum())
            ).sort_values(ascending=False)
            
            chart_data = pd.DataFrame({
                'Department': dept_data.index,
                'Attrition Count': dept_data.values
            })
            st.bar_chart(chart_data.set_index('Department'))
        
        with col2:
            st.markdown("#### Job Role Attrition")
            role_data = df_original.groupby('JobRole')['Attrition'].apply(
                lambda x: (x == 'Yes').sum() if x.dtype == 'object' else int(x.sum())
            ).sort_values(ascending=False)
            
            chart_data2 = pd.DataFrame({
                'Job Role': role_data.index,
                'Attrition Count': role_data.values
            })
            st.bar_chart(chart_data2.set_index('Job Role'))
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Age Distribution by Attrition")
            age_attrition = df_encoded.groupby('Age')['Attrition'].sum().sort_index()
            st.line_chart(age_attrition)
        
        with col4:
            st.markdown("#### Overtime Impact on Attrition")
            ot_data = df_encoded.groupby('OverTime')['Attrition'].mean() * 100
            ot_labels = ['No Overtime', 'With Overtime'] if len(ot_data) == 2 else ot_data.index.tolist()
            ot_df = pd.DataFrame({'Overtime': ot_labels[:len(ot_data)], 'Attrition Rate %': ot_data.values})
            st.bar_chart(ot_df.set_index('Overtime'))

if __name__ == "__main__":
    main()