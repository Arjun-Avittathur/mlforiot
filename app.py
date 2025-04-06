import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from model import calculate_student_metrics, identify_strengths_weaknesses, calculate_average_performance, evaluate_model, generate_specific_recommendations
from utils import visualize_student_performance, visualize_student_vs_average
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Get API key from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
def main():
    st.set_page_config(
        page_title="Student Performance Analysis",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("Student Performance Analysis and Recommendations")
    
    upload_mode = st.radio("Select upload mode:", ["Existing database", "New student data"])

    if upload_mode == "Existing database":
        try:
            data = pd.read_csv('data/student_data.csv')
            st.success("Using existing student database")
        except FileNotFoundError:
            st.error("Database file not found. Please upload a CSV file.")
            data = None
    else:
        uploaded_file = st.file_uploader("Upload new student data", type=["csv"])
        if uploaded_file is not None:
            new_student_data = pd.read_csv(uploaded_file)
            
            try:
                # Save uploaded data
                os.makedirs('data', exist_ok=True)
                new_student_data.to_csv('data/student_data.csv', index=False)
                data = new_student_data
                st.success(f"Created new database with {len(new_student_data['student_id'].unique())} students")
            except Exception as e:
                st.error(f"Error saving data: {e}")
                data = new_student_data
        else:
            st.warning("Please upload student data")
            data = None
    
    if data is not None:
        # Student selector
        student_ids = data['student_id'].unique()
        selected_student = st.selectbox("Select a student to analyze:", student_ids)
        
        # Display raw data for selected student
        if st.checkbox("Show raw data for selected student"):
            st.subheader("Raw Data Preview")
            st.dataframe(data[data['student_id'] == selected_student], use_container_width=True)
        
        # Process data
        student_section_performance, student_overall = calculate_student_metrics(data)
        
        # Calculate average performance
        avg_section_performance, avg_overall_performance = calculate_average_performance(student_section_performance, student_overall)
        
        # Display class averages
        st.subheader("Class Performance Averages")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Overall Class Average", f"{avg_overall_performance:.2f}%")
        
        with col2:
            # Display section averages
            section_mapping = {'A': 'Math', 'B': 'Verbal', 'C': 'Non-verbal', 'D': 'Comprehension'}
            section_avg_data = pd.DataFrame({
                'Section': [f"{section_mapping[row['section']]} ({row['section']})" for _, row in avg_section_performance.iterrows()],
                'Average Score': [f"{row['avg_score_percentage']:.2f}%" for _, row in avg_section_performance.iterrows()]
            })
            st.dataframe(section_avg_data, hide_index=True, use_container_width=True)
        
        # Model evaluation
        with st.expander("Model Evaluation Metrics"):
            accuracy, precision = evaluate_model(data)
            st.write(f"Model Accuracy: {accuracy:.2f}%")
            st.write(f"Model Precision: {precision:.2f}%")
            st.write("Model accuracy measures how often our prediction model correctly identifies whether a student is performing above or below average.")
        
        # Visualize overall performance
        st.subheader("Overall Performance Analysis")
        fig = visualize_student_performance(student_section_performance, student_overall, 
                                           avg_section_performance, avg_overall_performance, selected_student)
        st.pyplot(fig)
        
        # Identify strengths and weaknesses
        strengths_weaknesses = identify_strengths_weaknesses(student_section_performance, avg_section_performance)
        
        # Individual student analysis
        st.subheader(f"Detailed Analysis for Student ID: {selected_student}")
        
        # Performance metrics
        student_data = student_section_performance[student_section_performance['student_id'] == selected_student]
        
        # Section performance
        st.markdown("### Section Performance:")
        section_data = student_data[['section', 'sum', 'count', 'score_percentage']]
        section_data.columns = ['Section', 'Correct Answers', 'Total Questions', 'Score (%)']
        
        # Map section codes to subject names
        section_mapping = {'A': 'Math', 'B': 'Verbal', 'C': 'Non-verbal', 'D': 'Comprehension'}
        section_data['Subject'] = section_data['Section'].map(section_mapping)
        
        # Add the class average for comparison
        section_data['Class Average (%)'] = section_data['Section'].apply(
            lambda x: avg_section_performance[avg_section_performance['section'] == x]['avg_score_percentage'].values[0]
        )
        
        # Add difference from average
        section_data['Difference from Average'] = section_data['Score (%)'] - section_data['Class Average (%)']
        
        # Reorder columns for better presentation
        section_data = section_data[['Subject', 'Section', 'Correct Answers', 'Total Questions', 
                                    'Score (%)', 'Class Average (%)', 'Difference from Average']]
        
        st.dataframe(section_data, hide_index=True, use_container_width=True)
        
        # Overall score comparison
        overall = student_overall[student_overall['student_id'] == selected_student]['overall_score'].values[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Student's Overall Score", f"{overall:.2f}%")
        with col2:
            st.metric("Class Average", f"{avg_overall_performance:.2f}%")
        with col3:
            difference = overall - avg_overall_performance
            st.metric("Difference from Average", f"{difference:.2f}%", 
                      delta=f"{difference:.2f}%", delta_color="normal")
        
        # Strengths and weaknesses
        st.markdown("### Performance Summary:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Strengths:**", ', '.join(f"{section_mapping[s]} (Section {s})" for s in strengths_weaknesses[selected_student]['strengths']))
        with col2:
            st.write("**Areas for Improvement:**", ', '.join(f"{section_mapping[w]} (Section {w})" for w in strengths_weaknesses[selected_student]['weaknesses']))
        
        # Generate highly specific recommendations
        with st.spinner("Generating personalized recommendations..."):
            specific_recommendations = generate_specific_recommendations(data, selected_student, student_section_performance, 
                                                                    avg_section_performance, OPENROUTER_API_KEY)
        
        # Display personalized recommendations
        st.markdown("### Personalized Recommendations:")
        
        for section, section_name in section_mapping.items():
            with st.expander(f"{section_name} (Section {section}):", expanded=(section in strengths_weaknesses[selected_student]['weaknesses'])):
                if section in specific_recommendations:
                    for recommendation in specific_recommendations[section]:
                        st.markdown(f"• {recommendation}")
                else:
                    st.write("No specific recommendations available for this section.")
        
        # Comparison with average
        st.subheader("Comparison with Average Performance")
        fig_comp = visualize_student_vs_average(student_data, avg_section_performance, section_mapping)
        st.pyplot(fig_comp)

if __name__ == "__main__":
    main()
