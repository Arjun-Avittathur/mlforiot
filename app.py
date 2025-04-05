import streamlit as st
import os
# Add this at the beginning of your app.py file
os.makedirs('data', exist_ok=True)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from model import calculate_student_metrics, identify_strengths_weaknesses, generate_recommendations, calculate_average_performance, evaluate_model
from utils import visualize_student_performance, visualize_student_vs_average

def main():
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
                # Load existing data and append new student data
                existing_data = pd.read_csv('data/student_data.csv')
                
                # Check if student already exists
                new_student_id = new_student_data['student_id'].unique()[0]
                if new_student_id in existing_data['student_id'].unique():
                    # Replace existing student data
                    existing_data = existing_data[existing_data['student_id'] != new_student_id]
                
                # Append new student data
                data = pd.concat([existing_data, new_student_data], ignore_index=True)
                
                # Save updated data
                data.to_csv('data/student_data.csv', index=False)
                st.success(f"New student data added to database. Total students: {len(data['student_id'].unique())}")
            except FileNotFoundError:
                # Create new database with this student
                new_student_data.to_csv('data/student_data.csv', index=False)
                data = new_student_data
                st.success("Created new database with this student")
        else:
            st.warning("Please upload student data")
            data = None
    
    if data is not None:
        # Student selector (moved up to select student first)
        student_ids = data['student_id'].unique()
        selected_student = st.selectbox("Select a student to analyze:", student_ids)
        
        # Display raw data for the selected student only
        if st.checkbox("Show raw data for selected student"):
            st.subheader(f"Raw Data for Student ID: {selected_student}")
            student_raw_data = data[data['student_id'] == selected_student]
            st.dataframe(student_raw_data)
        
        # Process data
        student_section_performance, student_overall, learning_rates_df = calculate_student_metrics(data)
        
        # Calculate average performance
        avg_section_performance, avg_overall_performance = calculate_average_performance(student_section_performance, student_overall)
        
        # Display overall class average information
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
            st.dataframe(section_avg_data, hide_index=True)
        
        # Identify strengths and weaknesses
        strengths_weaknesses = identify_strengths_weaknesses(student_section_performance, avg_section_performance)
        
        # Generate comprehensive recommendations
        all_section_recommendations = generate_recommendations(strengths_weaknesses, learning_rates_df, 
                                                            student_section_performance, avg_section_performance, all_sections=True)
        
        # Model evaluation
        accuracy, precision, model_results = evaluate_model(data, learning_rates_df)
        
        with st.expander("Model Evaluation Metrics"):
            st.write(f"Model Accuracy: {accuracy:.2f}%")
            st.write(f"Model Precision: {precision:.2f}%")
            st.write("Model accuracy measures how often our prediction model correctly identifies whether a student is performing above or below average.")
            st.write("Model precision measures how often our predictions of above-average performance are correct.")
        
        # Visualize overall performance - only for selected student vs average
        st.subheader("Overall Performance Analysis")
        fig = visualize_student_performance(student_section_performance, student_overall, 
                                           avg_section_performance, avg_overall_performance, selected_student)
        st.pyplot(fig)
        
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
        section_data['Performance'] = section_data['Difference from Average'].apply(
            lambda x: "Above Average" if x > 5 else "Average" if abs(x) <= 5 else "Below Average"
        )
        
        # Reorder columns for better presentation
        section_data = section_data[['Subject', 'Section', 'Correct Answers', 'Total Questions', 
                                    'Score (%)', 'Class Average (%)', 'Difference from Average', 'Performance']]
        
        st.dataframe(section_data, hide_index=True)
        
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
        
        # Comprehensive recommendations for all sections
        st.markdown("### Personalized Recommendations:")
        
        for section, recommendations in all_section_recommendations[selected_student].items():
            subject_name = section_mapping.get(section, section)
            st.markdown(f"#### {subject_name} (Section {section}):")
            for rec in recommendations:
                st.write(f"- {rec}")
        
        # Comparison with average
        st.subheader("Comparison with Average Performance")
        fig_comp = visualize_student_vs_average(student_data, avg_section_performance, section_mapping)
        st.pyplot(fig_comp)

if __name__ == "__main__":
    main()
