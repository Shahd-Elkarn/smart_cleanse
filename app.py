import streamlit as st
import pandas as pd
from io import StringIO
from methods import *

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Data Quality Analysis & Chating")
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"], key='file_uploader')

    if uploaded_file is not None:
        if 'data' not in st.session_state:
            try:
                if uploaded_file.name.endswith(".csv"):
                    csv_file = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    df = pd.read_csv(csv_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)
                st.session_state['data'] = df
                st.sidebar.success("Dataset uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
        else:
            df = st.session_state['data'].copy()

        # Sidebar Menu
        st.sidebar.title("Analysis & Chat")
        section = st.sidebar.radio(
            "Choose From Here: ",
            (
                "Show Data",
                "Info Data",
                "Describe Data",
                "Column Name Analysis",
                "Missing Value Analysis",
                "Data Type Analysis",
                "Handle Duplicates",
                "Outlier Analysis",
                "Data Visualization",
                "Correlation Matrix",
                "Chat Using RAG"
            )
        )

        # Show Data
        if section == "Show Data":
            st.header("Data")
            st.write(df.head())


        elif section == "Info Data":
            st.header("Info Data")
            
            # Display the info table
            info_df = info_data(df)
            st.table(info_df)
            
            # Display the total number of rows below the table
            total_rows = len(df)
            st.write(f"**Total Rows:** {total_rows}")


        # Describe Data
        elif section == "Describe Data":
            st.header("Data Description")
            st.table(describe_data(df))


        # Data Type Analysis
        elif section == "Data Type Analysis":
            st.session_state['data_type_analysis_clicked'] = True

            st.header("Data Type Analysis")
            st.write("Original Data:")
            st.write(df)

            # Analyze data types
            data_types = df.dtypes
            st.write("### Data Types Before Conversion:")
            st.write(data_types)

            # Example: Select a column to change its type
            column_to_convert = st.selectbox("Select Column to Convert Data Type", df.columns)
            new_data_type = st.selectbox("Select New Data Type", ["int", "float", "str", "datetime"])

            # Flag to track if conversion was successful
            conversion_successful = False

            if st.button("Preview Conversion", key='preview_conversion'):
                try:
                    # Make a copy of the dataframe for preview
                    st.session_state['df_preview'] = df.copy()

                    if new_data_type == "int":
                        # Check if the column contains strings that can't be converted to int
                        if st.session_state['df_preview'][column_to_convert].apply(lambda x: isinstance(x, str)).any():
                            raise ValueError("Cannot convert from string to int. Please clean your data first.")
                        st.session_state['df_preview'][column_to_convert] = st.session_state['df_preview'][column_to_convert].astype(int)
                    
                    elif new_data_type == "float":
                        st.session_state['df_preview'][column_to_convert] = st.session_state['df_preview'][column_to_convert].astype(float)
                    
                    elif new_data_type == "str":
                        st.session_state['df_preview'][column_to_convert] = st.session_state['df_preview'][column_to_convert].astype(str)
                    
                    elif new_data_type == "datetime":
                        st.session_state['df_preview'][column_to_convert] = pd.to_datetime(st.session_state['df_preview'][column_to_convert])

                    # If no exception was raised, set the flag to True
                    conversion_successful = True

                    st.write("### Data After Type Conversion (Preview)")
                    st.write(st.session_state['df_preview'])
                    st.write("### Data Types After Conversion:")
                    st.write(st.session_state['df_preview'].dtypes)

                except ValueError as ve:
                    st.warning(f"Conversion Error: {ve}")
                    conversion_successful = False  # Reset the flag if a warning occurs
                except Exception as e:
                    st.warning(f"Error previewing conversion: {e}")
                    conversion_successful = False  # Reset the flag if a warning occurs

            # Only show the "OK" and "Cancel" buttons if conversion preview was successful
            if conversion_successful:
                if 'df_preview' in st.session_state:
                    if st.button("OK", key='confirm_conversion'):
                        df = st.session_state['df_preview']
                        st.session_state['data'] = df
                        st.session_state['type_converted'] = True
                        st.success(f"Column '{column_to_convert}' successfully converted to {new_data_type}.")
                    elif st.button("Cancel", key='cancel_conversion'):
                        st.warning("Conversion cancelled. Original data remains unchanged.")

            # Display final data and types only if conversion was successful
            if 'type_converted' in st.session_state and st.session_state['type_converted'] and conversion_successful:
                st.write("### Final Data After Type Conversion")
                st.write(df)
                st.write("### Updated Data Types:")
                st.write(df.dtypes)
                st.session_state['type_converted'] = False
                reset_all_flags()




        #... Column Name Analysis
        elif section == "Column Name Analysis":
            df = column_names_analysis(df)



        # Missing Value Analysis
        elif section == "Missing Value Analysis":
            reset_all_flags()
            st.header("Missing Value Analysis")
            st.write("Original Data:")
            st.write(df)

            # Display the number of missing values
            missing_value_analysis(df)

            if df.isnull().values.any():
                st.subheader("Handle Missing Values")
                
                # Sidebar selections for handling missing values
                method = st.sidebar.selectbox("Select Method", ["mean", "median", "mode", "drop"], key="missing_method")
                column = st.sidebar.selectbox("Select Column (optional)",  list(df.columns), key="missing_col")
                
                if st.button("Preview Changes", key='preview_missing_btn'):
                    df_temp, changes_made = handle_missing_values(df.copy(), method, column)
                    
                    # Only store the preview data if changes were made
                    if changes_made:
                        st.session_state['temp_data'] = df_temp
                    else:
                        st.session_state['temp_data'] = None
                        
                # Display data after handling missing values
                if 'temp_data' in st.session_state and st.session_state['temp_data'] is not None:
                    st.subheader("Data After Handling Missing Values")
                    st.write(st.session_state['temp_data'])
                    
                    # Perform missing value analysis on the modified data
                    missing_value_analysis(st.session_state['temp_data'])
                    
                    if st.button("Apply Changes", key="apply_missing_btn"):
                        df = st.session_state['temp_data']
                        st.session_state['data'] = df
                        del st.session_state['temp_data']
                        st.success("Missing values handled successfully!")
                elif 'temp_data' in st.session_state and st.session_state['temp_data'] is None:
                    st.warning("No changes applied due to an error or invalid input.")
            else:
                st.success("No missing values found in the dataset.")

        #Handle Duplicate
        elif section == "Handle Duplicates":
            reset_all_flags()
           
            st.header("Handle Duplicates")
            st.write("Original Data:")
            st.write(df)          
            original_row_count = len(df)
            num_duplicates = df.duplicated().sum()
            st.write(f"Number of duplicate rows: {num_duplicates}")
            if num_duplicates > 0:
                st.write("Duplicated Rows (if any):")
                st.write(df[df.duplicated(keep=False)])
                           
                if st.button("Remove Duplicate Rows", key='remove_duplicates'):
                    df.drop_duplicates(inplace=True)
                    st.session_state['data'] = df
                    st.session_state['duplicates_handled'] = True
                    st.success("Duplicate rows removed.")

            
            if 'duplicates_handled' in st.session_state and st.session_state['duplicates_handled']:
                updated_row_count = len(st.session_state['data'])
                st.header("Data After Handling Duplicates")
                st.write(f"Row count before removing duplicates: {original_row_count}")
                st.write(f"Row count after removing duplicates: {updated_row_count}")
                st.write(st.session_state['data'])
                st.session_state['duplicates_handled'] = False
        
        # handle outlier
        elif section == "Outlier Analysis":
            reset_all_flags()

            st.header("Outlier Analysis")
            st.write("Original Data:")
            st.write(df)

            column_for_outlier = st.selectbox(
                "Select Column for Outlier Analysis",
                df.select_dtypes(include=['float64', 'int64']).columns,
                key="outlier_col"
            )

            if column_for_outlier:
                st.write("Box Plot of Original Data:")
                fig, ax = plt.subplots()
                sns.boxplot(x=df[column_for_outlier], ax=ax)
                st.pyplot(fig)

                lower_bound, upper_bound = outlier_analysis(df, column_for_outlier)
                if lower_bound is not None and upper_bound is not None:
                    st.write(f"Outlier thresholds for {column_for_outlier}:")
                    st.write(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

                    outlier_method = st.selectbox(
                        "Select Outlier Handling Method", ['clip', 'drop'], key="outlier_method"
                    )

                    if st.button("Preview Changes", key='preview_outliers_btn'):
                        temp_df = handle_outliers(df.copy(), column_for_outlier, lower_bound, upper_bound, outlier_method)
                        st.session_state['temp_outlier_data'] = temp_df
                        st.session_state['outliers_previewed'] = True
                        st.write("Data After Handling Outliers (Preview):")
                        st.write(temp_df)

                        st.write("Box Plot After Handling Outliers (Preview):")
                        fig, ax = plt.subplots()
                        sns.boxplot(x=temp_df[column_for_outlier], ax=ax)
                        st.pyplot(fig)

                    if 'outliers_previewed' in st.session_state and st.session_state['outliers_previewed']:
                        if st.button("Apply Changes", key='apply_outliers_btn'):
                            df = st.session_state['temp_outlier_data']
                            st.session_state['data'] = df
                            st.session_state['outliers_applied'] = True

                    if 'outliers_applied' in st.session_state and st.session_state['outliers_applied']:
                        st.header("Data After Applying Outlier Handling")
                        st.write(df)

                        st.write("Box Plot After Applying Outliers:")
                        fig, ax = plt.subplots()
                        sns.boxplot(x=df[column_for_outlier], ax=ax)
                        st.pyplot(fig)

                        st.session_state['outliers_previewed'] = False
                        st.session_state['outliers_applied'] = False

        # Data Visualization
        elif section == "Data Visualization":
            column_to_visualize = st.sidebar.selectbox("Select Column for Visualization", df.columns, key="visualize_col")
            st.header("Data Visualization")
            fig1, fig2 = visualize_data(df, column_to_visualize)
            st.pyplot(fig1)
            st.pyplot(fig2)

        # Correlation Matrix
        elif section == "Correlation Matrix":
            st.header("Correlation Matrix")
            fig = correlation_matrix(df)
            if fig is not None:
                st.pyplot(fig)

                
        # Chat with Data
        elif section == "Chat Using RAG":
            st.title("Chat with your Data")
            process_files = st.checkbox("Process Files for Chat", value=False)

            if process_files:
                split_docs = []
                try:
                    if uploaded_file.name.endswith(".csv"):
                       split_docs.extend(prepare_and_split_csv([uploaded_file]))
                    if uploaded_file.name.endswith((".xlsx", ".xls")):
                        split_docs.extend(prepare_and_split_excel([uploaded_file]))
                    vector_db = ingest_into_vectordb(split_docs)
                    retriever = vector_db.as_retriever()
                    conversational_chain = get_conversation_chain(retriever)
                    st.session_state['conversational_chain'] = conversational_chain
                    st.success("Documents processed and vector database created!")
                except Exception as e:
                    st.error(f"Error processing files: {e}")

            if 'conversational_chain' in st.session_state:
                user_input = st.text_input("Ask a question about your data:")
                if st.button("Submit"):
                    try:
                        session_id = "abc123"  # You can dynamically generate this if needed
                        conversational_chain = st.session_state.conversational_chain
                        response = conversational_chain.invoke(
                            {"input": user_input},
                            {"configurable": {"session_id": session_id}}
                        )
                        context_docs = response.get('context', [])
                        st.session_state.chat_history.append({"user": user_input, "bot": response['answer'], "context_docs": context_docs})
                    except Exception as e:
                        st.error(f"Error in chat: {e}")

            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            for message in st.session_state.chat_history:
                st.markdown(user_template.format(msg=message['user']), unsafe_allow_html=True)
                st.markdown(bot_template.format(msg=message['bot']), unsafe_allow_html=True)

        # Download Dataset
        if st.sidebar.button("Download dataset", key='download_btn'):
            download_dataset(df)

if __name__ == "__main__":
    main()
