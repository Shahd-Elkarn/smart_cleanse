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
        #Info
        elif section =="Info Data":
            st.header("Info Data")
            st.table(info_data(df))
        # Describe Data
        elif section == "Describe Data":
            st.header("Data Description")
            st.table(describe_data(df))


        # Data Type Analysis
        elif section == "Data Type Analysis":
            st.session_state['data_type_analysis_clicked'] = True
            df = data_types_analysis(df)

        if 'type_converted' in st.session_state and st.session_state['type_converted']:
            st.write(df)
            st.session_state['type_converted'] = False
            reset_all_flags()
            st.session_state['data_type_analysis_clicked'] = True
            df = data_types_analysis(df)

        if 'type_converted' in st.session_state and st.session_state['type_converted']:
            st.write(df)
            st.session_state['type_converted'] = False

        # Column Name Analysis
        elif section == "Column Name Analysis":
            df = column_names_analysis(df)

        # Missing Value Analysis
        elif section == "Missing Value Analysis":
            reset_all_flags()
            st.session_state['missing_analysis_run'] = True  # Set the flag to indicate this section is clicked
            
            if 'missing_analysis_run' in st.session_state and st.session_state['missing_analysis_run']:
                st.header("Missing Value Analysis")
                missing_value_analysis(df)  # Perform missing value analysis (your existing method)
                st.session_state['missing_analysis_run'] = False  # Reset flag to prevent re-running analysis on every refresh

            # Sidebar selections for handling missing values
            method = st.sidebar.selectbox("Select Method", ["mean", "median", "mode", "drop"], key="missing_method")
            column = st.sidebar.selectbox("Select Column (optional)", df.columns, key="missing_col")
            
            # Button to handle missing values
            if st.sidebar.button("Handle Missing Values", key='handle_missing_btn'):
                reset_all_flags()  # Reset all flags to ensure the process is clean
                df = handle_missing_values(df, method, column)  # Handle missing values based on the selected method
                st.session_state['data'] = df  # Store the updated data in session state
                st.session_state['missing_values_handled'] = True  # Set flag to indicate that missing values were handled

            # Show data after handling missing values
            if 'missing_values_handled' in st.session_state and st.session_state['missing_values_handled']:
                st.header("Data after Handling Missing Values")
                st.write(df)  
                st.session_state['missing_values_handled'] = False 
        
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
        
        #Outliers
        elif section == "Outlier Analysis":
            column_for_outlier = st.sidebar.selectbox(
                "Select Column for Outlier Analysis",
                df.select_dtypes(include=['float64', 'int64']).columns,
                key="outlier_col"
            )
            lower_bound, upper_bound = outlier_analysis(df, column_for_outlier)
            if lower_bound is not None and upper_bound is not None:
                outlier_method = st.sidebar.selectbox("Select Outlier Handling Method", ['clip', 'drop'], key="outlier_method")
                if st.sidebar.button("Handle Outliers", key='handle_outliers_btn'):
                    df = handle_outliers(df, column_for_outlier, lower_bound, upper_bound, outlier_method)
                    st.session_state['data'] = df

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
