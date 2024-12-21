# smart_cleanse
Data Quality Analysis and Cleaning Tool
🌟 Overview
This project is a Streamlit-based interactive tool designed to assist in exploring, cleaning, and analyzing datasets. Whether you're dealing with Excel (.xlsx) or CSV files, this tool provides a user-friendly interface for performing common data quality tasks, advanced transformations, and exploratory data analysis (EDA).

✨ Features
1. Data Exploration
Upload .xlsx or .csv datasets.
View the dataset: Quickly inspect the data.
Get dataset info: View details about columns, data types, and memory usage.
Descriptive statistics: Summarize your data with measures like mean, median, standard deviation, etc.
2. Data Transformation
Rename columns to make them more meaningful.
Convert column data types to suit your needs.
3. Handle Missing Values
Automatically identify missing values in the dataset.
Choose how to handle missing data:
Drop rows/columns.
Fill with mean, median, or mode.
4. Handle Duplicates
Detect and remove duplicate rows to ensure data consistency.
5. Outlier Detection & Handling
Identify outliers using statistical methods.
Choose how to address them (e.g., removal or replacement).
6. Data Visualization
Select columns to visualize with various chart options.
Display the correlation matrix to analyze relationships between variables.
7. Interactive RAG (Red-Amber-Green) Assistant
An integrated AI chatbot (powered by Ollama) for querying your dataset.
Ask questions about your dataset to gain insights and recommendations.
8. Download Cleaned Dataset
After making transformations and cleaning the data, download the updated dataset for further use.

🔧 Installation
Clone this repository:

git clone https://github.com/your-username/smart_cleanse.git  
cd smart_cleanse  


Create a virtual environment:

python -m venv env  
source env/bin/activate  # On Windows: env\Scripts\activate  


Install dependencies:
pip install -r requirements.txt 

 
Run the application:
streamlit run app.py

📂 Usage
Upload your dataset in .xlsx or .csv format.
Explore, clean, and transform your data using the intuitive interface.
Use the RAG Assistant to ask detailed questions about your data.
Download the cleaned dataset for your records or further analysis.
📊 Tech Stack
Streamlit: For building the interactive web application.
Pandas: For data manipulation and analysis.
Matplotlib/Seaborn: For data visualization.
Ollama API: For AI chatbot integration.
🌟 Why Use This Tool?
This project is perfect for data professionals and students looking to quickly clean and analyze datasets without diving into complex code. It brings the power of EDA and data transformation to your fingertips with an AI-enhanced assistant for deeper insights.

🚀 Contribute
Contributions are welcome! Feel free to fork this repository, create issues, or submit pull requests to help make this tool even better.
