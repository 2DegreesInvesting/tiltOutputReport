# tiltOutputReport
The goal of tiltOutputReport project is to generate PDF reports with visual and textual data representations for (selected) companies. It can be used to create a pdf for a single company (with or without revenue share data) or for a list of companies. 

## Project Structure
- Data Preprocessing: This section deals with data manipulation and preparation using pandas to create DataFrames that are used throughout the project.
- Data Visualization: Visualization functions using matplotlib and seaborn to create plots and save them as images for inclusion in the PDF reports.
- PDF Generation: Utilizes the reportlab library to create the PDF reports that include company information, performance metrics, and data quality assessments.

## Installation
To run this project, ensure you have Python installed. You can install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage
1. Data Preparation: Place your input CSV files in the appropriate directory. The input files should include:
- companies.csv
- tiltLedger_mapping.csv
- tiltLedger.csv
- company_product_indicators.csv
- company_indicators.csv
- sbi_activities.csv
- companies_sbi_activities2.csv
2. Run the `single_company_pdf.py` script
3. Output: The script will generate PDF reports in the output directory. Visualizations will be saved in the figures directory.