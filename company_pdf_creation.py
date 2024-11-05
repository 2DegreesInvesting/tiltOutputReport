import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import seaborn as sns

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors

import os

from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
styles = getSampleStyleSheet()
normal_style = styles['Normal']

# Ensure the 'plots' directory exists
os.makedirs('figures', exist_ok=True)

# Ensure the 'output' directory exists
os.makedirs('output', exist_ok=True)

# Load CSV files into DataFrames
company_df = pd.read_csv('input/data/companies.csv')
company_tilt_ledger_df = pd.read_csv('input/data/tiltLedger_mapping.csv')
tilt_ledger_df = pd.read_csv('input/data/tiltLedger.csv')
company_product_indicators_df = pd.read_csv('input/data/company_product_indicators.csv')
company_indicators_df = pd.read_csv('input/data/company_indicators.csv')
sbi_activities_df = pd.read_csv('input/data/sbi_activities.csv')
companies_sbi_activities_df = pd.read_csv('input/data/companies_sbi_activities2.csv')

company_product_indicators_df = company_product_indicators_df.drop_duplicates(keep='first')
company_product_indicators_df.rename(columns={'tiltledger_id': 'tiltLedger_id'}, inplace=True)

# Import functions 
from data_preprocessing import create_single_activity_type_df, calculate_average_ranking, calculate_average_ranking_with_revenue_share
from data_visualisation_and_descriptions import create_single_indicator_figure, create_single_indicator_figure_with_revenue_shares, create_legend_image, describe_emission_rank, describe_upstream_emission_rank, describe_sector_rank, describe_upstream_sector_rank, describe_transition_risk

# Add revenue_share column
company_product_indicators_df['revenue_share'] = company_product_indicators_df['CPC_Name'].map({
    'Wheat and meslin flour': 10,
    'Groats, meal and pellets of wheat and other cereals': 80,
    'Grain mill product manufacturing services': 10
})

# Select rows with the specified company_id
selected_company_product_indicators_df = company_product_indicators_df[company_product_indicators_df['company_id'] == 'grutterij-en-spliterwtenfabriek-j-trouw-bv_nld109318-00101']

# without firm-specific information (without revenue shares)

def create_single_pdf(output_pdf, title, company_df, companies_sbi_activities_df, create_single_indicator_figure_func, calculate_average_ranking_func, single_activity_type=True, manual_average_calculation=True):
    pdf = SimpleDocTemplate(output_pdf, pagesize=letter, leftMargin=34, rightMargin=34, topMargin=30, bottomMargin=10)
    styles = getSampleStyleSheet()
    companies_sbi_activities_df['company_id'] = companies_sbi_activities_df['company_id'].astype(str)

    # Define styles
    company_style = ParagraphStyle('CompanyStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=9, textColor=colors.black, leading=12)
    bold_style = ParagraphStyle('BoldStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, textColor='#408168', leading=12)
    bold_style2 = ParagraphStyle('BoldStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, textColor='white', leading=12)
    large_bold_style = ParagraphStyle('BoldStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12, textColor='#408168', leading=14)
    normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=8, textColor=colors.black, leading=10)

    elements = []
    all_company_indicators_df = []


    elements.append(Paragraph(f"<b>Company Report ({title})</b>", large_bold_style))
    elements.append(Spacer(1, 6))

    footnotes = []

    company_id = company_df['company_id']
    if company_id in companies_sbi_activities_df['company_id'].values:
        # Get the SBI code for the company_id
        sbi_code = companies_sbi_activities_df.loc[companies_sbi_activities_df['company_id'] == company_id, 'sbi_code'].values[0]
        sbi_info = f" (SBI Code: {sbi_code})"
    else:
        sbi_info = ""
        
    # Logo and company info
    logo = Image('input/tiltLogo.png', width=60, height=30)
    company_info = f"""
    <b>Company Name:</b> {company_df['company_name']}{sbi_info}<br/>
    <b>Address:</b> {company_df['address']}, {company_df['company_city']}, {company_df['postcode']}<br/>
    <b>Description:</b> {company_df['company_description']}<br/>
    """
    logo_and_info = [[logo, Paragraph(company_info, company_style)]]
    logo_info_table = Table(logo_and_info, colWidths=[70, 420])
    logo_info_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 12)]))
    elements.append(logo_info_table)
    elements.append(Spacer(1, 4))

    # Performance Section
    elements.append(Paragraph("<b>Performance</b>", bold_style))
    elements.append(Spacer(1, 6))

    # Obtain the unique company products
    selected_company_tilt_ledger_df = company_tilt_ledger_df[company_tilt_ledger_df['company_id'] == company_id]
    selected_company_tilt_ledger_df = selected_company_tilt_ledger_df.merge(tilt_ledger_df, on='tiltLedger_id', how='inner')

    seleted_company_product_indicators_df = company_product_indicators_df[company_product_indicators_df['company_id'] == company_id]
    
    if single_activity_type:
        seleted_company_product_indicators_df = create_single_activity_type_df(seleted_company_product_indicators_df)
    
    company_products = seleted_company_product_indicators_df.dropna(subset=['CPC_Name'])[['CPC_Name', 'geography', 'ISIC_Name', 'tilt_sector', 'tiltLedger_id', 'revenue_share']].drop_duplicates(subset=['CPC_Name', 'geography'])

    seleted_company_indicators_df = company_indicators_df[company_indicators_df['company_id'] == company_id]
    
    if manual_average_calculation:
        seleted_company_indicators_df = calculate_average_ranking_func(seleted_company_product_indicators_df, seleted_company_indicators_df)
        
    all_company_indicators_df.append(seleted_company_indicators_df)

    benchmarks = {
        "EP": "tilt_sector",
        "EPU": "input_tilt_sector",
        "SP": "ipr_1.5c rps_2030",
        "SPU": "ipr_1.5c rps_2030",
        "TR": "1.5c rps_2030_tilt_sector"
    }

    indicator_labels = {
        "EP": "Relative Emission Intensity Indicator<br/>",
        "EPU": "Input Relative Emission Intensity Indicator<br/>",
        "SP": "Sector Decarbonisation Indicator<br/>",
        "SPU": "Input Sector Decarbonisation Indicator<br/>",
        "TR": "Transition Risk Indicator<br/>",
        "S1": "Scope 1 Emissions<br/>",
        "S2": "Scope 2 Emissions<br/>",
        "S3": "Scope 3 Emissions<br/>"
    }

    benchmark_labels = {
        "tilt_sector": "<i>tilt</i> sector",
        "input_tilt_sector": "input <i>tilt</i> sector",
        "ipr_1.5c rps_2030": "IPR 1.5°C RPS 2030",
        "ipr_1.5c rps_2050": "IPR 1.5°C RPS 2050",
        "weo_nz 2050_2030": "WEO NZE 2030",
        "weo_nz 2050_2050": "WEO NZE 2050", 
        "1.5c rps_2030_tilt_sector": "input <i>tilt</i> sector & IPR 1.5 RPS 2030", 
        "nz 2050_2030_tilt_sector": "input <i>tilt</i> sector & WEO NZE 2030"
    }

    # summary description
    for indicator, benchmark in benchmarks.items():
        indicator_data = seleted_company_indicators_df[
            (seleted_company_indicators_df['Indicator'] == indicator) &
            (seleted_company_indicators_df['benchmark'] == benchmark)
        ]
        
        if not indicator_data.empty:
            average_ranking = round(indicator_data['average_ranking'].values[0], 2)
            company_score = indicator_data['company_score'].values[0]
            if company_score not in ["high", "medium", "low"]:
                average_ranking = np.nan
                company_score = "N/A"
        else:
            average_ranking = np.nan
            company_score = "N/A"

        benchmark_label = benchmark_labels[benchmark]
        
        # summary description
        if indicator == 'EP':
            description = describe_emission_rank(average_ranking, benchmark_label)
            elements.append(Paragraph(f"{description}", normal_style))
            elements.append(Spacer(1, 4))

        if indicator == 'EPU':
            description = describe_upstream_emission_rank(average_ranking, benchmark_label)
            elements.append(Paragraph(f"{description}", normal_style))
            elements.append(Spacer(1, 4))

        if indicator == 'SP':
            description = describe_sector_rank(average_ranking, benchmark, benchmark_label)
            elements.append(Paragraph(f"{description}", normal_style))
            elements.append(Spacer(1, 4))

        if indicator == 'SPU':
            description = describe_upstream_sector_rank(average_ranking, benchmark, benchmark_label)
            elements.append(Paragraph(f"{description}", normal_style))
            elements.append(Spacer(1, 4))

        if indicator == "TR": 
            description = describe_transition_risk(average_ranking, benchmark, benchmark_label)
            elements.append(Paragraph(f"{description}", normal_style))
            elements.append(Spacer(1, 8))
    
    # Define the tables with captions
    indicator_table_data = [[Paragraph("Indicator", bold_style2)]]
    combined_table_data = [[Paragraph("Average Company Level Scores", bold_style2)]]
    score_breakdown_table_data = [[Paragraph("Score Breakdown by Product", bold_style2)]]
    coverage_table_data = [[Paragraph("Share of Products With a Score", bold_style2)]]

    row_height = 65  # Set the row height based on the image height

    indicators_colors = {}
    for indicator, benchmark in benchmarks.items():
        indicator_data = seleted_company_indicators_df[
            (seleted_company_indicators_df['Indicator'] == indicator) &
            (seleted_company_indicators_df['benchmark'] == benchmark)
        ]
        
        if not indicator_data.empty:
            average_ranking = round(indicator_data['average_ranking'].values[0], 2)
            company_score = indicator_data['company_score'].values[0]
            display_score = f"{company_score.capitalize()} ({average_ranking}<super>1</super>)" if company_score in ["high", "medium", "low"] else "N/A"
        else:
            display_score = "N/A"
            company_score = "Not Available"
            
        bg_color = {
            'low': '#B3D15D',    # Green
            'medium': '#F8C646', # Yellow
            'high': '#F47B3E',   # Red
            'Not Available': '#E2E5E9'
        }
        
        indicators_colors[indicator] = bg_color[company_score]

        no_total_products, no_available_products = create_single_indicator_figure_func(
            seleted_company_product_indicators_df, company_id, company_products, indicator, include_unavailable=False
        )
        
        indicator_table_data.append([Paragraph(indicator_labels[indicator], normal_style)])
        combined_table_data.append([Paragraph(display_score, normal_style)])
        score_breakdown_table_data.append([Image(
            f'figures/{indicator}_product_score_distribution_{company_id}.png',
            width=140, height=row_height
        ) if os.path.exists(f'figures/{indicator}_product_score_distribution_{company_id}.png') else Image('figures/coming_soon.png', width=180, height=row_height)])
        
        coverage_lines = [
            f"For {benchmark_labels[benchmark]}: {no_available}/{no_total_products}"
            for benchmark, no_available in no_available_products.items()
        ]
        coverage_text = "<br/>".join(coverage_lines)
        
        coverage_table_data.append([Paragraph(f'{coverage_text}', normal_style)])
    
    # Create tables
    first_row_height = 30  # Example height for the caption row
    regular_row_height = 70  # Height for the other rows
    
    indicator_table = Table(indicator_table_data, colWidths=[100], rowHeights=[first_row_height] + [regular_row_height] * (len(indicator_table_data) - 1))
    combined_table = Table(combined_table_data, colWidths=[100], rowHeights=[first_row_height] + [regular_row_height] * (len(combined_table_data) - 1))
    score_breakdown_table = Table(score_breakdown_table_data, colWidths=[160], rowHeights=[first_row_height] + [regular_row_height] * (len(score_breakdown_table_data) - 1))
    coverage_table = Table(coverage_table_data, colWidths=[130], rowHeights=[first_row_height] + [regular_row_height] * (len(coverage_table_data) - 1))
    
    # Apply styles
    for table in [indicator_table, combined_table, score_breakdown_table, coverage_table]:
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#287155')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.whitesmoke),  # Horizontal line below header
            ('LINEBELOW', (0, 1), (-1, -1), 0.5, colors.whitesmoke), # Horizontal lines below each row
            ('BOX', (0, 0), (-1, -1), 0.5, colors.whitesmoke)        # Outer box
        ]))
        
    # Apply row-specific background colors to combined table
    for i in range(1, len(combined_table_data)):
        indicator = list(benchmarks.keys())[i - 1]
        combined_table.setStyle(TableStyle([
            ('BACKGROUND', (0, i), (0, i), indicators_colors[indicator])
        ]))

    # Combine tables horizontally
    combined_tables = Table([[indicator_table, combined_table, score_breakdown_table, coverage_table]])
    combined_tables.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER')
    ]))

    elements.append(combined_tables)
    
    # Define data for the table
    emission_table_data = [
        [Paragraph("GHG Emissions Indicators", bold_style2), Paragraph("Average Absolute emissions (in tCO2eq)", bold_style2), Paragraph("Share of Products With a Score", bold_style2)]
    ]
    
    # Loop through each GHG scope indicator
    for s_indicator in ['S1', 'S2', 'S3']:
        emission_indicator_data = seleted_company_indicators_df[
            seleted_company_indicators_df['Indicator'] == s_indicator
        ]
        
        if not emission_indicator_data.empty:
            average_ranking = emission_indicator_data['average_ranking_no_benchmark'].values[-1]
            average_ranking = f"{average_ranking:.8f}"
        else:
            average_ranking = "N/A"
        
        no_total_products = len(company_products)
        group = seleted_company_product_indicators_df[seleted_company_product_indicators_df['Indicator'] == s_indicator]
        no_available_products = group['CPC_Name'].nunique()
        
        # Prepare row based on the current indicator
        if s_indicator == 'S1':
            row_label = "Scope 1 Indicator"
        elif s_indicator == 'S2':
            row_label = "Scope 2 Indicator"
        else:
            row_label = "Scope 3 Indicator"
        
        coverage_text = f"{no_available_products}/{no_total_products}"
        
        # Append the row to the table data
        emission_table_data.append([
            Paragraph(row_label, normal_style),
            Paragraph(str(average_ranking), normal_style),
            Paragraph(coverage_text, normal_style)
        ])
    
    # Create the table
    emission_table = Table(emission_table_data, colWidths=[175, 175, 175])
    
    # Apply styles to the table
    emission_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#287155')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    
    # Append the table to the elements list
    elements.append(emission_table)
    elements.append(Spacer(1, 20))

    # Add footnote text at the bottom of the page
    footnotes.append(Paragraph("<super>1</super> Company score: this is the average of the available product scores.", normal_style))
    elements.extend(footnotes)

    #elements.append(PageBreak())
    # Create the single header table
    #climate_action_table_data = [[Paragraph("Climate Action", bold_style2)]]
    
    #climate_action_table = Table(climate_action_table_data, colWidths=[500])
    #climate_action_table.setStyle(TableStyle([
    #   ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#287155')),
    #   ('TEXTCOLOR', (0, 0), (-1, -1), colors.whitesmoke),
    #  ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    #    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    #   ('FONTSIZE', (0, 0), (-1, -1), 14),
    #   ('TOPPADDING', (0, 0), (-1, -1), 10),
    #    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    #]))
    
    #elements.append(climate_action_table)
    
    # Add space for pasting a graph (1/2 page)
    #elements.append(Spacer(1, 400))  # Adjust the height as needed

    elements.append(PageBreak())
    
    # Appendix
    elements.append(Paragraph("<b>Appendix</b>", large_bold_style))
    elements.append(Spacer(1, 10))

    footnotes = []
    
    # Data Quality Section
    elements.append(Paragraph("<b>Data Quality</b>", bold_style))
    elements.append(Spacer(1, 4))

    # Matched Products Table
    elements.append(Paragraph("Matched Products", bold_style))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("The Tilt Ledger stores product and activity combinations for commercial activities in a specific geography.", normal_style))
    elements.append(Spacer(1, 6))
    
    # Create ledger_data with additional columns
    ledger_data = [
        [
            Paragraph("CPC Name", bold_style2),
            Paragraph("Geography<super>3</super>", bold_style2),
            Paragraph("ISIC Name", bold_style2),
            Paragraph("Tilt Sector", bold_style2),
            Paragraph("Revenue Share", bold_style2)
        ]
    ]
    
    # Iterate over the unique rows in company_products
    for cpc_name, geography, isic_name, tilt_sector, revenue_share in company_products[['CPC_Name', 'geography', 'ISIC_Name', 'tilt_sector', 'revenue_share']].drop_duplicates().itertuples(index=False):
        # Check for revenue_share value and format as percentage
        if title == "with firm-specific information": 
            revenue_share_percentage = f"{revenue_share:.0f}%" 
        else: 
            revenue_share_percentage = "33%"
        
        ledger_data.append([
            Paragraph(str(cpc_name) if not pd.isna(cpc_name) else 'N/A', normal_style),
            Paragraph(str(geography) if not pd.isna(geography) else 'N/A', normal_style),
            Paragraph(str(isic_name) if not pd.isna(isic_name) else 'N/A', normal_style),
            Paragraph(str(tilt_sector) if not pd.isna(tilt_sector) else 'N/A', normal_style),
            Paragraph(revenue_share_percentage, normal_style)
        ])
        
    # Create the table
    ledger_table = Table(ledger_data, colWidths=[170, 60, 140, 70, 80])
    ledger_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), '#287155'),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),  # Align text to the top of each cell
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    elements.append(ledger_table)
    elements.append(Spacer(1, 8))

    # DQ Cards
    indicators = ['EP', 'EPU', 'SP', 'SPU']
    grouped = seleted_company_indicators_df.groupby('Indicator')
    
    # Create a template DataFrame for indicators
    template_df = pd.DataFrame({
        'Indicator': indicators,
        'model_certainty': [float('nan')] * len(indicators),
        'data_source_reliability': [float('nan')] * len(indicators),
        'data_granularity': [float('nan')] * len(indicators)
    })
    
    # Update template DataFrame with available data
    for indicator, group in grouped:
        row = group.iloc[0][['model_certainty', 'data_source_reliability', 'data_granularity']]
        template_df.loc[template_df['Indicator'] == indicator, ['model_certainty', 'data_source_reliability', 'data_granularity']] = row.values
            
    # Create dq cards
    dq_cards = []
    for row in template_df.itertuples():
        dq_card_data = [
            [Paragraph(f"<b>{indicator_labels[row.Indicator]}</b>", bold_style)],
            [Paragraph(f"Model Certainty: {row.model_certainty:.0f}", normal_style)],
            [Paragraph(f"Data Source Reliability: {row.data_source_reliability:.0f}", normal_style)],
            [Paragraph(f"Data Granularity: {row.data_granularity:.0f}", normal_style)]
        ]
        dq_card = Table(dq_card_data, colWidths=[120])  # Set card width to 100
        dq_card.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.grey),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        dq_cards.append(dq_card)
    
    # Calculate spacing between cards
    num_cards = len(dq_cards)
    total_width = 520
    card_width = 120
    total_card_width = card_width * num_cards
    # Space between cards
    space_between = (total_width - total_card_width) / (num_cards - 1)
    
    # Arrange KPI cards horizontally with spaces in between
    card_row = []
    for dq_card in dq_cards:
        card_row.append(dq_card)
        card_row.append(Spacer(space_between, 1))  # Add horizontal space
    
    # Remove the last spacer
    card_row.pop()
    
    # Create the table
    dq_card_table = Table([card_row], colWidths=[card_width, space_between] * (num_cards - 1) + [card_width])
    
    # Style the table
    dq_card_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    
    # Add the table to elements
    elements.append(dq_card_table)
    elements.append(Spacer(1, 8))

    # Glossary
    elements.append(Paragraph("<b>Glossary</b>", bold_style))
    elements.append(Spacer(1, 4))

    glossary_entries = [
    ("Benchmark:", "The benchmark defines the specific subgroup of products against which the company's products are compared (the reference group)."),
    ("Data granularity:", "Evaluates the availability of detailed product information, the coverage of location data, and the regional level at which scenario and emissions data are collected (from country to global). The final score is determined by the highest (worst) score among the three criteria. The score is given on a scale from 1 (best) to 5 (worst)."),
    ("Data Source Reliability:", "Evaluates how trustworthy and suitable external data sources are. It assesses fit-for-purpose, source quality, and data quality. The final score is determined by the highest (worst) score among the three criteria. If multiple data sources are used, the overall reliability is based on the lowest score among them. The score is given on a scale from 1 (best) to 5 (worst)."),
    ("Emission rank:", "The rank of the product's emissions compared to all other products within the same benchmark group."),
    ("IPR 1.5 RPS:", "The Inevitable Policy Response (IPR) Required Policy Scenario (RPS) outlines a series of necessary policy actions aimed at limiting global warming to 1.5°C."),
    ("Target year:", "The target year for reaching the goals of the chosen scenario. The target year can be either 2030 or 2050."),
    ("Input Relative Emission Intensity Indicator:", "The upstream variant of the Relative Emission Intensity Indicator, which only considers the input products."),
    ("Input Sector Decarbonisation Indicator:", "The input variant of the Sector Profile, which only considers the input products."),
    ("Model Certainty:", "Model certainty is about ensuring the data fits well into a structured system (the tiltLedger), with lower distances and fewer mismatches indicating higher confidence in the accuracy of the data matching. The score is given on a scale from 1 (best) to 5 (worst)."),
    ("Relative Emission Intensity Indicator:", "Measures a product's emission intensity expressed in carbon dioxide equivalents (CO2e) per product unit (e.g. kg, ha, sqm, kWh) compared to the emission intensity of other products."),
    ("Scenario:", "The scenario determines the specific sector emission reduction targets."),
    ("Sector Decarbonisation Indicator:", "Indicates the pressure on a firm to reduce its CO2e (kg/unit) per product, based on forward-looking and sector-specific decarbonisation targets from the climate scenario."),
    ("Tilt sector:", "The corresponding tilt sector of the company's product (in case no Ecoinvent match exists) or the Ecoinvent's activity, if a match exists."),
    ("Transition Risk Indicator:", "The average of the Relative Emission Intensity Indicator, which indicates the firm's exposure, and the Sector Decarbonisation Indicator, which indicates the hazard."),
    ("WEO NZE:", "The World Energy Outlook (WEO) Net Zero Emissions (NZE) Scenario outlines a series of necessary policy actions aimed at achieving net-zero carbon emissions globally by 2050."),
    ("GHG Scope 1<super>4</super>:", "Direct GHG emission amounts (in tonnes) resulting directly from the company's operations, based on the companies' activities data."),
    ("GHG Scope 2<super>4</super>:", "Indirect GHG emissions (in tonnes) from the generation of purchased energy and electricity, based on the companies' activities data."),
    ("GHG Scope 3<super>4</super>:", "All indirect GHG emissions (in tonnes) which are not included in scope 2 that occur in the value chain of the reporting company, including both upstream and downstream emissions, based on the companies' activities data.")
]

    for term, definition in sorted(glossary_entries):
        elements.append(Paragraph(f"<b>{term}</b> {definition}", normal_style))
        elements.append(Spacer(1, 4))

    footnotes.append(Paragraph("<super>2</super> The country of the matched product for which the indicators can be calculated.", normal_style))
    footnotes.append(Paragraph("<super>3</super> All GHG emissions are accounted for which are required GHGs for corporate or product inventories in the Accounting and Reporting Standard Amendment (Greenhouse gas protocol, 2013). ", normal_style))
    elements.extend(footnotes)

    elements.append(PageBreak())

    pdf.build(elements)

    # After your loop
    all_company_indicators_df = pd.concat(all_company_indicators_df, ignore_index=True)
    return all_company_indicators_df

# run the function without firm specific info 
company_indicators_df_v1 = create_single_pdf('output/single_company_report_no_firm_info.pdf', 'without firm-specific information', company_df.iloc[1], companies_sbi_activities_df, create_single_indicator_figure, calculate_average_ranking, single_activity_type=True, manual_average_calculation=True)

# run the function with firm specific info 
company_indicators_df_v1 = create_single_pdf('output/single_company_report_with_firm_info.pdf', 'with firm-specific information', company_df.iloc[1], companies_sbi_activities_df, create_single_indicator_figure_with_revenue_shares, calculate_average_ranking_with_revenue_share, single_activity_type=True, manual_average_calculation=True)

# combined pdf
def create_pdf(output_pdf, company_df, companies_sbi_activities_df, single_activity_type=True, manual_average_calculation=True):
    pdf = SimpleDocTemplate(output_pdf, pagesize=letter, leftMargin=34, rightMargin=34, topMargin=30, bottomMargin=10)
    styles = getSampleStyleSheet()
    companies_sbi_activities_df['company_id'] = companies_sbi_activities_df['company_id'].astype(str)

    # Define styles
    company_style = ParagraphStyle('CompanyStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=9, textColor=colors.black, leading=12)
    bold_style = ParagraphStyle('BoldStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, textColor='#408168', leading=12)
    bold_style2 = ParagraphStyle('BoldStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, textColor='white', leading=12)
    large_bold_style = ParagraphStyle('BoldStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12, textColor='#408168', leading=14)
    normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=8, textColor=colors.black, leading=10)

    elements = []
    all_company_indicators_df = []

    for index, company in company_df.iterrows():
        elements.append(Paragraph("<b>Company Report (without firm-specific information)</b>", large_bold_style))
        elements.append(Spacer(1, 6))

        footnotes = []

        company_id = company['company_id']
        print(company_id)
        if company_id in companies_sbi_activities_df['company_id'].values:
            # Get the SBI code for the company_id
            sbi_code = companies_sbi_activities_df.loc[companies_sbi_activities_df['company_id'] == company_id, 'sbi_code'].values[0]
            print(sbi_code)
            sbi_info = f" (SBI Code: {sbi_code})"
        else:
            sbi_info = ""
            
        # Logo and company info
        logo = Image('input/tiltLogo.png', width=60, height=30)
        company_info = f"""
        <b>Company Name:</b> {company['company_name']}{sbi_info}<br/>
        <b>Address:</b> {company['address']}, {company['company_city']}, {company['postcode']}<br/>
        <b>Description:</b> {company['company_description']}<br/>
        """
        logo_and_info = [[logo, Paragraph(company_info, company_style)]]
        logo_info_table = Table(logo_and_info, colWidths=[70, 420])
        logo_info_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 12)]))
        elements.append(logo_info_table)
        elements.append(Spacer(1, 4))

        # Performance Section
        elements.append(Paragraph("<b>Performance</b>", bold_style))
        elements.append(Spacer(1, 6))

        # Obtain the unique company products
        selected_company_tilt_ledger_df = company_tilt_ledger_df[company_tilt_ledger_df['company_id'] == company_id]
        selected_company_tilt_ledger_df = selected_company_tilt_ledger_df.merge(tilt_ledger_df, on='tiltLedger_id', how='inner')
 
        seleted_company_product_indicators_df = company_product_indicators_df[company_product_indicators_df['company_id'] == company_id]
        
        if single_activity_type:
            seleted_company_product_indicators_df = create_single_activity_type_df(seleted_company_product_indicators_df)
        
        company_products = seleted_company_product_indicators_df.dropna(subset=['CPC_Name'])[['CPC_Name', 'geography', 'ISIC_Name', 'tilt_sector', 'tiltLedger_id']].drop_duplicates(subset=['CPC_Name', 'geography'])

        seleted_company_indicators_df = company_indicators_df[company_indicators_df['company_id'] == company_id]
        
        if manual_average_calculation:
            seleted_company_indicators_df = calculate_average_ranking(seleted_company_product_indicators_df, seleted_company_indicators_df)
            
        all_company_indicators_df.append(seleted_company_indicators_df)

        benchmarks = {
            "EP": "tilt_sector",
            "EPU": "input_tilt_sector",
            "SP": "ipr_1.5c rps_2030",
            "SPU": "ipr_1.5c rps_2030",
            "TR": "1.5c rps_2030_tilt_sector"
        }

        indicator_labels = {
            "EP": "Relative Emission Intensity Indicator<br/>",
            "EPU": "Input Relative Emission Intensity Indicator<br/>",
            "SP": "Sector Decarbonisation Indicator<br/>",
            "SPU": "Input Sector Decarbonisation Indicator<br/>",
            "TR": "Transition Risk Indicator<br/>",
            "S1": "Scope 1 Emissions<br/>",
            "S2": "Scope 2 Emissions<br/>",
            "S3": "Scope 3 Emissions<br/>"
        }

        benchmark_labels = {
            "tilt_sector": "<i>tilt</i> sector",
            "input_tilt_sector": "input <i>tilt</i> sector",
            "ipr_1.5c rps_2030": "IPR 1.5°C RPS 2030",
            "ipr_1.5c rps_2050": "IPR 1.5°C RPS 2050",
            "weo_nz 2050_2030": "WEO NZE 2030",
            "weo_nz 2050_2050": "WEO NZE 2050", 
            "1.5c rps_2030_tilt_sector": "input <i>tilt</i> sector & IPR 1.5 RPS 2030", 
            "nz 2050_2030_tilt_sector": "input <i>tilt</i> sector & WEO NZE 2030"
        }

        # summary description
        for indicator, benchmark in benchmarks.items():
            indicator_data = seleted_company_indicators_df[
                (seleted_company_indicators_df['Indicator'] == indicator) &
                (seleted_company_indicators_df['benchmark'] == benchmark)
            ]
            
            if not indicator_data.empty:
                average_ranking = round(indicator_data['average_ranking'].values[0], 2)
                company_score = indicator_data['company_score'].values[0]
                if company_score not in ["high", "medium", "low"]:
                    average_ranking = np.nan
                    company_score = "N/A"
            else:
                average_ranking = np.nan
                company_score = "N/A"

            benchmark_label = benchmark_labels[benchmark]
            
            # summary description
            if indicator == 'EP':
                description = describe_emission_rank(average_ranking, benchmark_label)
                elements.append(Paragraph(f"{description}", normal_style))
                elements.append(Spacer(1, 4))

            if indicator == 'EPU':
                description = describe_upstream_emission_rank(average_ranking, benchmark_label)
                elements.append(Paragraph(f"{description}", normal_style))
                elements.append(Spacer(1, 4))

            if indicator == 'SP':
                description = describe_sector_rank(average_ranking, benchmark, benchmark_label)
                elements.append(Paragraph(f"{description}", normal_style))
                elements.append(Spacer(1, 4))

            if indicator == 'SPU':
                description = describe_upstream_sector_rank(average_ranking, benchmark, benchmark_label)
                elements.append(Paragraph(f"{description}", normal_style))
                elements.append(Spacer(1, 4))

            if indicator == "TR": 
                description = describe_transition_risk(average_ranking, benchmark, benchmark_label)
                elements.append(Paragraph(f"{description}", normal_style))
                elements.append(Spacer(1, 8))
        
        # Define the tables with captions
        indicator_table_data = [[Paragraph("Indicator", bold_style2)]]
        combined_table_data = [[Paragraph("Average Company Level Scores", bold_style2)]]
        score_breakdown_table_data = [[Paragraph("Score Breakdown by Product", bold_style2)]]
        coverage_table_data = [[Paragraph("Share of Products With a Score", bold_style2)]]

        row_height = 65  # Set the row height based on the image height

        indicators_colors = {}
        for indicator, benchmark in benchmarks.items():
            indicator_data = seleted_company_indicators_df[
                (seleted_company_indicators_df['Indicator'] == indicator) &
                (seleted_company_indicators_df['benchmark'] == benchmark)
            ]
            
            if not indicator_data.empty:
                average_ranking = round(indicator_data['average_ranking'].values[0], 2)
                company_score = indicator_data['company_score'].values[0]
                display_score = f"{company_score.capitalize()} ({average_ranking}<super>1</super>)" if company_score in ["high", "medium", "low"] else "N/A"
            else:
                display_score = "N/A"
                company_score = "Not Available"
                
            bg_color = {
                'low': '#B3D15D',    # Green
                'medium': '#F8C646', # Yellow
                'high': '#F47B3E',   # Red
                'Not Available': '#E2E5E9'
            }
            
            indicators_colors[indicator] = bg_color[company_score]

            no_total_products, no_available_products = create_single_indicator_figure(
                seleted_company_product_indicators_df, company_id, company_products, indicator, include_unavailable=False
            )
            
            indicator_table_data.append([Paragraph(indicator_labels[indicator], normal_style)])
            combined_table_data.append([Paragraph(display_score, normal_style)])
            score_breakdown_table_data.append([Image(
                f'figures/{indicator}_product_score_distribution_{company_id}.png',
                width=140, height=row_height
            ) if os.path.exists(f'figures/{indicator}_product_score_distribution_{company_id}.png') else Image('figures/coming_soon.png', width=180, height=row_height)])
            
            coverage_lines = [
                f"For {benchmark_labels[benchmark]}: {no_available}/{no_total_products}"
                for benchmark, no_available in no_available_products.items()
            ]
            coverage_text = "<br/>".join(coverage_lines)
            
            coverage_table_data.append([Paragraph(f'{coverage_text}', normal_style)])
        
        # Create tables
        first_row_height = 30  # Example height for the caption row
        regular_row_height = 70  # Height for the other rows
        
        indicator_table = Table(indicator_table_data, colWidths=[100], rowHeights=[first_row_height] + [regular_row_height] * (len(indicator_table_data) - 1))
        combined_table = Table(combined_table_data, colWidths=[100], rowHeights=[first_row_height] + [regular_row_height] * (len(combined_table_data) - 1))
        score_breakdown_table = Table(score_breakdown_table_data, colWidths=[160], rowHeights=[first_row_height] + [regular_row_height] * (len(score_breakdown_table_data) - 1))
        coverage_table = Table(coverage_table_data, colWidths=[130], rowHeights=[first_row_height] + [regular_row_height] * (len(coverage_table_data) - 1))
        
        # Apply styles
        for table in [indicator_table, combined_table, score_breakdown_table, coverage_table]:
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#287155')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.whitesmoke),  # Horizontal line below header
                ('LINEBELOW', (0, 1), (-1, -1), 0.5, colors.whitesmoke), # Horizontal lines below each row
                ('BOX', (0, 0), (-1, -1), 0.5, colors.whitesmoke)        # Outer box
            ]))
            
        # Apply row-specific background colors to combined table
        for i in range(1, len(combined_table_data)):
            indicator = list(benchmarks.keys())[i - 1]
            combined_table.setStyle(TableStyle([
                ('BACKGROUND', (0, i), (0, i), indicators_colors[indicator])
            ]))

        # Combine tables horizontally
        combined_tables = Table([[indicator_table, combined_table, score_breakdown_table, coverage_table]])
        combined_tables.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER')
        ]))

        elements.append(combined_tables)
        
        # Define data for the table
        emission_table_data = [
            [Paragraph("GHG Emissions Indicators", bold_style2), Paragraph("Average Absolute CO2e Emissions (in tonnes)", bold_style2), Paragraph("Share of Products With a Score", bold_style2)]
        ]
        
        # Loop through each GHG scope indicator
        for s_indicator in ['S1', 'S2', 'S3']:
            emission_indicator_data = seleted_company_indicators_df[
                seleted_company_indicators_df['Indicator'] == s_indicator
            ]
            
            if not emission_indicator_data.empty:
                average_ranking = emission_indicator_data['average_ranking_no_benchmark'].values[-1]
                average_ranking = f"{average_ranking:.8f}"
            else:
                average_ranking = "N/A"
            
            no_total_products = len(company_products)
            group = seleted_company_product_indicators_df[seleted_company_product_indicators_df['Indicator'] == s_indicator]
            no_available_products = group['CPC_Name'].nunique()
            
            # Prepare row based on the current indicator
            if s_indicator == 'S1':
                row_label = "Scope 1 Indicator"
            elif s_indicator == 'S2':
                row_label = "Scope 2 Indicator"
            else:
                row_label = "Scope 3 Indicator"
            
            coverage_text = f"{no_available_products}/{no_total_products}"
            
            # Append the row to the table data
            emission_table_data.append([
                Paragraph(row_label, normal_style),
                Paragraph(str(average_ranking), normal_style),
                Paragraph(coverage_text, normal_style)
            ])
        
        # Create the table
        emission_table = Table(emission_table_data, colWidths=[175, 175, 175])
        
        # Apply styles to the table
        emission_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#287155')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        # Append the table to the elements list
        elements.append(emission_table)
        elements.append(Spacer(1, 20))

        # Add footnote text at the bottom of the page
        footnotes.append(Paragraph("<super>1</super> Company score: this is the average of the available product scores.", normal_style))
        elements.extend(footnotes)

        elements.append(PageBreak())
        
        # Appendix
        elements.append(Paragraph("<b>Appendix</b>", large_bold_style))
        elements.append(Spacer(1, 10))

        footnotes = []
        
        # Data Quality Section
        elements.append(Paragraph("<b>Data Quality</b>", bold_style))
        elements.append(Spacer(1, 4))

        # Matched Products Table
        elements.append(Paragraph("Matched Products", bold_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph("The Tilt Ledger stores product and activity combinations for commercial activities in a specific geography.", normal_style))
        elements.append(Spacer(1, 6))
        
        # Create ledger_data with additional columns
        ledger_data = [[Paragraph("CPC Name", bold_style2), Paragraph("Geography<super>3</super>", bold_style2), Paragraph("ISIC Name", bold_style2), Paragraph("Tilt Sector", bold_style2)]]
             
        for cpc_name, geography, isic_name, tilt_sector in company_products[['CPC_Name', 'geography', 'ISIC_Name', 'tilt_sector']].drop_duplicates().itertuples(index=False):
            ledger_data.append([
                Paragraph(str(cpc_name) if not pd.isna(cpc_name) else 'N/A', normal_style),
                Paragraph(str(geography) if not pd.isna(geography) else 'N/A', normal_style),
                Paragraph(str(isic_name) if not pd.isna(isic_name) else 'N/A', normal_style),
                Paragraph(str(tilt_sector) if not pd.isna(tilt_sector) else 'N/A', normal_style)
            ])
        
        # Create the table
        ledger_table = Table(ledger_data, colWidths=[180, 80, 160, 100])
        ledger_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#287155'),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),  # Align text to the top of each cell
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ]))
        elements.append(ledger_table)
        elements.append(Spacer(1, 8))

        # DQ Cards
        indicators = ['EP', 'EPU', 'SP', 'SPU']
        grouped = seleted_company_indicators_df.groupby('Indicator')
        
        # Create a template DataFrame for indicators
        template_df = pd.DataFrame({
            'Indicator': indicators,
            'model_certainty': [float('nan')] * len(indicators),
            'data_source_reliability': [float('nan')] * len(indicators),
            'data_granularity': [float('nan')] * len(indicators)
        })
        
        # Update template DataFrame with available data
        for indicator, group in grouped:
            row = group.iloc[0][['model_certainty', 'data_source_reliability', 'data_granularity']]
            template_df.loc[template_df['Indicator'] == indicator, ['model_certainty', 'data_source_reliability', 'data_granularity']] = row.values
                
        # Create dq cards
        dq_cards = []
        for row in template_df.itertuples():
            dq_card_data = [
                [Paragraph(f"<b>{indicator_labels[row.Indicator]}</b>", bold_style)],
                [Paragraph(f"Model Certainty: {row.model_certainty:.0f}", normal_style)],
                [Paragraph(f"Data Source Reliability: {row.data_source_reliability:.0f}", normal_style)],
                [Paragraph(f"Data Granularity: {row.data_granularity:.0f}", normal_style)]
            ]
            dq_card = Table(dq_card_data, colWidths=[120])  # Set card width to 100
            dq_card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.grey),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.grey),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ]))
            
            dq_cards.append(dq_card)
        
        # Calculate spacing between cards
        num_cards = len(dq_cards)
        total_width = 520
        card_width = 120
        total_card_width = card_width * num_cards
        # Space between cards
        space_between = (total_width - total_card_width) / (num_cards - 1)
        
        # Arrange KPI cards horizontally with spaces in between
        card_row = []
        for dq_card in dq_cards:
            card_row.append(dq_card)
            card_row.append(Spacer(space_between, 1))  # Add horizontal space
        
        # Remove the last spacer
        card_row.pop()
        
        # Create the table
        dq_card_table = Table([card_row], colWidths=[card_width, space_between] * (num_cards - 1) + [card_width])
        
        # Style the table
        dq_card_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        
        # Add the table to elements
        elements.append(dq_card_table)
        elements.append(Spacer(1, 8))

        # Glossary
        elements.append(Paragraph("<b>Glossary</b>", bold_style))
        elements.append(Spacer(1, 4))

        glossary_entries = [
        ("Benchmark:", "The benchmark defines the specific subgroup of products against which the company's products are compared (the reference group)."),
        ("Data granularity:", "Evaluates the availability of detailed product information, the coverage of location data, and the regional level at which scenario and emissions data are collected (from country to global). The final score is determined by the highest (worst) score among the three criteria. The score is given on a scale from 1 (best) to 5 (worst)."),
        ("Data Source Reliability:", "Evaluates how trustworthy and suitable external data sources are. It assesses fit-for-purpose, source quality, and data quality. The final score is determined by the highest (worst) score among the three criteria. If multiple data sources are used, the overall reliability is based on the lowest score among them. The score is given on a scale from 1 (best) to 5 (worst)."),
        ("Emission rank:", "The rank of the product's emissions compared to all other products within the same benchmark group."),
        ("IPR 1.5 RPS:", "The Inevitable Policy Response (IPR) Required Policy Scenario (RPS) outlines a series of necessary policy actions aimed at limiting global warming to 1.5°C."),
        ("Target year:", "The target year for reaching the goals of the chosen scenario. The target year can be either 2030 or 2050."),
        ("Input Relative Emission Intensity Indicator:", "The upstream variant of the Relative Emission Intensity Indicator, which only considers the input products."),
        ("Input Sector Decarbonisation Indicator:", "The input variant of the Sector Profile, which only considers the input products."),
        ("Model Certainty:", "Model certainty is about ensuring the data fits well into a structured system (the tiltLedger), with lower distances and fewer mismatches indicating higher confidence in the accuracy of the data matching. The score is given on a scale from 1 (best) to 5 (worst)."),
        ("Relative Emission Intensity Indicator:", "Measures a product's emission intensity expressed in carbon dioxide equivalents (CO2e) per product unit (e.g. kg, ha, sqm, kWh) compared to the emission intensity of other products."),
        ("Scenario:", "The scenario determines the specific sector emission reduction targets."),
        ("Sector Decarbonisation Indicator:", "Indicates the pressure on a firm to reduce its CO2e (kg/unit) per product, based on forward-looking and sector-specific decarbonisation targets from the climate scenario."),
        ("Tilt sector:", "The corresponding tilt sector of the company's product (in case no Ecoinvent match exists) or the Ecoinvent's activity, if a match exists."),
        ("Transition Risk Indicator:", "The average of the Relative Emission Intensity Indicator, which indicates the firm's exposure, and the Sector Decarbonisation Indicator, which indicates the hazard."),
        ("WEO NZE:", "The World Energy Outlook (WEO) Net Zero Emissions (NZE) Scenario outlines a series of necessary policy actions aimed at achieving net-zero carbon emissions globally by 2050."),
        ("GHG Scope 1<super>4</super>:", "Direct GHG emission amounts (in tonnes) resulting directly from the company's operations, based on the companies' activities data."),
        ("GHG Scope 2<super>4</super>:", "Indirect GHG emissions (in tonnes) from the generation of purchased energy and electricity, based on the companies' activities data."),
        ("GHG Scope 3<super>4</super>:", "All indirect GHG emissions (in tonnes) which are not included in scope 2 that occur in the value chain of the reporting company, including both upstream and downstream emissions, based on the companies' activities data.")
    ]
    
        for term, definition in sorted(glossary_entries):
            elements.append(Paragraph(f"<b>{term}</b> {definition}", normal_style))
            elements.append(Spacer(1, 4))

        footnotes.append(Paragraph("<super>2</super> The country of the matched product for which the indicators can be calculated.", normal_style))
        footnotes.append(Paragraph("<super>3</super> All GHG emissions are accounted for which are required GHGs for corporate or product inventories in the Accounting and Reporting Standard Amendment (Greenhouse gas protocol, 2013). ", normal_style))
        elements.extend(footnotes)

        elements.append(PageBreak())

    pdf.build(elements)

    # After your loop
    all_company_indicators_df = pd.concat(all_company_indicators_df, ignore_index=True)
    return all_company_indicators_df

# Usage
company_indicators_df_v1 = create_pdf('output/company_report.pdf', company_df, companies_sbi_activities_df)

