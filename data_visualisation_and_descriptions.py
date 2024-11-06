# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# define data visualisation functions
# without revenue share
def create_single_indicator_figure(data, company_id, company_products, indicator, include_unavailable=False, benchmarks_rei=['tilt_sector'], benchmarks_irei=['input_tilt_sector'], scenarios=['ipr_1.5c rps_2030', 'weo_nz 2050_2030'], benchmark_tr = ['1.5c rps_2030_tilt_sector'], max_label_length=20):
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 3))
    all_scores = ['low', 'medium', 'high']
    plot_colors = ['#b3d15d', '#f8c646', '#f47b3e']

    if include_unavailable:
        all_scores.append('unavailable')
        plot_colors.append('#E2E5E9')

    benchmarks_dict = {
        "REI": benchmarks_rei,
        "IREI": benchmarks_irei,
        "SD": scenarios,
        "ISD": scenarios, 
        "TR": benchmark_tr
    }

    benchmark_labels = {
        "tilt_sector": "Tilt Sector",
        "input_tilt_sector": "Input Tilt Sector",
        "ipr_1.5c rps_2030": "IPR 1.5 RPS 2030",
        "ipr_1.5c rps_2050": "IPR 1.5 RPS 2050",
        "weo_nz 2050_2030": "WEO NZE 2030",
        "weo_nz 2050_2050": "WEO NZE 2050", 
        "1.5c rps_2030_tilt_sector": "Tilt Sector &\nIPR 1.5 RPS 2030"
    }

    no_total_products = len(company_products)

    group = data[data['Indicator'] == indicator]
    indicator_benchmarks = benchmarks_dict[indicator]

    full_index = pd.MultiIndex.from_product(
        [indicator_benchmarks, all_scores],
        names=['benchmark', 'score']
    )
    score_counts = group.groupby(['benchmark', 'score']).size().reindex(full_index, fill_value=0).unstack(fill_value=np.nan)
    number_of_na = no_total_products - score_counts.sum(axis=1)
    no_available_products = no_total_products - number_of_na
    
    if not include_unavailable:
        score_counts = score_counts.loc[(score_counts > 0).any(axis=1)]

    if include_unavailable:
        score_counts['unavailable'] = number_of_na

    score_counts = score_counts[all_scores].iloc[::-1]
    score_percentages = (score_counts.T / no_available_products * 100).T

    if not score_percentages.empty and not score_percentages.isnull().all().all():
        score_percentages.plot(
            kind='barh',
            stacked=True,
            color=plot_colors,
            ax=ax,
            legend=False,
            width=0.3,
            edgecolor='none'
        )
        
        ax.set_yticklabels([benchmark_labels.get(label, label) for label in score_percentages.index])
    
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage", fontsize=12, color='black')
        ax.set_ylabel("Benchmark" if indicator in ["REI", "IREI"] else "Scenario", fontsize=12, color='black')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='y', rotation=0, colors='black', labelsize=10)
    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
        ax.xaxis.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.yaxis.grid(False)
        
        # Adjust space to account for the longest label
        plt.subplots_adjust(left=0.2 + max_label_length * 0.01)
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        combined_filename = f'figures/{indicator}_product_score_distribution_{company_id}.png'
        plt.savefig(combined_filename, bbox_inches='tight', dpi=300)
        #plt.show()
        
    plt.close()

    return no_total_products, no_available_products

# with revenue shares
def create_single_indicator_figure_with_revenue_shares(data, company_id, company_products, indicator, include_unavailable=False, benchmarks_rei=['tilt_sector'], benchmarks_irei=['input_tilt_sector'], scenarios=['ipr_1.5c rps_2030', 'weo_nz 2050_2030'], benchmark_tr = ['1.5c rps_2030_tilt_sector'], max_label_length=20):
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 3))
    all_scores = ['low', 'medium', 'high']
    plot_colors = ['#b3d15d', '#f8c646', '#f47b3e']

    if include_unavailable:
        all_scores.append('unavailable')
        plot_colors.append('#E2E5E9')

    benchmarks_dict = {
        "REI": benchmarks_rei,
        "IREI": benchmarks_irei,
        "SD": scenarios,
        "ISD": scenarios, 
        "TR": benchmark_tr
    }

    benchmark_labels = {
        "tilt_sector": "Tilt Sector",
        "input_tilt_sector": "Input Tilt Sector",
        "ipr_1.5c rps_2030": "IPR 1.5 RPS 2030",
        "ipr_1.5c rps_2050": "IPR 1.5 RPS 2050",
        "weo_nz 2050_2030": "WEO NZE 2030",
        "weo_nz 2050_2050": "WEO NZE 2050", 
        "1.5c rps_2030_tilt_sector": "Tilt Sector &\nIPR 1.5 RPS 2030"
    }

    no_total_products = len(company_products)

    group = data[data['Indicator'] == indicator]
    indicator_benchmarks = benchmarks_dict[indicator]

    full_index = pd.MultiIndex.from_product(
        [indicator_benchmarks, all_scores],
        names=['benchmark', 'score']
    )

    # original score counts for the coverage 
    score_counts_v1 = group.groupby(['benchmark', 'score']).size().reindex(full_index, fill_value=0).unstack(fill_value=np.nan)
    number_of_na = no_total_products - score_counts_v1.sum(axis=1)
    no_available_products = no_total_products - number_of_na
    
    if not include_unavailable:
        score_counts_v1 = score_counts_v1.loc[(score_counts_v1 > 0).any(axis=1)]

    if include_unavailable:
        score_counts_v1['unavailable'] = number_of_na

    # Calculate weighted score counts using revenue_share
    group['weighted_revenue_share'] = group['revenue_share'] / group['revenue_share'].sum()
    score_counts_v2 = group.groupby(['benchmark', 'score'])['weighted_revenue_share'].sum().reindex(full_index, fill_value=0).unstack(fill_value=np.nan)
    number_of_na_v2 = no_total_products - score_counts_v2.sum(axis=1)
    no_available_products_v2 = no_total_products - number_of_na_v2

    if not include_unavailable:
        score_counts_v2 = score_counts_v2.loc[(score_counts_v2 > 0).any(axis=1)]

    if include_unavailable:
        score_counts_v2['unavailable'] = number_of_na_v2

    score_counts_v2 = score_counts_v2[all_scores].iloc[::-1]
    score_percentages = (score_counts_v2.T / no_available_products_v2 * 100).T

    if not score_percentages.empty and not score_percentages.isnull().all().all():
        score_percentages.plot(
            kind='barh',
            stacked=True,
            color=plot_colors,
            ax=ax,
            legend=False,
            width=0.3,
            edgecolor='none'
        )
        
        ax.set_yticklabels([benchmark_labels.get(label, label) for label in score_percentages.index])
    
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage", fontsize=12, color='black')
        ax.set_ylabel("Benchmark" if indicator in ["REI", "IREI"] else "Scenario", fontsize=12, color='black')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='y', rotation=0, colors='black', labelsize=10)
    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
        ax.xaxis.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.yaxis.grid(False)
        
        # Adjust space to account for the longest label
        plt.subplots_adjust(left=0.2 + max_label_length * 0.01)
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        combined_filename = f'figures/{indicator}_product_score_distribution_{company_id}.png'
        plt.savefig(combined_filename, bbox_inches='tight', dpi=300)
        #plt.show()
        
    plt.close()

    return no_total_products, no_available_products

def create_legend_image():
    # Colors corresponding to the scores
    plot_colors = ['#b3d15d', '#f8c646', '#f47b3e', '#E2E5E9']
    labels = ['Low', 'Medium', 'High', 'Not Available']

    # Create a dummy figure for the legend
    fig, ax = plt.subplots(figsize=(6, 0.2))
    handles = [plt.Line2D([0], [0], color=color, lw=6) for color in plot_colors]

    fig.legend(handles, labels, loc='center', ncol=4, fontsize=12, frameon=False, labelcolor='black')
    ax.axis('off')

    # Save the legend as a separate image
    legend_filename = 'figures/legend.png'
    plt.savefig(legend_filename, bbox_inches='tight', dpi=300)
    plt.close()

def describe_emission_rank(avg_rank, benchmark_label):
    if np.isnan(avg_rank):
        description = "The average relative emission intensity indicator is not available."
    else:
        description = f"The average relative emission score is {avg_rank:.2f}. "
    
    if avg_rank < 1/3:
        description += "This indicates that, on average, the firm's products have a <b>low</b> CO2e (kg/unit) emission intensity compared to others in the chosen reference group."
    elif avg_rank < 2/3:
        description += "This suggests that, on average, the firm's products have a <b>medium</b> CO2e (kg/unit) emission intensity compared to others in the chosen reference group."
    elif avg_rank > 2/3:
        description += "This indicates that, on average, the firm's products have a <b>high</b> CO2e (kg/unit) emission intensity compared to others in the chosen reference group."
        
    description += f" The chosen reference group is the <b>{benchmark_label}</b>."
    
    return description

def describe_upstream_emission_rank(avg_rank, benchmark_label):
    if np.isnan(avg_rank):
        description = "The average input relative emission intensity indicator is not available."
    else:
        description = f"The average input relative emission intensity indicator is {avg_rank:.2f}. "
    
    if avg_rank < 0.33:
        description += "This indicates that, on average, the firm's input products have a <b>low</b> CO2e (kg/unit) emission intensity compared to others in the chosen reference group."
    elif avg_rank < 0.67:
        description += "This suggests that, on average, the firm's input products have a <b>medium</b> CO2e (kg/unit) emission intensity relative to others in the chosen reference group."
    elif avg_rank > 0.67:
        description += "This indicates that, on average, the firm's input products have a <b>high</b> CO2e (kg/unit) emission intensity compared to others in the chosen reference group."
        
    description += f" The chosen reference group is the <b>{benchmark_label}</b>."
    
    return description

def describe_sector_rank(avg_rank, benchmark, benchmark_label):
    if np.isnan(avg_rank):
        description = "The average sector decarbonisation indicator is not available."
    else:
        description = f"The average sector decarbonisation indicator is {avg_rank:.2f}. "

    # Determine pressure level based on benchmark
    if benchmark in ['ipr_1.5c rps_2030', 'weo_nz 2050_2030']:
        if avg_rank < 1/9:
            pressure_level = 'low'
        elif avg_rank > 2/9:
            pressure_level = 'high'
        elif  1/9 < avg_rank < 2/9:
            pressure_level = 'medium'
        else:
            pressure_level = np.nan
    elif benchmark in ['ipr_1.5c rps_2050', 'weo_nz 2050_2050']:
        if avg_rank < 1/3:
            pressure_level = 'low'
        elif avg_rank > 2/3:
            pressure_level = 'high'
        elif  1/3 < avg_rank < 2/3:
            pressure_level = 'medium'
        else:
            pressure_level = np.nan
    
    if pressure_level == 'low':
        description += "This indicates that, on average, there is <b>low</b> pressure on the firm to reduce its CO2e (kg/unit) emissions."
    elif pressure_level == 'medium':
        description += "This suggests that, on average, there is <b>medium</b> pressure on the firm to reduce its CO2e (kg/unit) emissions."
    elif pressure_level == 'high':
        description += "This indicates that, on average, there is <b>high</b> pressure on the firm to reduce its CO2e (kg/unit) emissions."

    scenario_name = benchmark_label[:-5]  # Removes the last 5 characters (including space)
    scenario_year = benchmark_label[-4:]  # Gets the last 4 characters
    description += f" The chosen scenario is the <b>{scenario_name}</b> scenario with the target year <b>{scenario_year}</b>."
    
    return description
    
def describe_upstream_sector_rank(avg_rank, benchmark, benchmark_label):
    if np.isnan(avg_rank):
        description = "The average input sector decarbonisation indicator is not available."
    else:
        description = f"The average input sector decarbonisation indicator is {avg_rank:.2f}. "

    # Determine pressure level based on benchmark
    if benchmark in ['ipr_1.5c rps_2030', 'weo_nz 2050_2030']:
        if avg_rank < 1/9:
            pressure_level = 'low'
        elif avg_rank > 2/9:
            pressure_level = 'high'
        elif  1/9 < avg_rank < 2/9:
            pressure_level = 'medium'
        else:
            pressure_level = np.nan
    elif benchmark in ['ipr_1.5c rps_2050', 'weo_nz 2050_2050']:
        if avg_rank < 1/3:
            pressure_level = 'low'
        elif avg_rank > 2/3:
            pressure_level = 'high'
        elif  1/3 < avg_rank < 2/3:
            pressure_level = 'medium'
        else:
            pressure_level = np.nan
    
    if pressure_level == 'low':
        description += "This indicates that, on average, there is <b>low</b> pressure on the firm to reduce its CO2e (kg/unit) emissions based on the input products."
    elif pressure_level == 'medium':
        description += "This suggests that, on average, there is <b>medium</b> pressure on the firm to reduce its CO2e (kg/unit) emissions based on the input products."
    elif pressure_level == 'high':
        description += "This indicates that, on average, there is <b>high</b> pressure on the firm to reduce its CO2e (kg/unit) emissions based on the input products."

    scenario_name = benchmark_label[:-5]  # Removes the last 5 characters (including space)
    scenario_year = benchmark_label[-4:]  # Gets the last 4 characters
    description += f" The chosen scenario is the <b>{scenario_name}</b> scenario with the target year <b>{scenario_year}</b>."

    return description

def describe_transition_risk(avg_rank, benchmark, benchmark_label):
    if np.isnan(avg_rank):
        description = "The average transition risk indicator is not available."
    else:
        description = f"The average transition risk indicator is {avg_rank:.2f}. "

    # Determine pressure level based on benchmark
    if benchmark == '1.5c rps_2030_tilt_sector':
        if avg_rank < 0.32:
            pressure_level = 'low'
        elif avg_rank > 0.48:
            pressure_level = 'high'
        elif  0.32 < avg_rank < 0.48:
            pressure_level = 'medium'
        else:
            pressure_level = np.nan
    
    if pressure_level == 'low':
        description += "This indicates that, on average, the transition risk is <b>low</b> relative to other firms in terms of their hazards and exposures."
    elif pressure_level == 'medium':
        description += "This indicates that, on average, the transition risk is <b>medium</b> relative to other firms in terms of their hazards and exposures."
    elif pressure_level == 'high':
        description += "This indicates that, on average, the transition risk is <b>high</b> relative to other firms in terms of their hazards and exposures."

    parts = benchmark_label.split(" & ")
    reference_group = parts[0]
    scenario_part = parts[1]

    scenario_name = scenario_part[:-5]  # Removes the last 5 characters (including space)
    scenario_year = scenario_part[-4:]  # Gets the last 4 characters

    description += f" The chosen reference group is the <b>{reference_group}</b> and the chosen scenario is the <b>{scenario_name}</b> scenario with the target year <b>{scenario_year}</b>."

    return description

# for portfolio -> NOT SURE HOW TO FORMULATE
def describe_emission_rank_portfolio_level(avg_rank, benchmark_label):
    if np.isnan(avg_rank):
        description = "The average relative emission intensity indicator is not available."
    else:
        description = f"The average relative emission score is {avg_rank:.2f}. "
    
    if avg_rank < 1/3:
        description += "This indicates that, on average, the companies in the portfolio have a <b>low</b> CO2e (kg/unit) emission intensity compared to others in the chosen reference group."
    elif avg_rank < 2/3:
        description += "This suggests that, on average, the firm's products have a <b>medium</b> CO2e (kg/unit) emission intensity compared to others in the chosen reference group."
    elif avg_rank > 2/3:
        description += "This indicates that, on average, the firm's products have a <b>high</b> CO2e (kg/unit) emission intensity compared to others in the chosen reference group."
        
    description += f" The chosen reference group is the <b>{benchmark_label}</b>."
    
    return description

def describe_upstream_emission_rank_portfolio_level(avg_rank, benchmark_label):
    if np.isnan(avg_rank):
        description = "The average input relative emission intensity indicator is not available."
    else:
        description = f"The average input relative emission intensity indicator is {avg_rank:.2f}. "
    
    if avg_rank < 0.33:
        description += "This indicates that, on average, the firm's input products have a <b>low</b> CO2e (kg/unit) emission intensity compared to others in the chosen reference group."
    elif avg_rank < 0.67:
        description += "This suggests that, on average, the firm's input products have a <b>medium</b> CO2e (kg/unit) emission intensity relative to others in the chosen reference group."
    elif avg_rank > 0.67:
        description += "This indicates that, on average, the firm's input products have a <b>high</b> CO2e (kg/unit) emission intensity compared to others in the chosen reference group."
        
    description += f" The chosen reference group is the <b>{benchmark_label}</b>."
    
    return description

def describe_sector_rank_portfolio_level(avg_rank, benchmark, benchmark_label):
    if np.isnan(avg_rank):
        description = "The average sector decarbonisation indicator is not available."
    else:
        description = f"The average sector decarbonisation indicator is {avg_rank:.2f}. "

    # Determine pressure level based on benchmark
    if benchmark in ['ipr_1.5c rps_2030', 'weo_nz 2050_2030']:
        if avg_rank < 1/9:
            pressure_level = 'low'
        elif avg_rank > 2/9:
            pressure_level = 'high'
        elif  1/9 < avg_rank < 2/9:
            pressure_level = 'medium'
        else:
            pressure_level = np.nan
    elif benchmark in ['ipr_1.5c rps_2050', 'weo_nz 2050_2050']:
        if avg_rank < 1/3:
            pressure_level = 'low'
        elif avg_rank > 2/3:
            pressure_level = 'high'
        elif  1/3 < avg_rank < 2/3:
            pressure_level = 'medium'
        else:
            pressure_level = np.nan
    
    if pressure_level == 'low':
        description += "This indicates that, on average, there is <b>low</b> pressure on the firm to reduce its CO2e (kg/unit) emissions."
    elif pressure_level == 'medium':
        description += "This suggests that, on average, there is <b>medium</b> pressure on the firm to reduce its CO2e (kg/unit) emissions."
    elif pressure_level == 'high':
        description += "This indicates that, on average, there is <b>high</b> pressure on the firm to reduce its CO2e (kg/unit) emissions."

    scenario_name = benchmark_label[:-5]  # Removes the last 5 characters (including space)
    scenario_year = benchmark_label[-4:]  # Gets the last 4 characters
    description += f" The chosen scenario is the <b>{scenario_name}</b> scenario with the target year <b>{scenario_year}</b>."
    
    return description
    
def describe_upstream_sector_rank_portfolio_level(avg_rank, benchmark, benchmark_label):
    if np.isnan(avg_rank):
        description = "The average input sector decarbonisation indicator is not available."
    else:
        description = f"The average input sector decarbonisation indicator is {avg_rank:.2f}. "

    # Determine pressure level based on benchmark
    if benchmark in ['ipr_1.5c rps_2030', 'weo_nz 2050_2030']:
        if avg_rank < 1/9:
            pressure_level = 'low'
        elif avg_rank > 2/9:
            pressure_level = 'high'
        elif  1/9 < avg_rank < 2/9:
            pressure_level = 'medium'
        else:
            pressure_level = np.nan
    elif benchmark in ['ipr_1.5c rps_2050', 'weo_nz 2050_2050']:
        if avg_rank < 1/3:
            pressure_level = 'low'
        elif avg_rank > 2/3:
            pressure_level = 'high'
        elif  1/3 < avg_rank < 2/3:
            pressure_level = 'medium'
        else:
            pressure_level = np.nan
    
    if pressure_level == 'low':
        description += "This indicates that, on average, there is <b>low</b> pressure on the firm to reduce its CO2e (kg/unit) emissions based on the input products."
    elif pressure_level == 'medium':
        description += "This suggests that, on average, there is <b>medium</b> pressure on the firm to reduce its CO2e (kg/unit) emissions based on the input products."
    elif pressure_level == 'high':
        description += "This indicates that, on average, there is <b>high</b> pressure on the firm to reduce its CO2e (kg/unit) emissions based on the input products."

    scenario_name = benchmark_label[:-5]  # Removes the last 5 characters (including space)
    scenario_year = benchmark_label[-4:]  # Gets the last 4 characters
    description += f" The chosen scenario is the <b>{scenario_name}</b> scenario with the target year <b>{scenario_year}</b>."

    return description

def describe_transition_risk_portfolio_level(avg_rank, benchmark, benchmark_label):
    if np.isnan(avg_rank):
        description = "The average transition risk indicator is not available."
    else:
        description = f"The average transition risk indicator is {avg_rank:.2f}. "

    # Determine pressure level based on benchmark
    if benchmark == '1.5c rps_2030_tilt_sector':
        if avg_rank < 0.32:
            pressure_level = 'low'
        elif avg_rank > 0.48:
            pressure_level = 'high'
        elif  0.32 < avg_rank < 0.48:
            pressure_level = 'medium'
        else:
            pressure_level = np.nan
    
    if pressure_level == 'low':
        description += "This indicates that, on average, the transition risk is <b>low</b> relative to other firms in terms of their hazards and exposures."
    elif pressure_level == 'medium':
        description += "This indicates that, on average, the transition risk is <b>medium</b> relative to other firms in terms of their hazards and exposures."
    elif pressure_level == 'high':
        description += "This indicates that, on average, the transition risk is <b>high</b> relative to other firms in terms of their hazards and exposures."

    parts = benchmark_label.split(" & ")
    reference_group = parts[0]
    scenario_part = parts[1]

    scenario_name = scenario_part[:-5]  # Removes the last 5 characters (including space)
    scenario_year = scenario_part[-4:]  # Gets the last 4 characters

    description += f" The chosen reference group is the <b>{reference_group}</b> and the chosen scenario is the <b>{scenario_name}</b> scenario with the target year <b>{scenario_year}</b>."

    return description
