#!/usr/bin/env python3.12
# coding=utf-8

# author:xkrejce00

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile


def load_data(filename : str, ds : str) -> pd.DataFrame:
    """
    Function searches a zip file for 'ds' file and converts to dataframe.

    Arguments:
    filename -- name of the zip file
    ds -- part of the name of the file name function is looking for
    """

    # Should be two
    data_frames = []

    with zipfile.ZipFile(filename, 'r') as zp:
        for file in zp.namelist():
            if ds in file:
                with zp.open(file) as f:
                    df = pd.read_html(f, encoding="cp1250")[0]
                    # Delete column without name(is Nan)
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

                    data_frames.append(df)

    return pd.concat(data_frames)

def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Function takes dataframe and adds new columns(region,date).

    Arguments:
    df -- dataframe to be parsed
    verbose -- print memory usage
    """

    # List of regions and their numbers in data
    region_table = {0: "PHA", 1: "STC", 2: "JHC", 3: "PLK", 4: "ULK", 5: "HKK",
                    6:"JHM", 7: "MSK", 14: "OLK", 15: "ZLK", 16: "VYS", 17: "PAK",
                    18: "LBK", 19: "KVK"}
    
    # Create a new dataframe and copy the data
    new_df = df.copy()

    # Convert into datetime format, if not a date -> NaT
    new_df['date'] = pd.to_datetime(df['p2a'], errors='coerce')

    # Replace values and rename the column
    new_df['p4a'] = new_df['p4a'].map(region_table)
    new_df = new_df.rename(columns={'p4a': 'region'})

    # Throw away rows with duplicated value in col p1
    new_df = new_df.drop_duplicates(subset='p1')
   
    if verbose:
        size = df.memory_usage(deep=True).sum()
        print(f"new_size={round(size/1000000,1)} MB")

    return new_df


def plot_state(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):
    """
    Function takes datafram and plots the number of accidents 
    by road state in regions.

    Arguments:
    df -- dataframe to be plotted
    fig_location -- location to save the figure
    show_figure -- show the figure
    """

    # List of different road states
    road_states = {
        1: "suchá_čistá", 
        2: "suchá_znečištěná", 
        3: "mokrá", 
        4: "zablácená", 
        5: "zasněžená_posolená", 
        6: "zasněžená_neposolená",
    }
    # New column
    df["road_state"] = df["p16"].map(road_states)

    # Divide data into groups ((1,2>,3>,4>,(5,6>),including right val
    df["road_group"] = pd.cut(df["p16"],bins=[0, 2, 3, 4, 6],
        labels=["suchá", "mokrá", "zablácená", "zasněžená"],
        right=True
    )

    # Create a new dataframe with needed data
    grouped_df = df.groupby(["region", "road_group"], observed=False).size().reset_index(name="accidents")

    # Start plotting
    sns.set_style(style="ticks")
    sns.set_palette("tab10")
    
    g = sns.catplot(
        x="region", y="accidents",
        data=grouped_df, 
        hue="road_group", kind="bar", 
        col="road_group", col_wrap=2, 
        sharex=False, sharey=False,
        palette="tab10", legend=False,
    )

    # Set titles and labels
    g.set_titles("Vozovka byla {col_name}")
    g.set_axis_labels("Kraj", "Počet nehod")
    g.figure.suptitle("Počty nehod podle povrchu vozovky", fontsize=18)
    g.figure.tight_layout()
    
    if fig_location:
        g.figure.savefig(fig_location)
    if show_figure:
        plt.show()


# Ukol4: alkohol a následky v krajích
def plot_alcohol(df: pd.DataFrame, df_consequences : pd.DataFrame, 
                 fig_location: str = None, show_figure: bool = False):
    """
    Function plots merges dataframs and plots 
    
    Arguments:
    df -- dataframe to be plotted
    df_consequences -- dataframe with consequences
    fig_location -- location to save the figure
    show_figure -- show the figure
    """

    injury_type = {1 : "usmrcení", 2 : "těžké zranění", 
                   3 : "lehké zranění", 4 : "bez zranění"}
    
    # Merge accidents and consequences
    merged_df = pd.merge(df, df_consequences, on="p1", how="inner")

    # Add new column
    merged_df["injury_type"] = merged_df["p59g"].map(injury_type)

    # Choose only accidents containing alcohol
    merged_df = merged_df[(merged_df["p11"] >= 3)]

    # Sort into Řidič (==1) and Spolujezdec
    merged_df["injured"] = merged_df["p59a"].apply(lambda x: "Řidič" if x == 1 else "Spolujezdec")

    # New dataframe with needed values
    final_df = (merged_df.groupby(["region", "injured", "injury_type"])
        .size().reset_index(name="count"))
    
    # Plotting
    g = sns.catplot(
        x="region",y="count",
        hue="injured",
        col="injury_type",
        kind="bar", data=final_df,
        col_wrap=2,
        sharex=False,sharey=False,
        palette="tab10",legend=True,
    )
    
    #adjust the visialization
    g.set_titles("Typ zranění: {col_name}")
    g.set_axis_labels("Kraj", "Počet nehod")

    g.figure.suptitle("Následky nehod pod vlivem alkoholu", fontsize=18)
    g.figure.tight_layout()
    
    sns.move_legend(g, "upper right",title="Osoba")
    
    if fig_location:
        g.savefig(fig_location)
    if show_figure:
        plt.show()

def plot_type(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Function to plot the number of accidents by type and region over time.
    
    Arguments:
    df -- dataframe containing accident data
    fig_location -- file path to save the plot
    show_figure -- if True, display the plot
    """
    # Selected regions and accident type mapping
    selected_regions = ["PHA", "MSK", "OLK", "ZLK"]
    accident = {
        1: "s nekolejovým vozidlem",
        2: "se zaparkovaným/odstaveným vozidlem",
        3: "s pevnou překážkou",
        4: "s chodcem",
        5: "s lesní zvěří",
        6: "s domácím zvířetem",
        7: "s vlakem",
        8: "s tramvají",
        9: "s havárií",
        10: "ostatní"
    }

    # Transforn data
    df["accident"] = df["p6"].map(accident)
    df["date"] = pd.to_datetime(df["p2a"], dayfirst=True)

    # Get only data for selected regions
    data = df[df["region"].isin(selected_regions)]

    # New dataframe with needed values
    # region|date|certain accident|...
    pivot_df = pd.pivot_table(
        data,index=["region", "date"],
        columns="accident",
        values="p1",aggfunc="count",
    )

    # Get the sum of values by the end of each month
    result_df = []
    for region,region_data in pivot_df.groupby(level="region"):
        # sum values for each type of accident
        stack_df = region_data.resample("ME", level="date").sum().stack().reset_index(name="accident_count")
        stack_df["region"] = region #add region
        result_df.append(stack_df)

    # Stacked dataframe, join list of dataframes
    # date(last of month)| accident| accident_count| region
    result_df = pd.concat(result_df)

    # Start plotting
    sns.set_style("ticks")
    g = sns.relplot(
        data=result_df,
        x="date", y="accident_count",
        hue="accident",col="region",
        col_wrap=2,kind="line",
        palette="tab10",
    )

    # Adjust labels, titles, legends
    g.set_titles("Kraj: {col_name}")
    g.set_xlabels("Měsíc")
    g.set_ylabels("Počet nehod")
    g.legend.set(title="Druh nehody")
    g.figure.suptitle("Druhy nehod ve vybraných krajích", fontsize=18)

    # set the date from 1.1.2023 to 1.10.2024
    x = g.axes.flat[0]
    x.set_xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-10-01"))
    xticks = pd.date_range("2023-01-01", "2024-10-01", freq="MS")
    x.set(xticks=xticks)
    g.set_xticklabels(xticks.strftime("%m/%Y"), rotation=45, ha="right")
    g.tight_layout()

    if fig_location:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()


if __name__ == "__main__":

    df = load_data("data_23_24.zip", "nehody")
    df_consequences = load_data("data_23_24.zip", "nasledky")
    df2 = parse_data(df, True)
    plot_state(df2, "01_state.png")
    plot_alcohol(df2, df_consequences, "02_alcohol.png")
    plot_type(df2, "03_type.png",True)
