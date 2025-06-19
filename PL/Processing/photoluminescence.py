"""
Photoluminescence - PL
This module contains functions and classes for performing PL data processing.
"""
import os
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def plot_collage(filepath, radius, edge_exclusion, substrate=None):
    """
    Mapping of the expected Si mapping
    """

    filename_csv = None
    cmap2 = None
    filename_png = None
    if substrate == "Top":
        filename_csv = "Top_grid_df.csv"
        filename_png = 'TS_transfert.png'
        cmap2 = mcolors.ListedColormap(['blue', 'green'])
    if substrate == 'Base':
        filename_csv = "Base_grid_df.csv"
        filename_png = 'GS_transfert.png'
        cmap2 = mcolors.ListedColormap(['red', 'blue'])
    try:
        # Load the data from the file
        data_frame = pd.read_csv(filepath)

        # Set all non-NaN values in data columns to 0
        data_frame.iloc[:, 2:] = data_frame.iloc[:, 2:].applymap(
            lambda x: 0 if pd.notna(x) else x)

        # Replace all NaN values with 1
        data_frame = data_frame.fillna(1)

        # Get the directory containing the file
        wafer_nbr = os.path.dirname(filepath)

        # Create a grid for the column
        grid_df = data_frame.pivot_table(index='Y',
                                         columns='X',
                                         values="Area (cps)")

        x_min, x_max = grid_df.columns[0], grid_df.columns[-1]
        y_min, y_max = grid_df.index[0], grid_df.index[-1]

        for i, x in enumerate(grid_df.columns):
            for j, y in enumerate(grid_df.index):
                x = float(x)
                y = float(y)
                if x ** 2 + y ** 2 > (radius - edge_exclusion) ** 2:
                    grid_df.iloc[j, i] = np.nan

        # Save the grid to a CSV file
        grid_csv_path = os.path.join(wafer_nbr, filename_csv)
        grid_df.to_csv(grid_csv_path)

        fig, ax = plt.subplots()
        plt.imshow(grid_df,
                   cmap=cmap2,
                   extent=[x_min, x_max, y_max, y_min], vmin=0, vmax=1)
        plt.xlabel('X (cm)', fontsize=20)
        plt.ylabel('Y (cm)', fontsize=20)

        if substrate == "Top":
            plt.text(-1, 10.5, "MoS\u2082", fontsize=20, ha='right',
                     color='blue')
            plt.text(2, 10.5, "and", fontsize=20, ha='right', color='black')
            plt.text(4, 10.5, "Si", fontsize=20, ha='right', color='green')
        if substrate == 'Base':
            plt.text(-1, 10.5, "MoS\u2082", fontsize=20,
                     ha='right', color='blue')
            plt.text(2, 10.5, "and", fontsize=20, ha='right', color='black')
            plt.text(6, 10.5, "SiO2", fontsize=20, ha='right', color='red')

        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{int(x)}'))

        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f'{int(y)}'))

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        circle = plt.Circle((0, 0), radius - edge_exclusion,
                            color='black', fill=False, linewidth=0.5)
        plt.gca().add_patch(circle)
        plt.xlim(-radius, radius)
        plt.ylim(-radius, radius)

        parent_dir = os.path.dirname(os.path.dirname(filepath))
        list_data_dir = os.path.join(parent_dir, "Liste_data")
        rates_file = os.path.join(list_data_dir, "rates.csv")

        if os.path.exists(rates_file):
            rates_data = pd.read_csv(rates_file)

            print(rates_data)
            filtered_data = rates_data[
                (rates_data['Slot'] == float(os.path.basename(wafer_nbr)))]

            print(filtered_data)

            if not filtered_data.empty:
                # Extract the mean and 3sigma values
                rate_value = filtered_data['Cov. rate'].values[0]

                print(rate_value)

                # Add text for mean and 3sigma
                ax.text(0.01, 0.01, f'Cov. : {rate_value:.1f} %',
                        transform=ax.transAxes, fontsize=14,
                        ha='left', va='bottom', color='black')

        # Save the secondary plot
        output_png = os.path.join(wafer_nbr, filename_png)
        plt.savefig(output_png, bbox_inches='tight')
        plt.close('all')

    except Exception as error:
        print(f"Error processing file {filepath}: {error}")


def plot_wafer_data(filepath, radius, edge_exclusion, slot_number, threshold,
                    identical=None, stats=None):
    """
    Generates intensity and energy mappings from the given data file.
    """
    vmin = None
    vmax = None
    try:
        # Load data from the file
        data_frame = pd.read_csv(filepath)
        subdir = os.path.dirname(filepath)

        # Process relevant columns
        for column in data_frame.columns[2:]:
            if column not in ["Intensity (cps)", "Energy (eV)", "Area (cps)"]:
                continue
            # if column not in ["Intensity (cps)", "Energy (eV)"]:
            #     continue

            # Create a grid for the column
            grid_df = data_frame.pivot_table(index='Y', columns='X',
                                             values=column, dropna=False)

            # Set output file names based on the column
            filename_grid = f"{column.split()[0]}_grid_df.csv"
            filename_png = f"{column.split()[0]}.png"

            # Save the grid to a CSV file
            grid_csv_path = os.path.join(subdir, filename_grid)
            grid_df.to_csv(grid_csv_path)

            # Define x and y limits
            x_min, x_max = float(grid_df.columns[0]), float(grid_df.columns[-1])
            y_min, y_max = float(grid_df.index[0]), float(grid_df.index[-1])

            print(
                f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")

            for i, x in enumerate(grid_df.columns):
                for j, y in enumerate(grid_df.index):
                    x = float(x)
                    y = float(y)
                    if x ** 2 + y ** 2 > (radius - edge_exclusion) ** 2:
                        grid_df.iloc[j, i] = np.nan

            if identical:
                vmin, vmax = get_vmin_vmax(data_frame, column, threshold,
                                           identical, subdir)
                filename_png = f"{column.split()[0]}_ID_scale.png"

            # Generate the heatmap
            fig, ax = plt.subplots()

            if identical:
                img = ax.imshow(grid_df,
                                extent=(x_min, x_max, y_min, y_max),
                                origin='lower', cmap='Spectral_r',
                                vmin=vmin, vmax=vmax)
            else:
                img = ax.imshow(grid_df,
                                extent=(x_min, x_max, y_min, y_max),
                                origin='lower', cmap='Spectral_r')

            # Add colorbar and labels
            cbar = plt.colorbar(img, ax=ax, shrink=1, aspect=10)
            cbar.ax.tick_params(labelsize=18)
            plt.xlabel('X (cm)', fontsize=24)
            plt.ylabel('Y (cm)', fontsize=24)
            title = f"S{os.path.basename(subdir)} - {column}" if slot_number \
                else column
            # plt.title(title, fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            # Add a black circle indicating the valid region
            circle = plt.Circle((0, 0), radius - edge_exclusion, color='black',
                                fill=False, linewidth=0.5)
            plt.gca().add_patch(circle)
            plt.xlim(-radius, radius)
            plt.ylim(-radius, radius)

            # Adjust tick formatting
            ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda y, _: f'{int(y)}'))

            # Add statistics if available
            if stats:
                add_statistics(ax, subdir, column)

            # Save the heatmap
            heatmap_path = os.path.join(subdir, filename_png)
            plt.savefig(heatmap_path, bbox_inches='tight')
            plt.close()

    except Exception as error:
        print(f"Error processing file {filepath}: {error}")


def get_vmin_vmax(data_frame, column, threshold, identical, subdir):
    """Helper function to determine vmin and vmax values."""
    if identical == 'Manual':
        idx = 0 if column == 'Intensity (cps)' else 2
        vmin, vmax = threshold[idx], threshold[idx + 1]
        return float(vmin) if vmin else data_frame[column].min(), float(
            vmax) if vmax else data_frame[column].max()

    elif identical == 'Auto':
        parent_dir = os.path.dirname(subdir)
        list_data_dir = os.path.join(parent_dir, "Liste_data")
        filename = 'Boxplot_energy.csv' if column == 'Energy (eV)' else \
            'Boxplot_intensity.csv'
        boxplot_file = os.path.join(list_data_dir, filename)

        if os.path.exists(boxplot_file):
            boxplot_data = pd.read_csv(boxplot_file, header=0, index_col=0)
            vmin, vmax = boxplot_data.min(
                numeric_only=True).min(), boxplot_data.max(
                numeric_only=True).max()
            print(f"Auto scaling for {column}: vmin={vmin}, vmax={vmax}")
            return vmin, vmax
        else:
            print(
                f"Boxplot file not found: {boxplot_file}. Using default "
                f"scaling.")
            return data_frame[column].min(), data_frame[column].max()

    # Default scaling
    return data_frame[column].min(), data_frame[column].max()


def add_statistics(ax, subdir, column):
    """Helper function to add statistics to the plot."""
    parent_dir = os.path.dirname(subdir)
    wafer_digit = os.path.basename(subdir)
    list_data_dir = os.path.join(parent_dir, "Liste_data")
    stats_file = os.path.join(list_data_dir, "Stats.csv")

    if os.path.exists(stats_file):
        stats_data = pd.read_csv(stats_file)
        filtered_data = stats_data[
            (stats_data['Slot'] == float(wafer_digit)) & (
                    stats_data['Parameters'] == str(column))]
        if not filtered_data.empty:
            mean_val = filtered_data['mean'].values[0]
            sigma_val = filtered_data['sigma'].values[0]
            uniformity = (1-sigma_val / mean_val) * 100 if mean_val != 0 else 0

            uniformity = max(uniformity, 0)

            # Add text for mean, 3sigma, and uniformity
            ax.text(0.01, 0.01, f'Mean: {mean_val:.2f}',
                    transform=ax.transAxes, fontsize=14, ha='left', va='bottom',
                    color='black')
            ax.text(0.96, 0.01, f'$\sigma$: {sigma_val:.2f}',
                    transform=ax.transAxes, fontsize=14, ha='right',
                    va='bottom', color='black')
            ax.text(0.96, 0.96, f'U: {uniformity:.1f}%', transform=ax.transAxes,
                    fontsize=14, ha='right', va='top', color='black')


class Photoluminescence:
    """
    Photoluminescence class
    This class includes functions for fitting and plotting PL spectra.
    """

    def __init__(self, dirname, wafer_size, edge_exclusion, step,
                 int_min=None, int_max=None, ev_min=None, ev_max=None,
                 subdir=None, model=None):
        self.dirname = dirname
        self.wafer_size = wafer_size
        self.edge_exclusion = edge_exclusion
        self.step = step
        self.int_min = int_min
        self.int_max = int_max
        self.ev_min = ev_min
        self.ev_max = ev_max
        self.radius = self.wafer_size / 2
        self.subdir = [str(x).strip().lower() for x in subdir]
        self.dirname_model = model

        print(self.wafer_size, self.step)

    def find_xy(self, pix_1, pix_2):
        """
        Function to find Y max values and the associated X values
        """

        path_liste = self.dirname + '\\Liste_data'
        if not os.path.exists(path_liste):
            os.makedirs(path_liste)

        rates = []
        subdirs = []
        for subdir, _, files in os.walk(self.dirname):
            if os.path.basename(subdir) in self.subdir:
                for file in files:
                    filepath = os.path.join(subdir, file)
                    if filepath.endswith("data.csv"):
                        raw_data = pd.read_csv(filepath, sep='\t', header=None)
                        rows, columns = raw_data.shape

                        if columns < 100:
                            return True

                        # Extract X and Y spectra data for the given pixel range
                        spectrum_x = raw_data.iloc[0, pix_1:pix_2].to_numpy()
                        spectrum_y = raw_data.iloc[1:, pix_1:pix_2].to_numpy()

                        print(spectrum_x)

                        # Calculate X and Y positions (scaled by step value)
                        xpos = raw_data.iloc[1:, 0].to_numpy() * self.step
                        ypos = raw_data.iloc[1:, 1].to_numpy() * self.step
                        if spectrum_x[1] > 3:
                            print("No 2D spectrum")
                            return True

                        # Get the maximum intensity and
                        # corresponding energy for each Y spectrum
                        max_y = np.max(spectrum_y, axis=1)
                        x_associe = spectrum_x[np.argmax(spectrum_y, axis=1)]

                        # Create a DataFrame to store the results
                        data = {"X": xpos, "Y": ypos,
                                "Intensity (cps)": max_y,
                                "Energy (eV)": x_associe}
                        output = pd.DataFrame(data)

                        # Save the DataFrame to 'data_e_i.csv'
                        output.to_csv(os.path.join(subdir, "data_e_i.csv"),
                                      index=False)

                        # Define radius excluding edge region
                        r_squared = (self.radius - self.edge_exclusion) ** 2

                        # Create a mask for filtering out data based on
                        # intensity,
                        # energy, and position
                        mask = (
                                (output['Intensity (cps)'] <= self.int_min) |
                                (output['Intensity (cps)'] >= self.int_max) |
                                (output['Energy (eV)'] <= self.ev_min) |
                                (output['Energy (eV)'] >= self.ev_max))

                        # Apply the mask and set filtered values to NaN
                        output.loc[mask, ['Intensity (cps)',
                                          'Energy (eV)']] = np.nan

                        # Calculate the percentage of data that
                        # is not filtered (non-NaN)
                        rates.append(output['Intensity (cps)'].count() /
                                     len(raw_data) * 100)
                        subdirs.append(os.path.basename(subdir))

                        mask = (output['X'] ** 2 + output[
                            'Y'] ** 2) >= r_squared

                        # Apply the mask and set filtered values to NaN
                        output.loc[mask, ['Intensity (cps)',
                                          'Energy (eV)']] = np.nan

                        # Save the filtered data to 'data_filtered.csv'
                        output.to_csv(os.path.join(subdir, "data_filtered.csv"),
                                      index=False)

            # Create a DataFrame to store the results
            data_rate = {"Slot": subdirs, "Cov. rate": rates}
            data_rate_df = pd.DataFrame(data_rate)
            # Save the DataFrame to 'data_e_i.csv'
            data_rate_df.to_csv(os.path.join(path_liste, "rate.csv"),
                                index=False)


    def plot(self, slot_number=None, identical=None, stats=None):
        """
        Search for filtered data files and process them using multiprocessing.
        """
        filepaths = []
        threshold = [self.int_min, self.int_max, self.ev_min, self.ev_max]

        for subdir, _, files in os.walk(self.dirname):
            if os.path.basename(subdir) in self.subdir:
                for file in files:
                    if file.endswith("data_filtered.csv"):
                        filepaths.append(os.path.join(subdir, file))
        
        print(filepaths)

        process_partial = partial(plot_wafer_data,
                                  radius=self.radius,
                                  edge_exclusion=self.edge_exclusion,
                                  slot_number=slot_number,
                                  threshold=threshold,
                                  identical=identical,
                                  stats=stats)

        num_cpu_cores = os.cpu_count()
        max_workers = max(1, num_cpu_cores // 2)

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process_partial, filepaths)

    def collage(self, substrate=None):
        """
        Search for Si-related data files and process them using multiprocessing.
        """
        filepaths = []
        for subdir, _, files in os.walk(self.dirname):
            if os.path.basename(subdir) in self.subdir:
                for file in files:
                    if file.endswith("data_filtered.csv"):
                        filepaths.append(os.path.join(subdir, file))

        print(filepaths)

        process_partial = partial(plot_collage,
                                  radius=self.radius,
                                  edge_exclusion=self.edge_exclusion,
                                  substrate=substrate)

        num_cpu_cores = os.cpu_count()
        max_workers = max(1, num_cpu_cores // 2)

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process_partial, filepaths)

    def plot_boxplot(self):
        """
        Plot the energy (eV) of the main peak, (X axis = Wafers)
        """

        filename_csv = ['Boxplot_intensity.csv', 'Boxplot_energy.csv']
        filename_png = ['Intensity.png', 'Energy.png']
        ylabel = ['Intensity (cps)', 'Energy (eV)']

        path = self.dirname + os.sep + 'Liste_data'
        if not os.path.exists(path):
            os.makedirs(path)

        path2 = self.dirname + os.sep + "Graphe" + os.sep + "Boxplot"
        if not os.path.exists(path2):
            os.makedirs(path2)

        columns_data = {}
        i = 0
        for j in range(2):
            for subdir, _, files in os.walk(self.dirname):
                if subdir != self.dirname:
                    for file in files:
                        filepath = subdir + os.sep + file
                        if filepath.endswith("data_filtered.csv"):
                            data_frame_e_i = pd.read_csv(filepath, sep=',')
                            print(data_frame_e_i)
                            nom_colonne = os.path.basename(subdir)
                            columns_data[nom_colonne] = data_frame_e_i[
                                ylabel[j]]

            df_merged = pd.DataFrame(columns_data)
            print(df_merged)

            df_merged.reset_index(drop=True, inplace=True)

            df_merged = df_merged.sort_index(axis=1,
                                             key=lambda x: x.astype(int))
            noms_colonnes_sorted = df_merged.columns.tolist()
            df_merged = df_merged.round(3)

            data_for_boxplot = [df_merged[col].dropna().values for col in
                                df_merged.columns]
            print(data_for_boxplot)

            data_for_boxplot_df = pd.DataFrame({
                col: pd.Series(data) for col, data in
                zip(df_merged.columns, data_for_boxplot)
            })

            data_for_boxplot_df.to_csv(
                self.dirname + os.sep + "Liste_data" + os.sep +
                filename_csv[j],
                index=False
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.tick_params(axis='both', which='major', labelsize=15)

            ax.boxplot(data_for_boxplot, showfliers=False)
            ax.set_xticklabels(noms_colonnes_sorted)
            ax.set_xlabel('Wafer', fontsize=26)
            ax.set_ylabel(ylabel[j], fontsize=26)

            plt.savefig(
                self.dirname + os.sep + "Graphe" + os.sep + "Boxplot" +
                os.sep + filename_png[j], bbox_inches='tight')
            plt.close()
            plt.cla()
            i += 1

    def find_siarea(self, ref_file=None, substrate=None):
        """
        Soustrait la colonne "Area (cps)" du fichier de référence à la colonne "Area (cps)"
        des fichiers "data_filtered.csv" trouvés dans le dossier parent.

        :param ref_file: Chemin du fichier de référence CSV
        :param parent_dir: Dossier parent contenant les fichiers "data_filtered.csv"
        :param output_filename: Nom du fichier de sortie contenant les données corrigées
        :param sep: Séparateur utilisé dans les CSV ("\t" pour tabulation, "," pour virgule)
        """
        # Charger le fichier de référence


        path_liste = self.dirname + '\\Liste_data'
        if not os.path.exists(path_liste):
            os.makedirs(path_liste)

        r_squared = (self.radius - self.edge_exclusion) ** 2


        for subdir, _, files in os.walk(self.dirname):
            rates = []
            if os.path.basename(subdir) in self.subdir:
                for file in files:
                    filepath = os.path.join(subdir, file)
                    print(filepath)
                    if filepath.endswith("data.csv"):

                        raw_data = pd.read_csv(filepath, sep='\t', header=None)
                        columns = raw_data.shape[1]

                        if columns < 100:
                            return True

                        # Extract X and Y spectra data
                        spectrum_x = raw_data.iloc[0, 2:].to_numpy()
                        spectrum_y = raw_data.iloc[1:, 2:].to_numpy()

                        # Calculate X and Y positions (scaled by step value)
                        xpos = raw_data.iloc[1:, 0].to_numpy() * self.step
                        ypos = raw_data.iloc[1:, 1].to_numpy() * self.step

                        area = np.trapezoid(spectrum_y, spectrum_x, axis=1)

                        data = {"X": xpos, "Y": ypos, "Area (cps)": area}
                        output = pd.DataFrame(data)

                        # Save the DataFrame to 'data_e_i.csv'
                        output.to_csv(os.path.join(subdir, "data_area.csv"),
                                      index=False)
                        
                        grid_df = output.pivot_table(index='Y', columns='X',
                                             values="Area (cps)", dropna=False)

                        # Define x and y limits
                        x_min, x_max = float(grid_df.columns[0]), float(grid_df.columns[-1])
                        y_min, y_max = float(grid_df.index[0]), float(grid_df.index[-1])

                        print(
                            f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
                        
                        # Generate the heatmap
                        fig, ax = plt.subplots()

                        img = ax.imshow(grid_df, extent=(x_min, x_max, y_min, y_max),
                                            origin='lower', cmap='Spectral_r')

                        # Add colorbar and labels
                        cbar = plt.colorbar(img, ax=ax, shrink=1, aspect=10)
                        cbar.ax.tick_params(labelsize=18)
                        plt.xlabel('X (cm)', fontsize=24)
                        plt.ylabel('Y (cm)', fontsize=24)
                        title = "Si area raw (cps)"
                        plt.title(title, fontsize=20)
                        plt.xticks(fontsize=20)
                        plt.yticks(fontsize=20)

                        # Add a black circle indicating the valid region
                        circle = plt.Circle((0, 0), 10, color='black',
                                            fill=False, linewidth=0.5)
                        plt.gca().add_patch(circle)
                        plt.xlim(-self.radius, self.radius)
                        plt.ylim(-self.radius, self.radius)

                        # Adjust tick formatting
                        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
                        ax.xaxis.set_major_formatter(
                            mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
                        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
                        ax.yaxis.set_major_formatter(
                            mticker.FuncFormatter(lambda y, _: f'{int(y)}'))
                        # Save the heatmap
                        heatmap_path = os.path.join(subdir, 'Raw_area.png')
                        plt.savefig(heatmap_path, bbox_inches='tight')
                        plt.close()


                        if ref_file:
                            ref_df = pd.read_csv(ref_file,sep=',', header=0)                 
                        else :
                            return

                        # Vérifier que la colonne "Area (cps)" est présente
                        if "Area (cps)" not in ref_df.columns:
                            raise ValueError(
                                f"Le fichier de référence {ref_file} ne contient pas la colonne 'Area (cps)'.")
                        # Vérifier la correspondance des colonnes et dimensions
                        if "Area (cps)" in output.columns and len(ref_df) == len(
                                output):

                            output["Area (cps)"] -= ref_df[
                                "Area (cps)"]

                            mask = (
                                    (output['Area (cps)'] <= self.int_min) |
                                    (output['Area (cps)'] >= self.int_max))

                            output.loc[mask, ['Area (cps)']] = np.nan

                            if substrate == 'Base':
                                rates.append(100 - (
                                        output['Area (cps)'].count() / len(
                                    output) * 100))
                            elif substrate == 'Top':
                                rates.append(output['Area (cps)'].count() / len(
                                    output) * 100)

                            mask = (output['X'] ** 2 + output[
                                'Y'] ** 2) >= r_squared

                            # Apply the mask and set filtered values to NaN
                            output.loc[mask, ['Area (cps)']] = np.nan

                            # Save the filtered data to 'data_filtered.csv'
                            output.to_csv(
                                os.path.join(subdir, "data_filtered.csv"),
                                index=False)
                        else:
                            print(
                                f"Problème de correspondance dans {file_path} (colonnes manquantes ou dimensions différentes).")

                    # Create a DataFrame to store the coverage rate results
                    data_rate = {"Slot": os.path.basename(subdir),
                                 "Cov. rate": rates}
                    data_rate_df = pd.DataFrame(data_rate)
                    # Save the DataFrame to 'rate.csv'
                    data_rate_df.to_csv(os.path.join(subdir, "rate.csv"),
                                        index=False)

        all_data = []
        for subdir, _, files in os.walk(self.dirname):
            for file in files:
                if file.endswith("rate.csv"):
                    filepath = subdir + os.sep + file
                    df_rates = pd.read_csv(filepath)
                    all_data.append(df_rates)

        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)
            merged_df.iloc[:, 0] = merged_df.iloc[:, 0].astype(int)
            merged_df = merged_df.sort_values(by="Slot", ascending=True)

            merged_df.to_csv(os.path.join(path_liste, "rates.csv"),
                             index=False)
        else:
            print("No 'rate.csv' found.")

    def find_siarea_ref(self):
        for subdir, _, files in os.walk(self.dirname):
            rates = []
            if os.path.basename(subdir) in self.subdir:
                for file in files:
                    filepath = os.path.join(subdir, file)
                    print(filepath)
                    if filepath.endswith("data.csv"):
                        raw_data = pd.read_csv(filepath, sep='\t', header=None)
                        columns = raw_data.shape[1]

                        if columns < 100:
                            return True

                        # Extract X and Y spectra data
                        spectrum_x = raw_data.iloc[0, 2:].to_numpy()
                        spectrum_y = raw_data.iloc[1:, 2:].to_numpy()

                        # Calculate X and Y positions (scaled by step value)
                        xpos = raw_data.iloc[1:, 0].to_numpy() * self.step
                        ypos = raw_data.iloc[1:, 1].to_numpy() * self.step

                        # Compute area
                        area = np.trapezoid(spectrum_y, spectrum_x, axis=1)

                        # Create output DataFrame
                        data = {"X": xpos, "Y": ypos, "Area (cps)": area}
                        output = pd.DataFrame(data)

                        mask = ((output['Area (cps)'] <= self.int_min) |    
                                (output['Area (cps)'] >= self.int_max))

                        output.loc[mask, ['Area (cps)']] = np.nan

                        # Save the DataFrame to 'data_area.csv'
                        output.to_csv(os.path.join(subdir, "data_filtered.csv"),
                                    index=False)

    def delete_csv_files_with_prefix(self):
       
        prefix = "data.csv  X="
        for dirpath, dirnames, filenames in os.walk(self.dirname):
            for filename in filenames:
                if filename.endswith(".csv") and filename.startswith(prefix):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    DIRNAME = r"C:\Users\TM273821\Desktop\PL\250307-1856-Si - Copie"
    # SUBDIR = [3,5,7,9,11,13,15,17,19,21,23,25]
    MODEL=None
    SUBDIR = [6]
    ref_dir=r'C:\Users\TM273821\Desktop\PL\Ref_bulk_Si - Copie\2'
    ref_file=r'C:\Users\TM273821\Desktop\PL\Ref_bulk_Si - Copie\2\data_area.csv'
    # ref_file=None
    start_time = time.time()

    PL = Photoluminescence(DIRNAME, 20, 0, 0.2,
                           int_min=5000, int_max=154000,
                           ev_min=1.81, ev_max=1.89, subdir=SUBDIR, model=MODEL)
    # PL.nm_to_ev()
    # PL.find_siarea(substrate = 'Top')
    # PL.find_xy(300,375)
    # PL.find_siarea(ref_file=ref_file, substrate='Top')
    # # PL.find_siarea_ref(ref_dir=ref_dir)
    PL.collage(substrate = 'Top')
    # PL.plot(slot_number=True, identical=None, stats=True)
    # PL.collage()
    # PL.delete_csv_files_with_prefix()
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Time : {elapsed_time:.2f} s")
