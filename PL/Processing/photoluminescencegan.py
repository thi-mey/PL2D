"""
Photoluminescence - PL

This module contains functions and classes for performing PL spectra fitting.
"""

import os
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


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
            if column not in ["Intensity (cps)", "Energy (eV)",
                              "Laser intensity (cps)",
                              "NBE area (cps)", "YB area (cps)",
                              "YB/NBE area ratio"]:
                continue

            print(column)

            # Create a grid for the column
            grid_df = data_frame.pivot_table(index='Y', columns='X',
                                             values=column, dropna=False)

            # Set output file names based on the column
            filename_grid = f"{column.split()[0]}_grid_df.csv"
            filename_png = f"{column.split()[0]}.png"
            if column == "YB/NBE area ratio":
                filename_grid = "YB_NBE_grid_df.csv"
                filename_png = "YB_NBE.png"

            # Save the grid to a CSV file
            grid_csv_path = os.path.join(subdir, filename_grid)
            grid_df.to_csv(grid_csv_path)

            # Define x and y limits
            x_min, x_max = grid_df.columns[0], grid_df.columns[-1]
            y_min, y_max = grid_df.index[0], grid_df.index[-1]

            for i, x in enumerate(grid_df.columns):
                for j, y in enumerate(grid_df.index):
                    x = float(x)
                    y = float(y)
                    if x ** 2 + y ** 2 > (radius - edge_exclusion) ** 2:
                        grid_df.iloc[j, i] = np.nan

            if identical and column in ["Intensity (cps)", "Energy (eV)",
                                        "YB/NBE area ratio", "Thickness"]:

                vmin, vmax = get_vmin_vmax(data_frame, column, threshold,
                                           identical, subdir)
                filename_png = f"{column.split()[0]}_ID_scale.png"
                if column == "YB/NBE area ratio":
                    filename_png = "YB_NBE_ID_scale.png"

            # Generate the heatmap
            fig, ax = plt.subplots()

            if identical and column in ["Intensity (cps)", "Energy (eV)",
                                        "YB/NBE area ratio", "Thickness"]:

                img = ax.imshow(grid_df,
                                extent=(x_min, x_max, y_min, y_max),
                                origin='lower', cmap='Spectral_r',
                                vmin=vmin, vmax=vmax)
            else:
                img = ax.imshow(grid_df,
                                extent=(x_min, x_max, y_min, y_max),
                                origin='lower', cmap='Spectral_r')

            # Add colorbar and labels
            cbar = plt.colorbar(img, ax=ax, shrink=1)
            cbar.ax.tick_params(labelsize=18)
            plt.xlabel('X (cm)', fontsize=20)
            plt.ylabel('Y (cm)', fontsize=20)
            title = f"S{os.path.basename(subdir)} - {column}" if slot_number \
                else column
            plt.title(title, fontsize=20)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

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
    """Determine vmin and vmax values for a given column."""

    vmin, vmax = None, None

    boxplot_files = {
        'Intensity (cps)': 'Boxplot_intensity.csv',
        'Energy (eV)': 'Boxplot_energy.csv',
        'YB area (cps)': 'Boxplot_YB_area.csv',
        'NBE area (cps)': 'Boxplot_NBE_area.csv',
        'YB/NBE area ratio': 'Boxplot_YB_NBE_ratio.csv',
        'Thickness': 'Thickness.csv'
    }

    filename = boxplot_files.get(column)

    if identical == 'Manual' and column in ["Intensity (cps)", "Energy (eV)",
                                            "YB/NBE area ratio", "Thickness"]:
        if column == 'Intensity (cps)':
            vmin = float(threshold[0])
            vmax = float(threshold[1])
        if column == 'Energy (eV):':
            vmin = float(threshold[2])
            vmax = float(threshold[3])
        if column == "YB/NBE area ratio" or column == "Thickness":
            vmin = float(threshold[0])
            vmax = float(threshold[1])
        return vmin, vmax

    elif identical == 'Auto' and filename:
        parent_dir = os.path.dirname(subdir)
        boxplot_file = os.path.join(parent_dir, "Liste_data", filename)

        print(boxplot_file)
        if os.path.exists(boxplot_file):
            boxplot_data = pd.read_csv(boxplot_file, header=0, index_col=0)
            print(boxplot_data)
            vmin = boxplot_data.min(numeric_only=True).min()
            vmax = boxplot_data.max(numeric_only=True).max()
            print(f"Auto scaling for {column}: vmin={vmin}, vmax={vmax}")
            return vmin, vmax
        else:
            print(
                f"Boxplot file not found: {boxplot_file}. Using default "
                f"scaling.")

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
            sigma_val = filtered_data['3sigma'].values[0]
            uniformity = (
                                     1 - sigma_val / mean_val) * 100 if \
                mean_val != 0 else 0

            uniformity = max(uniformity, 0)

            # Add text for mean, 3sigma, and uniformity
            ax.text(0.01, 0.01, f'Mean: {mean_val:.3f}',
                    transform=ax.transAxes, fontsize=14, ha='left', va='bottom',
                    color='black')
            ax.text(0.96, 0.01, f'3$\sigma$: {sigma_val:.3f}',
                    transform=ax.transAxes, fontsize=14, ha='right',
                    va='bottom', color='black')
            ax.text(0.96, 0.96, f'U: {uniformity:.1f}%', transform=ax.transAxes,
                    fontsize=14, ha='right', va='top', color='black')


def plot_wdf_thickness(filepath, radius, edge_exclusion, slot_number):
    """
    Generates intensity and energy mappings from the given data file.
    """
    try:
        # Load data from the file
        grid_df = pd.read_csv(filepath, index_col=0, header=0)
        print(grid_df)
        subdir = os.path.dirname(filepath)
        # Create a grid for the column

        # Define x and y limits
        # Define the x and y limits for the plot
        x_min = float(grid_df.columns[0])
        x_max = float(grid_df.columns[-1])
        y_min = float(grid_df.index[0])
        y_max = float(grid_df.index[-1])

        for i, x in enumerate(grid_df.columns):
            for j, y in enumerate(grid_df.index):
                x = float(x)
                y = float(y)
                if x ** 2 + y ** 2 > (radius - edge_exclusion) ** 2:
                    grid_df.iloc[j, i] = np.nan

        # Generate heatmap
        plt.figure(figsize=(10, 6))
        plt.imshow(grid_df, cmap='Spectral_r',
                   extent=[x_min, x_max, y_max, y_min])
        plt.colorbar().ax.tick_params(labelsize=18)
        plt.xlabel('X (cm)', fontsize=20)
        plt.ylabel('Y (cm)', fontsize=20)
        title = 'Thickness (um)'
        if slot_number:
            title = "S" + str(os.path.basename(subdir)) + " - " + title
        plt.title(title, fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Add a black circle indicating the valid region
        circle = plt.Circle((0, 0), radius - edge_exclusion,
                            color='black', fill=False, linewidth=0.5)
        plt.gca().add_patch(circle)
        plt.xlim(-radius, radius)
        plt.ylim(-radius, radius)

        # Save the heatmap
        heatmap_path = os.path.join(subdir, "Thickness.png")
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()

    except Exception as error:
        print(f"Error processing file {filepath}: {error}")


def _find_max_spectrum(x_value, y_value):
    """Helper function to find the maximum
    intensity and its corresponding energy."""
    max_y_local = np.max(y_value)
    idx_max_y_local = np.argmax(y_value)
    x_associe_local = x_value[idx_max_y_local]
    return max_y_local, x_associe_local


class PhotoluminescenceGan:
    """
    Class for photoluminescence analysis.
    Contains methods for performing PL spectra fitting and plotting.
    """

    def __init__(self, dirname, wafer_size, edge_exclusion, step,
                 int_min=None, int_max=None,
                 ev_min=None, ev_max=None,
                 area_min=None, area_max=None,
                 thickness_min=None, thickness_max=None, subdir=None):
        self.dirname = dirname
        self.wafer_size = wafer_size
        self.edge_exclusion = edge_exclusion
        self.radius = self.wafer_size / 2
        self.step = step
        self.int_min = int_min
        self.int_max = int_max
        self.ev_min = ev_min
        self.ev_max = ev_max
        self.min_area = area_min
        self.max_area = area_max
        self.min_thickness = thickness_min
        self.max_thickness = thickness_max
        self.subdir = [str(x).strip().lower() for x in subdir]

    def find_xy_gan(self):
        """
        Find the maximum of Y and associate it
        with its X value for each spectrum.
        """

        for subdir, _, files in os.walk(self.dirname):
            if os.path.basename(subdir) in self.subdir:
                for file in files:
                    filepath = os.path.join(subdir, file)
                    if filepath.endswith("data.csv"):
                        data = []
                        spectra = pd.read_csv(filepath, sep='\t', header=None)
                        columns = spectra.shape[1]
                        if columns < 100:
                            return True

                        # Extract relevant spectra and preprocess them
                        spectrum_x_gan = spectra.iloc[0, 600:900].to_numpy()
                        spectrum_x_yb = spectra.iloc[0, 2:30].to_numpy()
                        raw_data_values = spectra.iloc[1:, 600:900].to_numpy()
                        # Laser data (Y)
                        raw_data_laser = spectra.iloc[1:, 2:30].to_numpy()
                        raw_data_positions = spectra.iloc[1:, [0, 1]] \
                                                 .to_numpy() * self.step

                        # Calculate max intensity and energy for each spectrum
                        for i in range(len(raw_data_values)):
                            xpos, ypos = raw_data_positions[i]

                            max_y_gan, x_associe_gan = _find_max_spectrum(
                                spectrum_x_gan, raw_data_values[i])
                            max_y_laser, x_associe_laser = \
                                _find_max_spectrum(
                                spectrum_x_yb, raw_data_laser[i])

                            # Adjust energy for GaN spectrum based on laser
                            # offset
                            offset = 4.66 - x_associe_laser
                            x_associe_gan += offset

                            # Verify is laser component is there
                            if all(value < 4.0 for value in spectrum_x_yb):
                                print("No laser component")
                                return True

                            data.append(
                                [xpos, ypos, max_y_gan, x_associe_gan,
                                 max_y_laser,
                                 x_associe_laser])

                        # Save the data to CSV
                        columns = ["X", "Y", "Intensity (cps)", "Energy (eV)",
                                   "Laser intensity (cps)", "Laser energy (eV)"]
                        output_df = pd.DataFrame(data, columns=columns)
                        output_df.to_csv(os.path.join(subdir, "data_e_i.csv"),
                                         index=False)

                        # Define radius excluding edge region
                        r_squared = (self.radius - self.edge_exclusion) ** 2

                        # Create a mask for filtering out data based on
                        # intensity,
                        # energy, and position
                        mask = (
                                (output_df['Intensity (cps)'] <= self.int_min) |
                                (output_df['Intensity (cps)'] >= self.int_max) |
                                (output_df['Energy (eV)'] <= self.ev_min) |
                                (output_df['Energy (eV)'] >= self.ev_max) |
                                ((output_df['X'] ** 2 + output_df[
                                    'Y'] ** 2) >= r_squared)
                        )

                        # Apply the mask and set filtered values to NaN
                        output_df.loc[mask, ['Intensity (cps)',
                                             'Energy (eV)']] = np.nan

                        # Save the filtered data to 'data_filtered.csv'
                        output_df.to_csv(
                            os.path.join(subdir, "data_filtered.csv"),
                            index=False)

    def find_area(self):
        """
        Find the maximum of Y and associate it with its X value.
        Calculate the area under GaN and YB spectra and visualize them.
        """

        for subdir, _, files in os.walk(self.dirname):
            if os.path.basename(subdir) in self.subdir:
                for file in files:
                    filepath = os.path.join(subdir, file)
                    if filepath.endswith("data.csv"):
                        data = []
                        raw_data = pd.read_csv(filepath, sep='\t', header=None)
                        columns = raw_data.shape[1]
                        if columns < 100:
                            return True

                        # Extract necessary spectrum ranges (reversed)
                        spectrum_nbe_x = raw_data.iloc[0, 45:75].to_numpy()[
                                         ::-1]
                        spectrum_yb_x = raw_data.iloc[0, 200:600].to_numpy()[
                                        ::-1]

                        # Loop over each spectrum to calculate areas
                        raw_data_length = len(raw_data) - 1
                        for i in range(raw_data_length):
                            xpos, ypos = raw_data.iloc[i + 1, 0] * self.step, \
                                         raw_data.iloc[
                                             i + 1, 1] * self.step

                            # GaN NBE spectrum and area
                            spectrum_y_nbe = raw_data.iloc[i + 1, 45:75].to_numpy()[::-1]
                            area_gan = np.trapezoid(spectrum_y_nbe,
                                                    spectrum_nbe_x)

                            # YB spectrum and area
                            spectrum_y_yb = raw_data.iloc[i + 1,
                                            200:600].to_numpy()[::-1]
                            area_yb = np.trapezoid(spectrum_y_yb, spectrum_yb_x)

                            # Calculate the area ratio
                            area_ratio = area_yb / area_gan if area_gan != 0 \
                                else \
                                np.nan

                            # Verify if YB and NBE are there

                            if spectrum_nbe_x[1] < 3 or 3 < spectrum_yb_x[1]:
                                print("No GaN spectrum")
                                return True

                            data.append(
                                [xpos, ypos, area_gan, area_yb, area_ratio])

                        # Save the results to CSV
                        columns = ["X", "Y", "NBE area (cps)", "YB area (cps)",
                                   "YB/NBE area ratio"]
                        output_df = pd.DataFrame(data, columns=columns)
                        output_df.to_csv(os.path.join(subdir, "data_e_i.csv"),
                                         index=False)
                        print(
                            f"Results saved to "
                            f"{os.path.join(subdir, 'data_e_i.csv')}")

                        # Define radius excluding edge region
                        r_squared = (self.radius - self.edge_exclusion) ** 2

                        # Create a mask for filtering out data based on
                        # intensity,
                        # energy, and position
                        mask = (
                                (output_df[
                                     'YB/NBE area ratio'] <= self.min_area) |
                                (output_df[
                                     'YB/NBE area ratio'] >= self.max_area) |
                                ((output_df['X'] ** 2 + output_df[
                                    'Y'] ** 2) >= r_squared)
                        )

                        # Apply the mask and set filtered values to NaN
                        output_df.loc[mask, ["NBE area (cps)", "YB area (cps)",
                                             "YB/NBE area ratio"]] = np.nan

                        # Save the filtered data to 'data_filtered.csv'
                        output_df.to_csv(
                            os.path.join(subdir, "data_filtered.csv"),
                            index=False)

    def plot(self, mat=None, slot_number=None, identical=None, stats=None):
        """
        Search for filtered data files and process them using multiprocessing.
        """
        filepaths = []
        threshold = None

        if mat == "GaN":
            threshold = [self.int_min, self.ev_min, self.int_max, self.ev_max]
        elif mat == "GaN_area":
            threshold = [self.min_area, self.max_area]
        elif mat == "Thickness":
            threshold = [self.min_thickness, self.max_thickness]

        for subdir, _, files in os.walk(self.dirname):
            if os.path.basename(subdir) in self.subdir:
                for file in files:
                    if file.endswith("data_filtered.csv"):
                        filepaths.append(os.path.join(subdir, file))

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

    def plot_thickness(self, slot_number=None):
        """
        Processes filtered data files and visualizes thickness data using
        multiprocessing.
        """
        filepath = os.path.join(self.dirname, "data.csv")
        if not os.path.exists(filepath):
            return True

        data_frame = pd.read_csv(filepath)

        # Identify columns and group them in sets of 3 (X, Y, Thickness)
        columns = data_frame.columns
        group_size = 3
        for i in range(0, len(columns), group_size):
            group_columns = columns[i:i + group_size]

            # Skip incomplete groups
            if len(group_columns) != group_size:
                print(
                    f"Skipping incomplete group at index {i}: {group_columns}")
                continue

            # Extract identifier based on the number of underscores
            column_name = group_columns[0]
            parts = column_name.split("_")
            print(parts)
            if len(parts) == 4:
                identifier = parts[1][1:].lstrip("0")
            elif len(parts) == 3:
                identifier = parts[0][1:].lstrip("0")
            else:
                print(
                    f"Unable to determine identifier for column: {column_name}")
                continue

            # Create a subdirectory for each identifier
            folder_path = os.path.join(self.dirname, identifier)
            os.makedirs(folder_path, exist_ok=True)

            # Create a new DataFrame with X, Y, and Thickness columns
            new_df = data_frame[group_columns]
            new_df.columns = ["X", "Y", "Thickness"]

            # Remove the first row and reset the index
            new_df = new_df.iloc[1:].reset_index(drop=True)

            # Convert X and Y to integers and scale by step, Thickness to float
            new_df["X"] = new_df["X"].astype(int) * self.step
            new_df["Y"] = new_df["Y"].astype(int) * self.step
            new_df["Thickness"] = new_df["Thickness"].astype(float)

            # Save the cleaned data to CSV
            output_path = os.path.join(folder_path, "data_e_i.csv")
            new_df.to_csv(output_path, index=False)

            # Define radius for excluding edge regions
            r_squared = (self.radius - self.edge_exclusion) ** 2

            # Mask to filter invalid thickness and edge regions
            mask = (
                    (new_df["Thickness"] <= self.min_thickness) |
                    (new_df["Thickness"] >= self.max_thickness) |
                    ((new_df["X"] ** 2 + new_df["Y"] ** 2) >= r_squared)
            )
            new_df.loc[mask, "Thickness"] = np.nan

            # Save the filtered data to CSV
            filtered_output_path = os.path.join(folder_path,
                                                "data_filtered.csv")
            new_df.to_csv(filtered_output_path, index=False)

            # Pivot data to create a grid format
            pivoted_df = new_df.pivot(index="Y", columns="X",
                                      values="Thickness")

            # Save the pivoted grid data to CSV
            grid_output_path = os.path.join(folder_path,
                                            "Thickness_grid_df.csv")
            pivoted_df.to_csv(grid_output_path)
            print(f"File saved at: {grid_output_path}")

        filepaths = []
        for subdir, _, files in os.walk(self.dirname):
            if os.path.basename(subdir) in self.subdir:
                for file in files:
                    if file.endswith("Thickness_grid_df.csv"):
                        filepaths.append(os.path.join(subdir, file))

        # Prepare multiprocessing
        process_partial = partial(
            plot_wdf_thickness,
            radius=self.radius,
            edge_exclusion=self.edge_exclusion,
            slot_number=slot_number,
        )
        max_workers = max(1, os.cpu_count() // 2)

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process_partial, filepaths)


if __name__ == "__main__":
    DIRNAME = r'C:\Users\TM273821\Desktop\PL\PL_GaN'
    SUBDIR = [3, 5]
    Photoluminescence = PhotoluminescenceGan(DIRNAME, 20, 0, 0.3,
                                             int_min=0, int_max=21000,
                                             ev_min=3.412, ev_max=3.425,
                                             area_min=0.001, area_max=0.007,
                                             thickness_min=2.1,
                                             thickness_max=4, subdir=SUBDIR)
    start_time = time.time()
    # Photoluminescence.plot_thickness(slot_number=True)
    # Photoluminescence.plot_no_filter()
    Photoluminescence.plot(mat="GaN_area", slot_number=True, identical='Manual',
                           stats=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time : {elapsed_time:.2f} s")
