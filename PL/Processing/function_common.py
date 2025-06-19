"""
Function_common Module

This module contains functions and classes which are common to all
characterizations.
"""
import re
import os
import math
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from PIL import Image
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numpy as np

rcParams.update({'figure.autolayout': True})


def convert(filepath, ev_conversion=None):
    """
    Process a single file: read, convert wavelengths (nm) to energy (eV).
    """
    # Read the file into a DataFrame



    file = pd.read_csv(filepath, header=None)


    # Check if the file is in the expected format
    if file.iloc[0, 0] == 'Index X':

        # Round the data values to one decimal place
        file.iloc[1:, :] = file.iloc[1:, :].astype(float).round(1)

        # Replace the first two elements with NaN
        file.iloc[0, 0] = file.iloc[0, 1] = math.nan
        # Convert the first row (wavelengths in nm) to energy in eV
        if ev_conversion :
            file.iloc[0, :] = 1240 / file.iloc[0, :].astype(float)

        # Save the modified DataFrame back to the file in tab-delimited format
        file.to_csv(filepath , sep='\t', index=False, header=False,
                    mode='w+')


class Common:
    """
    Common Class
    This class contains all functions.
    """

    def __init__(self, dirname, subdir):
        self.dirname = dirname
        self.subdir = [str(x).strip().lower() for x in subdir]

    def create_image_grid(self, zscale=None, mat=None):
        """
            Plots an image grid based on selected zscale and material type.
        """

        # Define image names based on zscale and mat parameters
        image_options = {
            ('Auto', 'Thickness'): ['Thickness.png'],
            ('Identical', 'Thickness'): ['Thickness_ID_scale.png'],
            ('Identical', 'GaN'): ['Intensity_ID_scale.png',
                                   'Energy_ID_scale.png'],
            ('Identical', '2D'): ['Intensity_ID_scale.png',
                                  'Energy_ID_scale.png'],
            ('Auto', '2D'): ['Intensity.png', 'Energy.png'],
            ('Auto', 'GaN'): ['Intensity.png', 'Energy.png'],
            ('Auto', 'GaN_area'): ['YB_NBE.png'],
            ('Identical', 'GaN_area'): ['YB_NBE_ID_scale.png'],
            ('Auto', 'Bonding'): ['GS_transfert.png', 'TS_transfert.png'],
            ('test', 'Bonding'): ['Area_ID_scale.png'],
            ('Identical', 'SiC'): ['A_ampli_ID_scale.png',
                                  'A_fwhm_ID_scale.png', "A_x0_ID_scale.png","B_ampli_ID_scale.png"]
        }
        image_names = image_options.get((zscale, mat), [])

        # Get sorted subfolders
        def sort_key(subfolder_name):
            parts = subfolder_name.split("\\")
            for part in parts:
                if part.isdigit():
                    return int(part)
            return float('inf')

        subfolders = sorted(
            [f.path for f in os.scandir(self.dirname) if f.is_dir()],
            key=sort_key)

        # Calculate grid dimensions
        num_subfolders = len(subfolders)
        rows = -(-num_subfolders // 5)  # Ceiling division

        # Load images from subfolders
        images_list = []
        for image_name in image_names:
            sub_images = [Image.open(os.path.join(subfolder, image_name))
                          for subfolder in subfolders if
                          os.path.exists(os.path.join(subfolder, image_name))]
            images_list.append(sub_images)

        # Determine grid dimensions based on image sizes
        image_spacing = 50
        grid_width, grid_height = 0, 0
        for sub_images in images_list:
            for image in sub_images:
                grid_width = max(grid_width, 5 * image.width)
                grid_height = max(grid_height, rows * image.height)

        # Create blank grid images and paste images into the grid
        grid_images = [Image.new('RGB', (grid_width + 250, grid_height + 200),
                                 (255, 255, 255))
                       for _ in image_names]

        for idx, sub_images in enumerate(images_list):
            for i, image in enumerate(sub_images):
                x, y = (i % 5) * (image.width + image_spacing), (i // 5) * (
                        image.height + image_spacing)
                grid_images[idx].paste(image, (x, y))

        # Save merged grid images
        save_path = os.path.join(self.dirname, "Graphe", "Mapping")
        os.makedirs(save_path, exist_ok=True)
        for idx, grid_image in enumerate(grid_images):
            grid_image.save(os.path.join(save_path, f"All_{image_names[idx]}"))

    def reboot(self, carac='None'):
        """
        Delete unnecessary files.
        """

        excel_file_extensions = (
            ".xlsx", ".xls", ".csv")

        for root, dirs, files in os.walk(self.dirname):
            if os.path.basename(root) in self.subdir:
                for folder in ["Graphe", "Mapping", "Spectra", "raw_data",
                               "Liste_data"]:
                    path = os.path.join(self.dirname, folder)
                    path_2 = os.path.join(root, folder)

                    if os.path.exists(path):
                        shutil.rmtree(path)
                        print(f"Delete: {path}")

                    if os.path.exists(path_2):
                        shutil.rmtree(path_2)
                        print(f"Delete: {path_2}")

        filenames_to_remove = {

            "photoluminescence": ["data_e_i", "data_filtered.csv", ".png",
                                  "mapping.csv", "grid_df", "Parameters.csv",
                                  "fitted", "_stats", "rate.csv", ".npy"],
            "thickness": ["data_thickness", "data_filtered.csv", ".png",
                          "mapping.csv", "grid_df", "Parameters.csv",
                          "fitted", "_stats", "rate", ".npy"], }

        if carac in filenames_to_remove:
            for subdir, dirs, files in os.walk(self.dirname):
                if os.path.basename(subdir) in self.subdir:
                    for file in files:
                        filepath = os.path.join(subdir, file)
                        print(filepath)
                        if any(name in filepath for name in
                               filenames_to_remove[carac]):
                            os.remove(filepath)

        # Delete low-size excel files (<10ko)
        if carac == "photoluminescence":
            for subdir, dirs, files in os.walk(self.dirname):
                if os.path.basename(subdir) in self.subdir:
                    for file in files:
                        filepath = os.path.join(subdir, file)

                        if filepath.endswith(excel_file_extensions) and not filepath.endswith("rate.csv"):
                            file_size_kb = os.path.getsize(filepath) / 1024
                            if file_size_kb < 10:
                                os.remove(filepath)

    def stats(self):
        """
            stats function
            Create Parameters files. Calculate the mean value for each
            .
        """

        path_liste = os.path.join(self.dirname, 'Liste_data')
        if not os.path.exists(path_liste):
            os.makedirs(path_liste)

        filename = "data_e_i.csv"
        filename_parameters = 'Parameters.csv'
        for subdir, _, files in os.walk(self.dirname):
            for file in files:
                filepat = subdir + os.sep
                filepath = subdir + os.sep + file
                if filepath.endswith(filename):
                    os.chdir(filepat)
                    data_frame = pd.read_csv(filepath)
                    stat = data_frame.describe()
                    mod_dataframe = stat.drop(
                        ['count', '25%', '50%', '75%'])
                    mod_dataframe.iloc[1, :] = mod_dataframe.iloc[1, :]
                    mod_dataframe = mod_dataframe.rename(
                        index={'std': 'sigma'})
                    mod_dataframe = mod_dataframe.transpose()

                    mod_dataframe.to_csv(filename_parameters)
                    mod_dataframe = mod_dataframe.drop(
                        ['X', 'Y'])

                    slot_number = \
                        os.path.split(os.path.dirname(filepath))[
                            -1]
                    mod_dataframe['Slot'] = slot_number
                    mod_dataframe.to_csv(filename_parameters)

        parameters_dataframe = pd.DataFrame(
            columns=['Unnamed: 0', 'mean', 'sigma', 'min', 'max'])
        for subdir, dirs, files in os.walk(self.dirname):
            for file in files:
                filepat = subdir + os.sep
                filepath = subdir + os.sep + file
                if filepath.endswith(filename_parameters):
                    os.chdir(filepat)
                    data_frame = pd.read_csv(filepath)
                    parameters_dataframe = pd.concat(
                        [parameters_dataframe, data_frame])

        # Reorder columns to have 'Slot' as the first column
        cols = ['Slot'] + [col for col in
                           parameters_dataframe.columns if
                           col != 'Slot']
        parameters_dataframe = parameters_dataframe[cols]

        parameters_dataframe = parameters_dataframe.rename(
            columns={'Unnamed: 0': 'Parameters'})
        parameters_dataframe.to_csv(
            self.dirname + os.sep + "Liste_data" + os.sep + 'Stats.csv',
            index=False)

    def plot_boxplot(self, mat=None):

        """
                plot_boxplot function
                Create boxplot based on parameters.
                """
        path_liste = self.dirname + '\\Liste_data'
        if not os.path.exists(path_liste):
            os.makedirs(path_liste)
        figure_height = 12
        figure_width = 8
        path2 = self.dirname + os.sep + "Graphe" + os.sep + "Boxplot"
        if not os.path.exists(path2):
            os.makedirs(path2)

        taille_df = []
        ylabel = []
        filename = []

        if mat == "Thickness":
            filename_png = ["thickness"]
            ylabel = ["Thickness (um)"]
        if mat == "GaN":
            filename_png = ["intensity", "energy", "Laser_int", "Laser_eV"]
            ylabel = ["NBE intensity (cps)", "NBE energy (eV)",
                      "Laser intensity (cps)", "Laser energy (eV)"]
        if mat == "GaN_area":
            filename_png = ["NBE_area", "YB_area", "YB_NBE_ratio"]
            ylabel = ["NBE area (cps)", "YB area (cps)",
                      "YB/NBE area ratio"]

        for subdir, _, files in os.walk(self.dirname):
            for file in files:
                filepath = subdir + os.sep + file
                filename = "data_e_i.csv"
                if filepath.endswith(filename):
                    data_frame = pd.read_csv(filepath)

                    taille_df = data_frame.shape

        column_number = taille_df[1] - 2
        for j in range(column_number):
            col_data = {}
            for subdir, _, files in os.walk(self.dirname):
                for file in files:
                    filepath = subdir + os.sep + file
                    filename = "data_filtered.csv"
                    if filepath.endswith(filename):
                        data_frame = pd.read_csv(filepath)
                        if not data_frame.empty:
                            nom_colonne = os.path.basename(subdir)
                            col_data[nom_colonne] = data_frame.iloc[:, j + 2]

            # Create a df
            df_merged = pd.DataFrame(col_data)
            df_merged = df_merged[~np.isnan(df_merged)]

            # Reset index of df
            df_merged.reset_index(drop=True, inplace=True)

            # Index => Int (for sort)

            df_merged.columns = df_merged.columns.astype(int)

            # Sort the index
            df_merged = df_merged.sort_index(axis=1)
            print(df_merged)

            data_for_boxplot = [df_merged[col].dropna().values for col in
                                df_merged.columns]
            print(data_for_boxplot)

            df_merged = pd.DataFrame({
                col: pd.Series(data) for col, data in
                zip(df_merged.columns, data_for_boxplot)
            })


            # Save the df
            nouveau_fichier = "Boxplot_" + filename_png[j] + ".csv"

            df_merged.to_csv(
                self.dirname + os.sep + "Liste_data" + os.sep + nouveau_fichier)
            fig, ax = plt.subplots(figsize=(figure_height, figure_width))
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.boxplot([df_merged[col].dropna() for col in df_merged.columns], showfliers=False)
            ax.set_xticklabels(df_merged.columns)
            ax.set_xlabel('Wafer', fontsize=26)
            ax.set_ylabel(ylabel[j], fontsize=26)
            plt.savefig(
                self.dirname + os.sep + "Graphe" + os.sep + "Boxplot" +
                os.sep + filename_png[j] + ".png", bbox_inches='tight')
            plt.close()

    def organize_files_by_numbers(self):
        """
        Scans files in a directory, creates
        subdirectories for each detected number,
        and moves the corresponding files into the appropriate subdirectories.
        """
        if not os.path.exists(self.dirname):
            print("The specified folder does not exist.")
            return

        # Regular expression to detect numbers in file names
        number_pattern = re.compile(r'\d+')

        for file_name in os.listdir(self.dirname):
            file_path = os.path.join(self.dirname, file_name)

            # Skip directories, only process files
            if not os.path.isfile(file_path):
                continue

            # Search for numbers in the file name
            match = number_pattern.search(file_name)
            if match and file_path.endswith(".emd"):
                # Get the first number found
                number = match.group()
                number = number.lstrip("0")

                # Create a subdirectory corresponding to the number
                subfolder_path = os.path.join(self.dirname, number)

                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)

        print("Organization completed.")

    def nm_to_ev(self, ev_conversion=False):
        """
        Convert files from nm to eV using parallel processing.
        """
        filepaths = []
        for subdir, _, files in os.walk(self.dirname):
            if os.path.basename(subdir) in self.subdir:
                for file in files:
                    filepath = os.path.join(subdir, file)
                    if filepath.endswith("data.csv"):
                        filepaths.append(filepath)

        process_partial = partial(convert, ev_conversion=ev_conversion)

        num_cpu_cores = os.cpu_count()
        max_workers = max(1, num_cpu_cores // 2)

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process_partial, filepaths)


if __name__ == "__main__":
    DIRNAME = r"C:\Users\TM273821\Desktop\PL\250428-D24S2619 - 2D"
    SETTINGS = r'C:\Users\TM273821\Desktop\Model\Setting_PL_nodecomp.json'
    DIRNAME_MODEL = r'C:\Users\TM273821\Desktop\Model\PL-ALD-MoSx_c.json'
    SUBDIR = [1,20,23,24]
    Common = Common(DIRNAME, SUBDIR)
    # Common.reboot('Photoluminescence')
    # Common.nm_to_ev(ev_conversion=False)
    # Common.stats()
    # Common.create_image_grid(zscale="Auto", mat='Bonding')
    Common.create_image_grid(zscale="Identical", mat='2D')
    # Common.organize_files_by_numbers()
    # Common.create_image_grid_scale()
