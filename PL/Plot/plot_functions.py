import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget
import matplotlib.colors as mcolors


class PlotFunctions(QWidget):
    """Class for handling plot functionalities."""

    def __init__(self, master, button_frame):
        super().__init__()
        self.master = master
        self.data_frame = None
        self.button_frame = button_frame
        self.step = None
        self.wafer_size = None
        self.edge_exclusion = None

    def plot_wdxrf(self, dirname, ax, numbers_str, parameters):
        """Add WDXRF mapping to the canvas."""
        global filename
        min_value = None
        max_value = None
        subdir = os.path.join(dirname, f"{numbers_str}")

        # Select the correct file based on the parameter
        file_mapping = {
            "Energy (eV)": "Energy_grid_df.csv",
            "Intensity (cps)": "Intensity_grid_df.csv",
            "YB/NBE area ratio": "YB_NBE_grid_df.csv",
            "GaN thickness": "Thickness_grid_df.csv",
            "Base - Top": ["Base_grid_df.csv", "Top_grid_df.csv"],
        }
        filename = file_mapping.get(parameters, None)

        if not filename:
            print(f"Invalid parameter: {parameters}")
            return

        # Si le paramètre est "Base - Top", vérifier l'existence des fichiers dans le dossier
        if parameters == "Base - Top":
            base_file = os.path.join(subdir, "Base_grid_df.csv")
            top_file = os.path.join(subdir, "Top_grid_df.csv")

            # Vérifier si les fichiers existent
            if os.path.exists(base_file):
                filepath = base_file
            elif os.path.exists(top_file):
                filepath = top_file
            else:
                print("Neither 'Base_grid_df.csv' nor 'Top_grid_df.csv' exist.")
                return
        else :
            filepath = os.path.join(subdir, filename)


        values = self.button_frame.get_values()
        scale_values = self.button_frame.get_scale_values()

        self.wafer_size = values.get('Wafer size (cm):', None)
        self.edge_exclusion = values.get('Edge Exclusion (cm):', None)
        self.step = values.get('Step (cm):', None)
        radius = self.wafer_size / 2
        # Extract scale type
        scale_checkbox = scale_values.get('Scale Type', None)

        # Determine min and max based on scale type and parameters
        if scale_checkbox == 'Identical scale' or scale_checkbox == 'Autoscale':
            if parameters == "Energy (eV)":
                max_value = values.get('Max energy (eV):', 0)
                min_value = values.get('Min energy (eV):', 0)
                print(f"eV min: {min_value}, eV max: {max_value}")

            elif parameters == "Intensity (cps)":
                max_value = values.get('Max intensity:', 0)
                min_value = values.get('Min intensity:', 0)
                print(f"Int min: {min_value}, Int max: {max_value}")

            elif parameters == "YB/NBE area ratio":
                max_value = values.get('Max area ratio:', 0)
                min_value = values.get('Min area ratio:', 0)
                print(f"Min: {min_value}, Max: {max_value}")

            elif parameters == "GaN thickness":
                max_value = values.get('Max thickness (um):', 0)
                min_value = values.get('Min thickness (um):', 0)
                print(f"Min: {min_value}, Max: {max_value}")

        elif scale_checkbox == 'Identical scale auto':
            if parameters == "Energy (eV)":
                data_frame = pd.read_csv(
                    os.path.join(dirname, 'Liste_data', "Boxplot_energy.csv"))
                max_value = data_frame.iloc[:, 1:].max().max()
                min_value = data_frame.iloc[:, 1:].min().min()
                print(f"eV min: {min_value}, eV max: {max_value}")

            elif parameters == "Intensity (cps)":
                data_frame = pd.read_csv(os.path.join(dirname, 'Liste_data',
                                                      "Boxplot_intensity.csv"))
                max_value = data_frame.iloc[:, 1:].max().max()
                min_value = data_frame.iloc[:, 1:].min().min()
                print(f"Int min: {min_value}, Int max: {max_value}")

            elif parameters == "YB/NBE area ratio":
                data_frame = pd.read_csv(os.path.join(dirname, 'Liste_data',
                                                      "Boxplot_YB_NBE_ratio.csv"))
                max_value = data_frame.iloc[:, 1:].max().max()
                min_value = data_frame.iloc[:, 1:].min().min()
                print(f"Int min: {min_value}, Int max: {max_value}")

        if parameters == "Base - Top":
            min_value = 0
            max_value = 1

        # Plot the WDXRF mapping
        if os.path.exists(filepath):
            data_frame = pd.read_csv(filepath, index_col=0, header=0)
            print(len(data_frame))
            if len(data_frame) < 2:
                ax.text(0, 0, "No data available :(", fontsize=10, color='red',
                        ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.5))
                ax.set_xlabel('X (cm)', fontsize=20)
                ax.set_ylabel('Y (cm)', fontsize=20)
                ax.tick_params(labelsize=14)
                ax.set_xlim(-radius, radius)
                ax.set_ylim(-radius, radius)
            else:
                x_min = float(data_frame.columns[0])
                x_max = float(data_frame.columns[-1])
                y_min = float(data_frame.index[0])
                y_max = float(data_frame.index[-1])

                x_coords = data_frame.columns.astype(float)
                y_coords = data_frame.index.astype(float)
                X, Y = np.meshgrid(x_coords, y_coords)

                # Mask data outside the wafer boundary
                condition = X ** 2 + Y ** 2 >= (radius - self.edge_exclusion) ** 2
                data_frame = data_frame.mask(condition)

                # Plot data
                plt.figure(figsize=(10, 6))
                cmap='Spectral_r'
                if parameters == "Base - Top" and filepath.endswith("Base_grid_df.csv"):
                    cmap = mcolors.ListedColormap(['red', 'blue'])
                if parameters == "Base - Top" and filepath.endswith("Top_grid_df.csv") :
                    cmap = mcolors.ListedColormap(['blue','green'])


                plot = ax.imshow(data_frame, extent=[x_min, x_max, y_max, y_min],
                           cmap=cmap,
                          vmin=min_value if scale_checkbox in ["Identical scale",
                                                               "Identical scale auto"] else None, vmax=max_value if scale_checkbox in [
                                  "Identical scale",
                                  "Identical scale auto"] else None
                                )

                print(parameters)
                if parameters != "Base - Top":
                    cbar = plt.colorbar(plot, ax=ax, shrink=0.6)
                    cbar.ax.tick_params(labelsize=16)

                ax.set_xlabel('X (cm)', fontsize=20)
                ax.set_ylabel('Y (cm)', fontsize=20)
                ax.tick_params(labelsize=14)

                # Add wafer boundary
                circle = plt.Circle((0, 0), radius - self.edge_exclusion,
                                    color='black', fill=False, linewidth=0.5)
                ax.add_patch(circle)
                ax.set_xlim(-radius, radius)
                ax.set_ylim(-radius, radius)
                ax.set_aspect('equal')

                list_data_dir = os.path.join(dirname, "Liste_data")
                rates_file = os.path.join(list_data_dir, "rates.csv")

                if os.path.exists(rates_file):
                    rates_data = pd.read_csv(rates_file)

                    print(rates_data)
                    filtered_data = rates_data[
                        (rates_data['Slot'] == float(
                            os.path.basename(subdir)))]

                    print(filtered_data)

                    if not filtered_data.empty:
                        # Extract the mean and 3sigma values
                        rate_value = filtered_data['Cov. rate'].values[0]

                        print(rate_value)

                        # Add text for mean and 3sigma
                        ax.text(0.01, 0.01, f'Cov. : {rate_value:.1f} %',
                                transform=ax.transAxes, fontsize=14,
                                ha='left', va='bottom', color='black')
        else:
            ax.text(0, 0, "No data available :(", fontsize=10, color='red',
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.5))
            ax.set_xlabel('X (cm)', fontsize=20)
            ax.set_ylabel('Y (cm)', fontsize=20)
            ax.tick_params(labelsize=14)
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_aspect('equal')

    def create_boxplots(self, filepaths, labels, axs, selected_option_numbers):
        """Create boxplots for selected data."""
        for i, file_path in enumerate(filepaths):
            print(os.path.basename(file_path))
            if os.path.basename(file_path) != "rate.csv":
                data_frame = pd.read_csv(file_path)
                filtered_columns = [col for col in selected_option_numbers
                                    if col in data_frame.columns]

                if not filtered_columns:
                    print(f"Warning: None of the selected "
                          f"columns are present in '{file_path}'.")
                    continue

                axs[i].boxplot([data_frame[col].dropna()
                                for col in filtered_columns], showfliers=False)
                axs[i].set_xlabel('Wafer', fontsize=16)
                axs[i].set_ylabel(labels[i], fontsize=16)
                axs[i].set_xticklabels(filtered_columns)
            else:
                data_frame = pd.read_csv(file_path)
                filtered_df = data_frame[
                    data_frame["Slot"].astype(str).isin(selected_option_numbers)]
                axs[i].scatter(filtered_df["Slot"],
                               filtered_df["Cov. rate"], color='blue')
                axs[i].set_xlabel('Slot', fontsize=16)
                axs[i].set_ylabel("Coverage Rate (%)", fontsize=16)
        plt.tight_layout()
