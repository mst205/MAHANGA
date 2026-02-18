import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.cm as cm_mpl
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os
import re
from PyPDF2 import PdfMerger
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from io import BytesIO, StringIO
from pathlib import Path
import pickle

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# plt.style.use('dark_background')
sns.set_palette("muted")

#===============================================================================================================================
# Small Task Functions
#===============================================================================================================================

def ordinal(n):
    """
    Returns a string of any integer with the appropriate ordinal.
    Inputs:
        n = any integer
    Outputs:
        a string of n with the appropriate ordinal
    """
    # Get numbers with th ordinal
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    # Get numbers with other ordinals
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


#===============================================================================================================================
# Data Import and Cleaning Functions
#===============================================================================================================================

def readin_binary_data(path, file1, file2, single_star=False, print_summary=True):
    """
    Reads in plot data from the given STARS code output files, and extracts key information.
    Optionally prints a summary of the key information.
    Inputs:
        path = a string of the path to the folder containing the files
        file1 = a string of the filename
        file2 = a string of the filename
        single_star = a boolean indicating whether the file reads a single star simulation - default is False
        print_summary = a boolean indicating whether a summary of the simulations is printed - default is True
    Outputs:
        star1 = a dataframe containing data from file1
        star2 = a dataframe containing data from file2
    """
    
    if single_star:
        star1 = pd.read_csv(f'{path}{file1}', sep="\s+", header=None, engine='python', encoding='latin-1')
        star2 = pd.DataFrame()
        mass1, period = star1[5][0], star1[35][0]
        if print_summary:
            print(f"This star is initially {mass1:.1f}M, with an orbital period of {period:.4g} days. \n")

    else:
        star1 = pd.read_csv(f'{path}{file1}', sep="\s+", header=None, engine='python', encoding='latin-1')
        star2 = pd.read_csv(f'{path}{file2}', sep="\s+", header=None, engine='python', encoding='latin-1')
        mass1, mass2, period = star1[5][0], star2[5][0], star1[35][0]

        if print_summary:
            print(f"This system initially has a {mass1:.1f}M star and a {mass2:.1f}M star, with an orbital period of {period:.4g} days. \n")
    return star1, star2



def combine_simulations(star1, star2, exts, star_to_add_ext=2, data='plot'):
    """
    Combines a series of simulations for one system into the complete lifetimes of each star
    Inputs:
        star1 = the primary star from the first simulation
        star2 = the secondary star from the second simulation
        exts = a list of extensions to add, in the order they are to be added
        star_to_add_ext = indicates which star to add the extensions to, can be either 1 or 2
    Outputs:
        star1_complete = a dataframe of the combined and cleaned star1 data
        star2_complete = a dataframe of tHhe combined and cleaned star2 data
    """
    if data in ['plot', 'Plot', 'plotfile']:
        ts_col = 0
    elif data in ['out', 'Out', 'outfile']:
        ts_col = 'timestep'
    else:
        print('Invalid data type. Please choose either \'plot\' or \'out\'')
        return
    
    if star_to_add_ext == 1:
        
        star2_complete = star2.drop_duplicates(subset=ts_col, keep='last').sort_values(ts_col).reset_index(drop=True)
        
        all_star1s = [star1] + exts
        star1_complete = pd.concat(all_star1s, ignore_index=True)
        star1_complete = star1_complete.drop_duplicates(subset=ts_col, keep='last').sort_values(ts_col).reset_index(drop=True)

    elif star_to_add_ext == 2:
        
        star1_complete = star1.drop_duplicates(subset=ts_col, keep='last').sort_values(ts_col).reset_index(drop=True)
        
        all_star2s = [star2] + exts
        
        star2_complete = pd.concat(all_star2s, ignore_index=True)
        star2_complete = star2_complete.drop_duplicates(subset=ts_col, keep='last').sort_values(ts_col).reset_index(drop=True)

    else:
        
        print("Invalid extension: Extension not applied. Please ensure star_to_add_ext is 1 or 2")
        star1_complete = star1
        star2_complete = star2

    return star1_complete, star2_complete



def get_out_file_data(path, star_type, data_needed = 'MHe'):
    """
    Gets the data from the out file of the indicated thing from data_needed, returns as a dataframe with the model timestep.
    Currently can only do MHe but I will update!
    """

    if data_needed == 'MHe':
        row_num = 2
        col_num = 2
    else:
        print('That data type is not yet supported. Currently supported: MHe')
    
    base_path = Path(path)
    
    df_dicts = []
    
    if star_type in ['Primary', 'primary', '1', 1]:
        full_path = base_path / 'out'
        type_path = ''
    elif star_type in ['Secondary', 'secondary', '2', 2]:
        full_path = base_path / 'out2'
        type_path = '2'
    else:
        print('ERROR: star_type invalid, please choose \'primary\' or \'secondary\'.')
        return
    
    phrases = ['dt/age/MH/MHe']
    number_pattern = r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?'

    with open(full_path, 'r', encoding='utf-8') as out:

        for line in out:
            # Check for matches
            if any(phrase in line for phrase in phrases):
                # print(line)
                
                ts_numbers = re.findall(number_pattern, line)
                timestep_num = float(ts_numbers[0])
                i = 0
                while i < row_num:
                    next(out)
                    i += 1
                data_line = next(out)
                # print(data_line)
                data_numbers = re.findall(number_pattern, data_line)
                data_val = float(data_numbers[col_num])

                # print('timestep:', timestep_num, ', MHe:', data_val, '\n\n\n\n')
                
                df_dicts.append({'timestep':timestep_num, data_needed:data_val})


    df = pd.DataFrame(df_dicts)
    df = df.drop_duplicates(subset='timestep', keep='last').sort_values('timestep').reset_index(drop=True)
    return df



def get_out_full_mesh_models(file_path, lines_after=199):
    """
    Returns lists of output blocks from the STARS code out file.
    
    This function searches for two different key phrases and captures the lines
    that follow them. It correctly handles cases where one key phrase appears
    while the program is counting lines for another.
    
    Parameters:
    - file_path (str): Path to the specific file.
    - lines_after (int): Number of lines to include after a match for the first key phrase.
    
    Returns:
    - A tuple of three lists:
      - first_lot_matches (list): A list of strings, each being a full block matching the first phrase.
      - model_nums (list): A list of strings, each being a specific line captured after each 'first_lot' block.
      - second_lot_matches (list): A list of strings, each being a full block matching the second phrase.
    """

    if lines_after == 199:
        extra = 241
    elif lines_after == 499:
        extra = 601
    else:
        print('ERROR: Please choose either 199 or 499 mesh point models.')
        return
    first_lot_matches = []
    second_lot_matches = []
    model_nums = []
    
    collecting_state = 'none'
    lines_to_collect = 0
    extra_line_count = 0
    temp_buffer = []

    phrases1 = ['H1        He4        C12        N14']
    phrases2 = ['kappa      grada      gr-ga        m']

    with open(file_path, 'r', encoding='utf-8') as out:
        for line in out:

            if extra_line_count > 0:
                extra_line_count -= 1
                if extra_line_count == 0:
                    model_nums.append(line.strip())

            # --- Logic Part 2: Handle active block collection ---
            if collecting_state == 'first':
                temp_buffer.append(line)
                lines_to_collect -= 1
                if lines_to_collect == 0:
                    first_lot_matches.append("".join(temp_buffer))
                    collecting_state = 'none'
                    extra_line_count = extra
                continue

            elif collecting_state == 'second':
                temp_buffer.append(line)
                lines_to_collect -= 1
                if lines_to_collect == 0:
                    second_lot_matches.append("".join(temp_buffer))
                    collecting_state = 'none'
                continue
                
            if any(phrase in line for phrase in phrases1):
                temp_buffer = [line]
                lines_to_collect = lines_after
                collecting_state = 'first'
            elif any(phrase in line for phrase in phrases2):
                temp_buffer = [line]
                lines_to_collect = lines_after
                collecting_state = 'second'

    return first_lot_matches, second_lot_matches, model_nums




def convert_strings_to_plot_arrays(model_str_list, cols, model_num_strs=None):
    """
    Converts strings of models from out files into arrays for plotting.

    This function can optionally extract model numbers if they are provided.

    Parameters:
    - model_str_list (list): The list of model strings from get_out_models.
    - cols (list): A list of the names of the desired columns (e.g., ['log_T', 'log_R']).
    - model_num_strs (list, optional): The list of strings containing model numbers. 
                                       If None, this step is skipped. Defaults to None.

    Returns:
    - A tuple of two NumPy arrays: (plots, mod_nums)
      - plots: An array of shape (N, C, L) where N is the number of models,
               C is the number of columns, and L is the length of each column.
      - mod_nums: An array of the model numbers. Will be empty if model_num_strs is not provided.
    """
    plots = []
    mod_nums = []

    # Enumerate to get an index 'i' for accessing the corresponding model number string
    for i, model_string in enumerate(model_str_list):
        # --- Optional Step: Process model numbers ---
        # This block only runs if model_num_strs is provided
        if model_num_strs:
            try:
                # Use the corresponding model number string
                df_modnum = pd.read_csv(StringIO(model_num_strs[i]), sep=r"\s+", engine='python', header=None)
                modnum = df_modnum.iloc[0, 0]
                mod_nums.append(modnum)
            except (IndexError, pd.errors.EmptyDataError):
                # Handle cases where a model string might not have a matching number string
                print(f"Warning: Could not find or parse model number for model string index {i}. Appending NaN.")
                mod_nums.append(np.nan)

        # --- Core Step: Process the main model data ---
        # This part runs for every model string in the list
        df = pd.read_csv(StringIO(model_string), sep=r"\s+", engine='python')
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        final_cols = []
        for col in cols:
            # Check if the column exists to prevent errors
            if col in df.columns:
                vals = df[col].values
                final_cols.append(vals)
            else:
                print(f"Warning: Column '{col}' not found in model string index {i}. Appending empty array.")
                # Append an empty array or an array of zeros of the expected length
                # For simplicity, we'll note it here. A robust implementation might need a placeholder.
                pass
    
        plots.append(final_cols)
    
    # Convert lists to NumPy arrays for easier manipulation later
    plots = np.array(plots)
    mod_nums = np.array(mod_nums)
    
    return plots, mod_nums



# def convert_strings_to_plot_arrays(model_str_list, model_num_strs, cols):
#     """
#     Converts the strings of models from the out files into an array containing the selected x and y values from all the models.

#     Parameters:
#     - model_str_list = the list of the strings of all the models from get_out_models function
#     - cols = a list of the names of the desired columns

#     Returns:
#     - An array of size (N, 2, 199) OR (N, 2, 499) with all the string models converted to N sets of [x, y] arrays
#     """
#     plots = []
#     mod_nums = []
#     for i, model in enumerate(model_str_list):
#         df_modnum = pd.read_csv(StringIO(model_num_strs[i]), sep="\s+", engine='python', header=None)
#         modnum = df_modnum.iloc[0,0]
#         mod_nums.append(modnum)

        
#         df = pd.read_csv(StringIO(model), sep="\s+", engine='python')
#         df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

#         final_cols = []
#         for col in cols:
#             vals = df[col].values
#             final_cols.append(vals)
    
#         plots.append(final_cols)
    
    
#     plots = np.array(plots)
#     mod_nums = np.array(mod_nums)
#     return plots, mod_nums



# def get_out_models(file_path, lines_after=199):
#     """
#     Returns a list of output blocks from the STARS code out file, along with a specified line after the block.
    
#     Each block starts with a match to a key phrase and includes the following 'lines_after' lines,
#     plus one additional line two lines after the block.
    
#     Parameters:
#     - file_path: path to the specific file
#     - lines_after: number of lines to include after a match
    
#     Returns:
#     - A tuple of two lists:
#       - A list of strings, each string being one full matched block
#       - A list of strings, each string being the line captured after each block
#     """
#     all_matches = []       # Stores all match blocks as strings
#     model_nums = []        # Stores the additional lines
#     print_next = 0
#     collecting = False
#     extra_line_count = 0
    
#     phrases = ['H1        He4        C12        N14']

#     with open(file_path, 'r', encoding='utf-8') as out:
#         temp_buffer = []

#         for line in out:
#             if collecting:
#                 temp_buffer.append(line)
#                 print_next -= 1
#                 if print_next == 0:  # Finished collecting the main block
#                     all_matches.append("".join(temp_buffer))  # Save full block
#                     collecting = False
#                     extra_line_count = 241  # Set to capture the line two lines after the block

#             elif extra_line_count > 0:
#                 extra_line_count -= 1
#                 if extra_line_count == 0:
#                     model_nums.append(line.strip())

#             if any(phrase in line for phrase in phrases):
#                 temp_buffer = [line]
#                 print_next = lines_after
#                 collecting = True

#     return all_matches, model_nums



#===============================================================================================================================
# Analysis Functions
#===============================================================================================================================

def save_text_as_image(text, savepath, width=2480, font_size=29):
    """
    Saves the given text as a PDF image.
    Inputs:
        text = a string of text to save as an image
        savepath = path to save the image to
        width = width of the image in pixels
        fontsize = size of the font for the text
    """
    # Set up font and image size
    try:
        font = ImageFont.truetype("C:\\Windows\\Fonts\\consola.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        
    wrapper = textwrap.TextWrapper(width=160)
    lines = []
    for paragraph in text.splitlines():
        lines.extend(wrapper.wrap(paragraph) or [""])  # Keep empty lines

    # Calculate image height
    line_height = font.getbbox("A")[3] + 4
    img_height = line_height * len(lines) + 20

    # Create the image
    img = Image.new("RGB", (width, img_height), color="black")
    draw = ImageDraw.Draw(img)

    # Draw the text line by line
    y = 10
    for line in lines:
        draw.text((10, y), line, fill="white", font=font)
        y += line_height

    # Save as PDF
    img.convert('RGB').save(savepath, format='PDF', dpi=(200, 200))



def check_conditions(star1, star2, text_block):
    """
    Checks if H and He concentration in the core has gone to zero, and if the core temperature is enough to begin C burning.
    Inputs:
        text_block = a block of text from the out file containing the central concentrations
    Outputs:
        result = a flag indicating whether the simulation passed (went supernova) or failed
        H_core = concentration of H at the center of the star
        He_core = concentration of He at the center of the star
        temp_core = temperature at the center of the star
    """
    lines = text_block.strip().splitlines()

    if len(lines) < 2:
        print("ERROR: Not enough lines in final output to check.")
        return

    second_line = lines[1]

    # Regex to extract all numbers including scientific notation
    number_pattern = r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?'
    numbers = re.findall(number_pattern, second_line)

    try:
        H_core = abs(float(numbers[6]))    # Hydrogen concentration in the core
        He_core = abs(float(numbers[7]))    # Helium concentration in the core
        temp_core = abs(float(numbers[15]))  # Temperature in the core (log)
        psi = float(numbers[13])    # Degeneracy in the core (log)

        # Check conditions
        condition_1 = H_core == 0.0
        condition_2 = He_core == 0.0
        condition_3 = temp_core >= 8.8
        condition_4 = psi >= 50.0


        # First check for the merger condition
        if len(star1) == 1:
            final_binary_sep = star1.iloc[-1,36]
        else:
            final_binary_sep = star1.iloc[-2,36]
        final_radius_star1 = 10**star1.iloc[-1,2]
        final_radius_star2 = 10**star2.iloc[-1,2]
        min_dist_between = final_radius_star1 + final_radius_star2

        if final_binary_sep < min_dist_between:
            result = 4
        
        # Now check other conditions
        elif condition_1 and condition_2:
            # If we are here, we know H=0 and He=0. Now we differentiate the success types.
            
            # Check for the primary PASS condition (High Temp)
            if condition_3:
                result = 1 # PASS
                
            # If not high temp, check for the secondary PASS NO SUPERNOVA condition (High Degeneracy)
            elif condition_4: 
                result = 3 # PASS NO SUPERNOVA
                
            # If H=0, He=0, but temp is low AND degeneracy is low, it's a weak pass.
            else:
                result = 2 # WEAK PASS
        
        # If the fundamental requirement (H=0 and He=0) was not met, it's an immediate fail.
        else:
            result = 0 # FAIL

    except IndexError:
        print("ERROR: Not enough numeric values in the second row.")
        result = 0
    except ValueError:
        print("ERROR: Could not convert one of the values to float.")
        result = 0

    return result, H_core, He_core, temp_core, psi



def write_pass_fail_file(result, H_core, He_core, temp_core, psi, savepath):
    """
    Creates a file called PASS or FAIL depending on the simulation, along with the 
    H, He, temperature, and degeneracy information at the core.
    
    Inputs:
        result (int): A flag indicating the simulation outcome.
                      1: PASS
                      2: WEAK_PASS
                      3: PASS_NO_SUPERNOVA
                      4: MERGER
                      11: PASS_REVERSE
                      12: WEAK_PASS_REVERSE
                      13: PASS_REVERSE_NO_SUPERNOVA
                      0: FAIL
        H_core (float): Concentration of H at the center of the relevant star.
        He_core (float): Concentration of He at the center of the relevant star.
        temp_core (float): Temperature (log) at the center of the relevant star.
        psi (float): Degeneracy (log) at the center of the relevant star.
        savepath (str): The path to the directory where the file should be saved.
    """
    
    # Determine the file flag, print message, and which star's data is being used 
    if result == 1:
        flag = 'PASS'
        star_label = 'Primary'
    elif result == 2:
        flag = 'WEAK_PASS'
        star_label = 'Primary'
    elif result == 3:
        flag = 'PASS_NO_SUPERNOVA'
        star_label = 'Primary'
    elif result == 4:
        flag = 'MERGER'
        star_label = 'Primary'
    elif result == 11:
        flag = 'PASS_REVERSE'
        star_label = 'Secondary'
    elif result == 12:
        flag = 'WEAK_PASS_REVERSE'
        star_label = 'Secondary'
    elif result == 13:
        flag = 'PASS_REVERSE_NO_SUPERNOVA'
        star_label = 'Secondary'
    else:
        flag = 'FAIL'
        star_label = 'Primary'
    
    to_write = flag.replace('_', ' ')

    # Prepare formatted output string with the dynamic label
    output_text = (
        f"{flag}\n"
        f"{star_label} star key stats:\n" 
        f"H_core = {H_core:.5f}\n"
        f"He_core = {He_core:.5f}\n"
        f"Temp_core = 1e{temp_core:.4f} K\n"
        f"Psi = 1e{psi:.4f}\n"
    )
    
    # Use pathlib to robustly create the full file path
    save_directory = Path(savepath)
    save_directory.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    filename = save_directory / f"{flag}.txt"
    
    # Write to the file
    with open(filename, "w") as f:
        f.write(output_text)

    print(f"Simulation result: {to_write}\n")



def get_final_out(star1, star2, path, savepath, star_type, lines_after=8, save=True):
    """
    Prints out the final output model from the STARS code out file.
    Inputs:
        path = a string of the path to the file (NOT including the file)
        savepath = a string of the path to the save location
        star_type = either 'primary' or 'secondary'. This determines which file from the folder is used (out for the primary or out2 for the secondary)
    Outputs:
        result = a flag indicating whether the simulation passed  or failed
        H_core = concentration of H at the center of the star
        He_core = concentration of He at the center of the star
        temp_core = temperature at the center of the star
    """
    last_match_buffer = []  # Stores the last match and its lines
    print_next = 0          # Countdown for how many lines to collect
    collecting = False      # Flag to indicate we're collecting after a match
    
    if star_type in ['Primary', 'primary', '1', 1]:
        full_path = f'{path}out'
        type_path = ''
        title = 'Final output for primary star:'
    elif star_type in ['Secondary', 'secondary', '2', 2]:
        full_path = f'{path}out2'
        type_path = '2'
        title = 'Final output for secondary star:'
    else:
        print('ERROR: star_type invalid, please choose \'primary\' or \'secondary\'.')
        return
    
    # phrases = ['1  dt', '2  dt', '3  dt', '4  dt', '5  dt', '6  dt', '7  dt', '8  dt', '9  dt', '0  dt']
    phrases = ['dt/age/MH/MHe']#, 'tn/tKH/Mb',  'P/rlf/dM', 'LH/LHe/LC', 'Lth/Lnu/m']

    with open(full_path, 'r', encoding='utf-8') as out:

        for line in out:
            # Check for matches
            if any(phrase in line for phrase in phrases):
                temp_buffer = []            # Temporary buffer to collect lines after a match
                print_next = lines_after    # Reset counter
                collecting = True           # Start collecting

            # If there is a match, collect the lines, otherwise move on to the next
            if collecting:
                temp_buffer.append(line)
                print_next -= 1
                if print_next == 0:
                    last_match_buffer = temp_buffer  # Store this as the latest match
                    collecting = False


    # After file is fully read
    if last_match_buffer:
        text_out = "".join(last_match_buffer)
        result, H_core, He_core, temp_core, psi = check_conditions(star1, star2, text_out)

        if save:
            text_to_save = f"Final output for the {star_type} star: \n" + text_out
            print(text_to_save)
            save_text_as_image(text_to_save, f'{savepath}Final_out{type_path}.pdf')

    else:
        print("No matches found.")
        result, H_core, He_core, temp_core, psi = 0, np.nan, np.nan, np.nan, np.nan
    return result, H_core, He_core, temp_core, psi



def contact_phase_check(star1, star2, savepath):
    """Creates a file containing the total length of contact phase during a simulation"""

    # Get radii
    binarysep = star1[36] #Binary Separation
    logradius1 = star1[2]
    radius1 = 10**logradius1 #Star radius
    logradius2 = star2[2]
    radius2 = 10**logradius2

    # Get roche radii
    dM1 = star1[5]
    dM2 = star2[5]
    q1 = dM1/dM2 #The mass ratio
    q2 = dM2/dM1
    roche1 = binarysep * ((0.49 * q1**(2/3)) / (0.6 * q1**(2/3) + np.log(1 + q1**(1/3)))) #Effective Roche Lobe
    roche2 = binarysep * ((0.49 * q2**(2/3)) / (0.6 * q2**(2/3) + np.log(1 + q2**(1/3))))

    # Check if roche lobes are filled
    rlof1 = radius1 - roche1
    rlof1_mask = rlof1 >= 0
    rlof2 = radius2 - roche2
    rlof2_mask = rlof2 >= 0

    # Check for contact phase
    contact_phase_mask = rlof1_mask * rlof2_mask
    
    if np.any(contact_phase_mask):
        print('Contact binary phase')
        contact_phase_flag = True
        filename_flag = 'CONTACT_PHASE'
    else:
        print('No contact binary phase')
        contact_phase_flag = False
        filename_flag = 'NO_CONTACT_PHASE'

    num_timesteps_in_contact = sum(contact_phase_mask)
    timestep_percent_in_contact = num_timesteps_in_contact / len(contact_phase_mask)

    all_dt = star1[26]
    time_spent_in_contact = np.sum(all_dt * contact_phase_mask)
    kyrs_in_contact = time_spent_in_contact / 1e3

    output_text = (
        f"Contact phase: {contact_phase_flag}\n"
        f"Number of timesteps in contact: {num_timesteps_in_contact}\n" 
        f"Percentage of timesteps in contact: {timestep_percent_in_contact}\n" 
        f"Kiloyears spent in contact: {kyrs_in_contact}\n" 
    )
    
    # Use pathlib to robustly create the full file path
    save_directory = Path(savepath)
    save_directory.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    filename = save_directory / f"{filename_flag}.txt"
    
    # Write to the file
    with open(filename, "w") as f:
        f.write(output_text)

    return contact_phase_flag, kyrs_in_contact



def get_max_mass(star1, star2, savepath):
    """Returns a flag which tells you whether the system had a contact phase"""

    mass1 = star1[5]
    mass2 = star2[5]

    prim_max_mass = np.max(mass1)
    sec_max_mass = np.max(mass2)

    filename = 'MAX_MASS.txt'
    

    output_text = (
        f"Primary maximum mass (M_solar): {prim_max_mass}\n"
        f"Secondary maximum mass (M_solar): {sec_max_mass}\n"
    )
    
    # Use pathlib to robustly create the full file path
    save_directory = Path(savepath)
    save_directory.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    filename = save_directory / f"{filename}"
    
    # Write to the file
    with open(filename, "w") as f:
        f.write(output_text)

    return prim_max_mass, sec_max_mass

    

def get_max_He_mass(primary_data, secondary_data, th_data, savepath):
    """Returns a flag which tells you whether the system had a contact phase"""

    He_mass1 = primary_data['MHe']
    He_mass2 = secondary_data['MHe']
    He_mass_th = th_data['MHe']

    prim_max_He_mass = np.nanmax(He_mass1)
    sec_max_He_mass = np.nanmax(He_mass2)
    th_max_He_mass = np.nanmax(He_mass_th)

    filename = 'He_MAX_MASS.txt'
    

    output_text = (
        f"Primary maximum He mass (M_solar): {prim_max_He_mass}\n"
        f"Secondary maximum He mass (M_solar): {sec_max_He_mass}\n"
        f"Thermohaline star maximum He mass (M_solar): {th_max_He_mass}\n"
    )
    
    # Use pathlib to robustly create the full file path
    save_directory = Path(savepath)
    save_directory.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    filename = save_directory / f"{filename}"
    
    # Write to the file
    with open(filename, "w") as f:
        f.write(output_text)

    return prim_max_He_mass, sec_max_He_mass, th_max_He_mass



def get_max_He_surface_abundance(star1, star2, savepath):
    """Returns maximum He surface abundance for Primary and Secondary"""

    X_He1 = star1[28]
    X_He2 = star2[28]

    prim_max_X_He = np.max(X_He1)
    sec_max_X_He = np.max(X_He2)

    filename = 'MAX_X_He_SURFACE.txt'
    

    output_text = (
        f"Primary maximum He surface abundance (%): {prim_max_X_He}\n"
        f"Secondary maximum He surface abundance (%): {sec_max_X_He}\n"
    )
    
    # Use pathlib to robustly create the full file path
    save_directory = Path(savepath)
    save_directory.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    filename = save_directory / f"{filename}"
    
    # Write to the file
    with open(filename, "w") as f:
        f.write(output_text)

    return prim_max_X_He, sec_max_X_He



def get_min_He_surface_abundance(star1, star2, savepath):
    """Returns minimum He surface abundance for Primary and Secondary"""

    X_He1 = star1[28]
    X_He2 = star2[28]

    

    prim_min_X_He_arg = np.argmin(X_He1)
    sec_min_X_He_arg = np.argmin(X_He2)

    prim_min_X_He = X_He1[prim_min_X_He_arg]
    sec_min_X_He = X_He2[sec_min_X_He_arg]

    filename = 'MIN_X_He_SURFACE.txt'
    

    output_text = (
        f"Primary minimum He surface abundance (%): {prim_min_X_He}\n"
        f"Primary minimum He surface abundance arg: {prim_min_X_He_arg}\n"
        f"Secondary minimum He surface abundance (%): {sec_min_X_He}\n"
        f"Secondary minimum He surface abundance arg: {sec_min_X_He_arg}\n"
    )
    
    # Use pathlib to robustly create the full file path
    save_directory = Path(savepath)
    save_directory.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    filename = save_directory / f"{filename}"
    
    # Write to the file
    with open(filename, "w") as f:
        f.write(output_text)

    return prim_min_X_He, sec_min_X_He, prim_min_X_He_arg, sec_min_X_He_arg



def find_helium_flash_timestep(star, plot_figure=False):
    """
    Finds the timestep where He core mass suddenly increases (gradient > 0.1 over 10 points).
    
    Parameters:
    star (pd.DataFrame): DataFrame with timestep in column 0 and He core mass in column 6
    
    Returns:
    int/float: The timestep where the sudden increase occurs, or None if not found
    """
    # Get the He core mass column (7th column, index 6)
    he_core_mass = star.iloc[:, 6].values
    
    # Calculate gradient over 10 points (current + next 9)
    gradients = []
    for i in range(len(he_core_mass) - 9):  # Stop 9 points before the end
        # Calculate the difference between point i+9 and point i
        gradient = (he_core_mass[i + 9] - he_core_mass[i])  # Average gradient per step
        gradients.append(gradient)
    
    gradients = np.array(gradients)
    
    # Find where gradient is greater than 0.1
    sudden_increase_indices = np.where(gradients > 0.1)[0]
    
    if len(sudden_increase_indices) > 0:
        # Get the first occurrence
        first_increase_index = sudden_increase_indices[0]
        
        # We want the timestep where the increase starts to be significant
        # This would be at the beginning of the 10-point window
        timestep_index = first_increase_index
        
        # Get the timestep from the first column
        timestep = star.iloc[timestep_index, 0]
        
        # print(f"Found sudden He core mass increase starting at timestep {timestep}")
        # print(f"  Mass at start: {he_core_mass[first_increase_index]}")
        # print(f"  Mass after 9 steps: {he_core_mass[first_increase_index + 9]}")
        # print(f"  Total change: {he_core_mass[first_increase_index + 9] - he_core_mass[first_increase_index]:.4f}")

        if plot_figure:
            plt.plot(star[0], star[6])
            ymin, ymax = plt.ylim()
            plt.vlines(timestep, ymin, ymax, colors='grey')
            plt.show()

        
        return timestep
    else:
        print("No sudden increase in He core mass found (gradient > 0.1 over 10 points)")
        return None



def find_end_of_He_burning_timestep(star, plot_figure=False):
    """
    Finds the timestep where He core mass suddenly increases (gradient > 0.1 over 10 points).
    
    Parameters:
    star (pd.DataFrame): DataFrame with timestep in column 0 and He core mass in column 6
    
    Returns:
    int/float: The timestep where the sudden increase occurs, or None if not found
    """
    # Get the He core mass column (7th column, index 6)
    he_core_mass = star.iloc[:, 7].values
    
    # Calculate gradient over 10 points (current + next 9)
    gradients = []
    for i in range(len(he_core_mass) - 9):  # Stop 9 points before the end
        # Calculate the difference between point i+9 and point i
        gradient = (he_core_mass[i + 9] - he_core_mass[i])  # Average gradient per step
        gradients.append(gradient)
    
    gradients = np.array(gradients)
    
    # Find where gradient is greater than 0.1
    sudden_increase_indices = np.where(gradients > 0.1)[0]
    
    if len(sudden_increase_indices) > 0:
        # Get the first occurrence
        first_increase_index = sudden_increase_indices[0]
        
        # We want the timestep where the increase starts to be significant
        # This would be at the beginning of the 10-point window
        timestep_index = first_increase_index
        
        # Get the timestep from the first column
        timestep = star.iloc[timestep_index, 0]
        
        # print(f"Found sudden He core mass increase starting at timestep {timestep}")
        # print(f"  Mass at start: {he_core_mass[first_increase_index]}")
        # print(f"  Mass after 9 steps: {he_core_mass[first_increase_index + 9]}")
        # print(f"  Total change: {he_core_mass[first_increase_index + 9] - he_core_mass[first_increase_index]:.4f}")

        if plot_figure:
            plt.plot(star[0], star[6])
            ymin, ymax = plt.ylim()
            plt.vlines(timestep, ymin, ymax, colors='grey')
            plt.show()

        
        return timestep
    else:
        print("No sudden increase in He core mass found (gradient > 0.1 over 10 points)")
        return None



def thermohaline_lifetime_comparison(no_th_star_complete, th_star_complete, savepath):
    """
    A function that writes the lifetimes of the given stars and their difference (non_th - thermohaline) 
    AND ratio (thermohaline/non_th) to a file - for both complete lifetime AND main seqeunce lifetime.
    Inputs:
    - no_th_star_complete = a dataframe containing all the plot inforation for the no thermohaline star through its entire lifetime
    - th_star_complete = a dataframe containing all the plot inforation for the thermohaline star through its entire lifetime
    - savepath = a string to the path the text file will be saved to
    """
    # Complete lifetime
    no_th_lifetime = no_th_star_complete.iloc[-1,1]
    th_lifetime = th_star_complete.iloc[-1,1]

    lifetime_difference = no_th_lifetime - th_lifetime
    lifetime_ratio = th_lifetime / no_th_lifetime

    #Main sequence lifetime
    timestep_eoMS = find_helium_flash_timestep(no_th_star_complete)
    timestep_eoMS_th = find_helium_flash_timestep(th_star_complete)    
    print(f"timesteps: {timestep_eoMS}, {timestep_eoMS_th}")
    # plt.plot(no_th_star_complete[0], no_th_star_complete[6])
    # plt.plot(th_star_complete[0], th_star_complete[6])
    # plt.show()
    
    no_th_MS_lifetime = no_th_star_complete.iloc[timestep_eoMS,1]
    th_MS_lifetime = th_star_complete.iloc[timestep_eoMS_th,1]
    
    MS_lifetime_difference = no_th_MS_lifetime - th_MS_lifetime
    MS_lifetime_ratio = th_MS_lifetime / no_th_MS_lifetime
    

    output_text = (
        "Thermohaline star lifetime comparison\n"
        f"    - No thermohaline lifetime: {no_th_lifetime}\n"
        f"    - Thermohaline lifetime: {th_lifetime}\n" 
        f"    - Lifetime difference: {lifetime_difference}\n" 
        f"    - Lifetime ratio: {lifetime_ratio}\n" 
        "Main Sequence lifetime comparison\n"
        f"    - MS no thermohaline lifetime: {no_th_MS_lifetime}\n" 
        f"    - MS thermohaline lifetime: {th_MS_lifetime}\n" 
        f"    - MS lifetime difference: {MS_lifetime_difference}\n" 
        f"    - MS lifetime ratio: {MS_lifetime_ratio}\n" 
    )
    
    # Use pathlib to robustly create the full file path
    save_directory = Path(savepath)
    save_directory.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    filename = save_directory / "thermohaline_lifetime_comparison.txt"
    
    # Write to the file
    with open(filename, "w") as f:
        f.write(output_text)

    print(f'Thermohaline comparison file written to {filename}')



#===============================================================================================================================
# Plot Functions
#===============================================================================================================================

def plot_HR_diagrams(star1, star2, savepath, add_thermohaline=False, th_star=pd.DataFrame(), 
                     include_title=True, save_figure=True, xlims=None, figuresize=(10,8), textsize=None):
    """
    Makes a plot of the evolutionary tracks of both stars in the binary on an HR diagram.
    Inputs:
        star1 = dataframe for the primary star
        star2 = dataframe for the secondary star
        savepath = string of the path to save the graph to
        add_thermohaline = a boolean flag to indicate whether a thermohaline comparsion is added - default is False
        th_star = dataframe for the thermohaline star - default is an empty dataframe
        save_figure = a boolean to indicate whether the image is saved - default is True
        textsize = font size for all text elements - default is None (uses matplotlib default)
    """
    mass1, mass2, period = star1[5][0], star2[5][0], star1[35][0]
    
    log_Teff1, log_L1 = star1[3], star1[4]
    log_Teff2, log_L2 = star2[3], star2[4]
    
    plt.figure(figsize=figuresize)
    plt.plot(log_Teff1, log_L1, label=f'{mass1:.0f}M$_\odot$')
    plt.plot(log_Teff2, log_L2, label=f'{mass2:.0f}M$_\odot$')
    
    if add_thermohaline:
        mass_th = th_star[5][0]
        log_Teff_th, log_L_th = th_star[3], th_star[4]
        plt.plot(log_Teff_th, log_L_th, label=f'{mass_th:.0f}M$_\odot$ with thermohaline mixing')
    
    if xlims != None:
        plt.xlim(xlims[0], xlims[1])
    
    plt.gca().invert_xaxis()
    
    if textsize is not None:
        plt.xlabel(r'$\log_{10}(T_{\text{eff}}$[K]$)$', fontsize=textsize)
        plt.ylabel(r'$\log_{10}(L_*/L_{\text{sun}})$', fontsize=textsize)
        plt.tick_params(axis='both', which='major', labelsize=textsize)
        plt.legend(fontsize=textsize-4)
        if include_title:
            plt.title(f'Evolution of Stars in a {period:.3g} day Binary', fontsize=textsize)
    else:
        plt.xlabel(r'$\log_{10}(T_{\text{eff}}$[K]$)$')
        plt.ylabel(r'$\log_{10}(L_*/L_{\text{sun}})$')
        plt.legend()
        if include_title:
            plt.title(f'Evolution of Stars in a {period:.3g} day Binary')
    
    if save_figure:
        plt.savefig(f'{savepath}{mass1:.0f}M_{mass2:.0f}M_{np.log10(period):.2g}day_HR.pdf', 
                    dpi=300, bbox_inches='tight')
    plt.show()


def plot_change_in_mass(star1, star2, savepath, add_thermohaline=False, th_star=pd.DataFrame(), save_figure=True):
    """
    Plots the total binary mass and individual star masses against simulation time step.
    Inputs:
        star1 = dataframe for the primary star
        star2 = dataframe for the secondary star
        savepath = string of the path to save the graph to
        add_thermohaline = a boolean flag to indicate whether a thermohaline comparsion is added - default is False
        th_star = dataframe for the thermohaline star - default is an empty dartaframe
        save_figure = a boolean to indicate whether the image is saved - default is True
    """
    mass1, mass2, period = star1[5][0], star2[5][0], star1[35][0]
    
    dM1 = star1[5]
    timestep1 = star1[0]
    dM2 = star2[5]
    timestep2 = star2[0]

    if len(timestep1) <= len(timestep2):
        binary_dM = star1[37]
        timestep_DM = timestep1
    else:
        binary_dM = star2[37]
        timestep_DM = timestep2

    plt.figure(figsize=(10,8))
    plt.plot(timestep_DM, binary_dM, 'w--',label='Total Binary Mass')
    plt.plot(timestep1, dM1, label=f'{mass1:.0f}M$_\odot$ Star')
    plt.plot(timestep2, dM2, label=f'{mass2:.0f}M$_\odot$ Star')
    if add_thermohaline:
        mass_th = th_star[5][0]
        dM_th = th_star[5]
        timestep_th = th_star[0]
        
        plt.plot(timestep_th, dM_th, label=f'{mass_th:.0f}M$_\odot$ Star with thermohaline mixing')
    plt.xlabel('Time step')
    plt.ylabel('Mass [M$_\odot$]')
    plt.legend()
    plt.title(f'Change in Mass for a {mass1:.0f}M$_\odot$ and a {mass2:.0f}M$_\odot$ Binary')
    if save_figure:
        plt.savefig(f'{savepath}{mass1:.0f}M_{period:.1g}day_Change_in_Mass.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_radii(star1, star2, savepath, include_title=True, textsize=None, end_of_MS=(None, None), save_figure=True):
    """
    Makes plots of the radius of each star in the binary as well as their roche lobe radii and the
    binary separation against simulation timestep.
Inputs:
        star1 = dataframe for the primary star ONLY THE ORIGINAL SIMS FOR NOW
        star2 = dataframe for the secondary star ONLY THE ORIGINAL SIMS FOR NOW
        savepath = string of the path to save the graph to
        save_figure = a boolean to indicate whether the image is saved - default is True
    """
    mass1, mass2, period = star1[5][0], star2[5][0], star1[35][0]
    
    timestep1 = star1[0]
    radius1 = 10**star1[2]
    timestep2 = star2[0]
    radius2 = 10**star2[2]

    if len(timestep1) <= len(timestep2):
        binarysep = star1[36]
        timestep_DM = timestep1
    else:
        binarysep = star2[36]
        timestep_DM = timestep2

    dM1 = star1[5]
    dM2 = star2[5]
    q1 = dM1/dM2 #The mass ratio
    q2 = dM2/dM1
    roche1 = binarysep * ((0.49 * q1**(2/3)) / (0.6 * q1**(2/3) + np.log(1 + q1**(1/3)))) #Effective Roche Lobe
    roche2 = binarysep * ((0.49 * q2**(2/3)) / (0.6 * q2**(2/3) + np.log(1 + q2**(1/3))))

    plt.figure(figsize=(10,8))
    # plt.plot(timestep_DM, binarysep, 'w-', label='Binary Separation')

    plt.plot(timestep1, radius1, label=f'{mass1:.0f}M$_\odot$ Star Radius')
    plt.plot(timestep1, roche1, c='cadetblue', linestyle='--', label=f'{mass1:.0f}M$_\odot$ Roche Lobe')
    plt.plot(timestep2, radius2, label=f'{mass2:.0f}M$_\odot$ Star Radius')
    plt.plot(timestep2, roche2, c='goldenrod', linestyle='--', label=f'{mass2:.0f}M$_\odot$ Roche Lobe')
    if end_of_MS[0] != None:
        ymin, ymax = plt.ylim()
        plt.vlines(end_of_MS[0], ymin, ymax, colors='grey', linestyle=':', label=f'End of {end_of_MS[1]} Main Sequence')
        plt.ylim(ymin, ymax)



    if textsize is not None:
        plt.xlabel('Time Step', fontsize=textsize)
        plt.ylabel(r'Radius [R$_\odot$]', fontsize=textsize)
        plt.tick_params(axis='both', which='major', labelsize=textsize)
        legend_txtsize = textsize - 4
        plt.legend(framealpha=0.6, fontsize=legend_txtsize)
        if include_title:
            plt.title(f'Change in Radii of {mass1:.0f}M$_\odot$ and {mass2:.0f}M$_\odot$ Stars in a {period:.3g} day Binary', fontsize=textsize)
    else:
        plt.xlabel('Time Step')
        plt.ylabel(r'Radius [R$_\odot$]')
        plt.legend(framealpha=0.6)
        if include_title:
            plt.title(f'Change in Radii of {mass1:.0f}M$_\odot$ and {mass2:.0f}M$_\odot$ Stars in a {period:.3g} day Binary')

    # plt.yticks(np.arange(0, 18.1, 4))
    # plt.xticks([0, 3000, 6000, 9000])

            
    if save_figure:
        plt.savefig(f'{savepath}{mass2:.0f}M_{period:.1g}day_Roche.pdf', dpi=150, bbox_inches='tight')
    plt.show()


def kippenhahn(star1, star2, star2_th, savepath, save_figure=True):
    """Plots Kippenhahn diagrams"""    
    mass1, mass2, period = star1[5][0], star2[5][0], star1[35][0]
    
    timestep1 = star1[0]
    timestep2 = star2[0]
    timestep2_th = star2_th[0]
    
    total_mass1 = star1[5]
    total_mass2 = star2[5]
    total_mass2_th = star2_th[5]
    
    He_core1 = star1[6]
    He_core2 = star2[6]
    He_core2_th = star2_th[6]
    
    
    CO_core1 = star1[7]
    CO_core2 = star2[7]
    CO_core2_th = star2_th[7]


    mconv_star1 = np.abs(star1.iloc[:, 11:22:2])
    mconv_star1 = mconv_star1.loc[:, (mconv_star1 != 0).any(axis=0) & (mconv_star1 >= -1).all(axis=0)]
    
    mconv_star2 = np.abs(star2.iloc[:, 11:22:2])
    mconv_star2 = mconv_star2.loc[:, (mconv_star2 != 0).any(axis=0) & (mconv_star2 >= -1).all(axis=0)]
    
    mconv_star2_th = np.abs(star2_th.iloc[:, 11:22:2])
    mconv_star2_th = mconv_star2_th.loc[:, (mconv_star2_th != 0).any(axis=0) & (mconv_star2_th >= -1).all(axis=0)]

    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 5], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 1]) # Secondary
    ax2 = fig.add_subplot(gs[1, 1])#, sharex=ax1) # Secondary with th
    ax3 = fig.add_subplot(gs[:, 0]) # Primary
    # plt.setp(ax1.get_xticklabels(), visible=False)
    
    custom_lines_primary = [
        Line2D([0], [0], color='white', lw=2.5, label='Primary mass'),
        Line2D([0], [0], color='slateblue', lw=2.5, label='He core mass'),
        Line2D([0], [0], color='crimson', lw=2.5, label='CO core mass'),
        Line2D([0], [0], color='green', lw=2, label='Convective boundary')
    ]

    custom_lines_secondary = [
        Line2D([0], [0], color='white', lw=2.5, label='Secondary mass'),
        Line2D([0], [0], color='slateblue', lw=2.5, label='He core mass'),
        Line2D([0], [0], color='crimson', lw=2.5, label='CO core mass'),
        Line2D([0], [0], color='green', lw=2, label='Convective boundary')
    ]
    
    # Star2
    for i, mconv in enumerate(mconv_star2.columns):
        ax1.scatter(timestep2, abs(mconv_star2[mconv]), s=0.5, c='green', rasterized=True)
    ax1.plot(timestep2, total_mass2, c='white', label='Secondary mass', lw=2.5)
    ax1.plot(timestep2, He_core2, label='He core mass', c='slateblue', lw=2.5)
    ax1.plot(timestep2, CO_core2, label='CO core mass', c='crimson', lw=2.5)
    ax1.set_ylabel('Mass [M$_\\odot$]')
    ax1.set_title('Secondary Star')
    ax1.legend(handles=custom_lines_secondary, loc='upper left', framealpha=0.4)
    
    # Star2 thermohaline
    for i, mconv in enumerate(mconv_star2_th.columns):
        ax2.scatter(timestep2_th, abs(mconv_star2_th[mconv]), s=0.5, c='green', rasterized=True)
    ax2.plot(timestep2_th, total_mass2_th, c='white', label='Secondary mass', lw=2.5)
    ax2.plot(timestep2_th, He_core2_th, label='He core mass', c='slateblue', lw=2.5)
    ax2.plot(timestep2_th, CO_core2_th, label='CO core mass', c='crimson', lw=2.5)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Mass [M$_\\odot$]')
    ax2.set_title('Secondary Star with Thermohaline Mixing')
    ax2.legend(handles=custom_lines_secondary, loc='upper left', framealpha=0.4)
    
    #Star2
    for i, mconv in enumerate(mconv_star1.columns):
        ax3.scatter(timestep1, abs(mconv_star1[mconv]), s=0.5, c='green', rasterized=True)
    ax3.plot(timestep1, total_mass1, c='white',label='Primary Mass', lw=2.5)
    ax3.plot(timestep1, He_core1, label='He core mass', c='slateblue', lw=2.5)
    ax3.plot(timestep1, CO_core1, label='CO core mass', c='crimson', lw=2.5)
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Mass [M$_\odot$]')
    ax3.set_title('Primary Star')
    ax3.legend(handles=custom_lines_primary, loc='upper left', framealpha=0.4)
    
    fig.suptitle(f'Kippenhahn diagrams for a {period:.3g} day Binary with {mass1:.0f}M$_\\odot$ and {mass2:.0f}M$_\\odot$ Stars', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_figure:
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(f'{savepath}{mass1:.0f}M_{mass2:.0f}M_{period:.3g}day_Kippenhahn_diagram.pdf', dpi=300) 
    plt.show()



def kippenhahn_reverse(star2, star1, star1_th, savepath, save_figure=True):
    """Plots Kippenhahn diagrams"""    

    mass1, mass2, period = star1[5][0], star2[5][0], star1[35][0]
    
    timestep2 = star2[0]
    timestep1 = star1[0]
    timestep1_th = star1_th[0]
    
    total_mass2 = star2[5]
    total_mass1 = star1[5]
    total_mass1_th = star1_th[5]
    
    He_core2 = star2[6]
    He_core1 = star1[6]
    He_core1_th = star1_th[6]
        
    CO_core2 = star2[7]
    CO_core1 = star1[7]
    CO_core1_th = star1_th[7]

    mconv_star2 = np.abs(star2.iloc[:, 11:22:2])
    mconv_star2 = mconv_star2.loc[:, (mconv_star2 != 0).any(axis=0) & (mconv_star2 >= -1).all(axis=0)]
    
    mconv_star1 = np.abs(star1.iloc[:, 11:22:2])
    mconv_star1 = mconv_star1.loc[:, (mconv_star1 != 0).any(axis=0) & (mconv_star1 >= -1).all(axis=0)]
    
    mconv_star1_th = np.abs(star1_th.iloc[:, 11:22:2])
    mconv_star1_th = mconv_star1_th.loc[:, (mconv_star1_th != 0).any(axis=0) & (mconv_star1_th >= -1).all(axis=0)]

    fig = plt.figure(figsize=(14, 10))

    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 5], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 1]) # Primary
    ax2 = fig.add_subplot(gs[1, 1])#, sharex=ax1) # Primary with th
    ax3 = fig.add_subplot(gs[:, 0]) # Secondary
    
    custom_lines_primary = [
        Line2D([0], [0], color='white', lw=2.5, label='Primary mass'),
        Line2D([0], [0], color='slateblue', lw=2.5, label='He core mass'),
        Line2D([0], [0], color='crimson', lw=2.5, label='CO core mass'),
        Line2D([0], [0], color='green', lw=2, label='Convective boundary')
    ]

    custom_lines_secondary = [
        Line2D([0], [0], color='white', lw=2.5, label='Secondary mass'),
        Line2D([0], [0], color='slateblue', lw=2.5, label='He core mass'),
        Line2D([0], [0], color='crimson', lw=2.5, label='CO core mass'),
        Line2D([0], [0], color='green', lw=2, label='Convective boundary')
    ]
    
    # Star1
    for mconv in mconv_star1.columns:
        # Add rasterized=True to the complex scatter plot
        ax1.scatter(timestep1, abs(mconv_star1[mconv]), s=0.5, c='green', rasterized=True)
    ax1.plot(timestep1, total_mass1, c='white', lw=2.5)
    ax1.plot(timestep1, He_core1, c='slateblue', lw=2.5)
    ax1.plot(timestep1, CO_core1, c='crimson', lw=2.5)
    ax1.set(ylabel='Mass [M$_\\odot$]', title='Primary Star')
    ax1.legend(handles=custom_lines_primary, loc='upper left', framealpha=0.4)
    
    # Star1 thermohaline
    for mconv in mconv_star1_th.columns:
        # Add rasterized=True to the complex scatter plot
        ax2.scatter(timestep1_th, abs(mconv_star1_th[mconv]), s=0.5, c='green', rasterized=True)
    ax2.plot(timestep1_th, total_mass1_th, c='white', lw=2.5)
    ax2.plot(timestep1_th, He_core1_th, c='slateblue', lw=2.5)
    ax2.plot(timestep1_th, CO_core1_th, c='crimson', lw=2.5)
    ax2.set(xlabel='Time step', ylabel='Mass [M$_\\odot$]', title='Primary Star with Thermohaline Mixing')
    ax2.legend(handles=custom_lines_primary, loc='upper left', framealpha=0.4)

    # Star2
    for mconv in mconv_star2.columns:
        # Add rasterized=True to the complex scatter plot
        ax3.scatter(timestep2, abs(mconv_star2[mconv]), s=0.5, c='green', rasterized=True)
    ax3.plot(timestep2, total_mass2, c='white', lw=2.5)
    ax3.plot(timestep2, He_core2, c='slateblue', lw=2.5)
    ax3.plot(timestep2, CO_core2, c='crimson', lw=2.5)
    ax3.set(xlabel='Time step', ylabel='Mass [M$_\odot$]', title='Secondary Star')
    ax3.legend(handles=custom_lines_secondary, loc='upper left', framealpha=0.4)
    
    fig.suptitle(f'Kippenhahn diagrams for a {period:.3g} day Binary with {mass1:.0f}M$_\\odot$ and {mass2:.0f}M$_\\odot$ Stars', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_figure:
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(f'{savepath}{mass1:.0f}M_{mass2:.0f}M_{period:.3g}day_Kippenhahn_diagram.pdf', dpi=300) 

    plt.show()
    plt.close(fig)



def create_simulation_grid(path, savepath, include_title=True, textsize=None, save_figure=True):
    """
    Scans a nested directory of simulation results, parses parameters, 
    reads the outcome flag, and plots the results on a 2D grid with
    custom colors and hatching for reversed conditions.

    Inputs:
        path (str): The absolute path to the parent directory containing the mass_ratio_* folders.
        savepath (str): The path to the directory where the figure will be saved.
        save_figure (bool): If True, saves the figure to a file (defaults to True).
    """
    
    data = []
    
    flag_to_result = {
        
        'FAIL.txt': 0,
        'MERGER.txt': 4,
        'PASS.txt': 1,
        'WEAK_PASS.txt': 2,
        'PASS_NO_SUPERNOVA.txt': 3,
        'PASS_REVERSE.txt': 11,           # 1 + 10
        'WEAK_PASS_REVERSE.txt': 12,      # 2 + 10
        'PASS_REVERSE_NO_SUPERNOVA.txt': 13 # 3 + 10
    }
    
    print(f"Scanning simulation folders in: {Path(path).resolve()}")
    
    # This data scanning part remains the same as your original function
    for parent_folder_name in os.listdir(path):
        parent_folder_path = Path(path) / parent_folder_name
        if not parent_folder_path.is_dir(): continue
        try:
            mass_ratio = float(parent_folder_name.split('_')[0])
        except (IndexError, ValueError):
            print(f"Warning: Skipping parent folder with unexpected name format: {parent_folder_name}")
            continue

        for sim_folder_name in os.listdir(parent_folder_path):
            sim_folder_path = parent_folder_path / sim_folder_name
            if not sim_folder_path.is_dir(): continue
            try:
                parts = sim_folder_name.split('-')
                log_period = float(parts[-1])
                prim_mass = int(parts[0][1:])
            except (IndexError, ValueError):
                print(f"Warning: Skipping simulation folder with unexpected name format: {sim_folder_name} in {parent_folder_name}")
                continue

            result_code = -1 # Default for missing flag
            for flag_file, code in flag_to_result.items():
                if (sim_folder_path / flag_file).exists():
                    result_code = code
                    break
            
            data.append({
                'mass_ratio': mass_ratio,
                'log_period': log_period,
                'result': result_code
            })

    if not data:
        print("Error: No valid simulation folders found in the specified path.")
        return

    # Create the Grid (DataFrame)
    df = pd.DataFrame(data)
    grid_df = df.pivot_table(index='log_period', columns='mass_ratio', values='result')
    grid_df = grid_df.sort_index(ascending=False).sort_index(axis=1, ascending=True)

    # Plotting the Grid
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define your colors and labels in one place
    # Hatch is the "banding" pattern for reversed conditions
    style_map = {
        0: {'color': '#d62728', 'hatch': None, 'label': 'Fail'},
        4: {'color': '#f556e5', 'hatch': None, 'label': 'Merger'}, # Pink
        1: {'color': '#2ca02c', 'hatch': '///', 'label': 'Pass'}, # Green
        2: {'color': '#ffdd00', 'hatch': '///', 'label': 'Weak Pass'}, # Yellow
        3: {'color': '#1f77b4', 'hatch': '///', 'label': 'No Supernova'} # Blue
    }

    # Determine cell sizes for drawing rectangles
    x_coords = grid_df.columns.to_numpy()
    y_coords = grid_df.index.to_numpy()
    dx = (x_coords[1:] - x_coords[:-1]).mean() if len(x_coords) > 1 else 1
    dy = (y_coords[:-1] - y_coords[1:]).mean() if len(y_coords) > 1 else 1

    # Iterate over each cell in the grid and draw a styled rectangle
    for iy, y in enumerate(y_coords):
        for ix, x in enumerate(x_coords):
            result_code = grid_df.iloc[iy, ix]
            
            if pd.isna(result_code) or result_code == -1:
                continue # Skip empty cells

            is_reversed = result_code >= 10
            base_code = result_code % 10

            if base_code in style_map:
                style = style_map[base_code]
                face_color = style['color']
                hatch = style['hatch'] if is_reversed else None
                
                # Create and add the patch
                rect = Rectangle(
                    (x - dx / 2, y - dy / 2), dx, dy,
                    facecolor=face_color,
                    hatch=hatch,
                    edgecolor='black', # Add an edge for clarity
                    linewidth=0.5
                )
                ax.add_patch(rect)

    # --- Customizing the Plot ---
    legend_elements = []
    # Sort keys to ensure consistent legend order
    for code in style_map.keys():
        style = style_map[code]
        # Add the normal condition
        legend_elements.append(Patch(facecolor=style['color'], edgecolor='k', label=style['label']))
        # Add the reversed/banded condition if it has a hatch defined
        if style['hatch']:
            legend_elements.append(Patch(facecolor=style['color'], edgecolor='k', hatch=style['hatch'], label=f"{style['label']} Reverse"))
            

    if textsize is not None:
        if include_title:
            ax.set_title(f'Pass/Fail Grid for Systems with a {prim_mass}' + r'M$_\odot$ Primary Star', fontsize=textsize, pad=20)
        ax.set_xlabel('Mass Ratio (q = M$_2$/M$_1$)', fontsize=textsize)
        ax.set_ylabel('Binary Period [log(days)]', fontsize=textsize)
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=3, fontsize=textsize)
        plt.tick_params(axis='both', which='major', labelsize=textsize)

    else:
        if include_title:
            ax.set_title(f'Pass/Fail Grid for Systems with a {prim_mass}' + r'M$_\odot$ Primary Star', fontsize=16, pad=20)
        ax.set_xlabel('Mass Ratio (q = M$_2$/M$_1$)', fontsize=12)
        ax.set_ylabel('Binary Period [log(days)]', fontsize=12)
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=3)


    # Set ticks and limits based on the data grid
    ax.set_xticks(grid_df.columns)
    ax.set_yticks(grid_df.index)
    ax.set_xlim(grid_df.columns.min() - dx / 2, grid_df.columns.max() + dx / 2)
    ax.set_ylim(grid_df.index.min() - dy / 2, grid_df.index.max() + dy / 2)

    
    plt.tight_layout()
    plt.show()

    # --- Saving the Figure ---
    if save_figure:
        save_directory = Path(savepath)
        save_directory.mkdir(parents=True, exist_ok=True)
        figure_path = save_directory / 'simulation_grid.pdf'
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {figure_path}")

    print('Grid completed!')



def create_contact_grid_percent(path, savepath, save_figure=True):
    """
    Creates a grid showing what percentage of the simulation the stars were in contact
    Missing data points (where a results file is not found) are represented as NaN
    and colored black on the plot to avoid skewing the main colormap.

    Args:
        path (str): Root directory path containing simulation folders.
        savepath (str): Directory where the output figure should be saved.
        save_figure (bool): If True, saves the figure to a PDF. Defaults to False.
    """    
    data = []
    print(f"Scanning simulation folders in: {Path(path).resolve()}")

    # Scan nested directories, parsing mass ratio and log period from folder names.
    for parent_folder_name in os.listdir(path):
        parent_folder_path = Path(path) / parent_folder_name
        if not parent_folder_path.is_dir(): continue
        try:
            mass_ratio = float(parent_folder_name.split('_')[0])
        except (IndexError, ValueError):
            continue

        for sim_folder_name in os.listdir(parent_folder_path):
            sim_folder_path = parent_folder_path / sim_folder_name
            if not sim_folder_path.is_dir(): continue
            try:
                log_period = float(sim_folder_name.split('-')[-1])
            except (IndexError, ValueError):
                continue
            
            percent_in_contact = np.nan # Default code for "Incomplete" or "Failed"
            
            fail_file = sim_folder_path / 'FAIL.txt'
            contact_file = sim_folder_path / 'CONTACT_PHASE.txt'
            no_contact_file = sim_folder_path / 'NO_CONTACT_PHASE.txt'

            # Priority 1: Check for a failed simulation
            if fail_file.exists():
                percent_in_contact = np.nan
            # Priority 2: Check for a contact phase result
            elif contact_file.exists():
                try:
                    with open(contact_file, 'r') as f:
                        for line in f:
                            if line.strip().startswith('Percentage of timesteps in contact:'):
                                parts = line.strip().split(':', 1)
                                percent_in_contact = float(parts[1].strip())
                except Exception as e:
                    print(f"An error occurred with file {lifetime_comparison_file}: {e}")
            # Priority 3: Check for a no-contact phase result
            elif no_contact_file.exists():
                percent_in_contact = 0.0
            
            # If none of these files exist, result_code remains -1
            
            data.append({
                'mass_ratio': mass_ratio,
                'log_period': log_period,
                'percent_in_contact': percent_in_contact
            })

    if not data:
        print("Error: No valid simulation folders found in the specified path.")
        return

    # --- Data Structuring and Plotting ---
    # Pivot the collected data into a 2D grid suitable for a heatmap.
    df = pd.DataFrame(data)
    grid_df = df.pivot_table(index='log_period', columns='mass_ratio', values='percent_in_contact')
    
    # Sort axes for a clean, logical plot layout.
    grid_df = grid_df.sort_index(ascending=False).sort_index(axis=1, ascending=True)

    my_cmap = plt.colormaps['viridis'].copy()
    my_cmap.set_bad(color='black')  # Set the color for "bad" (NaN) values to black
    
    # Create the plot.
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        grid_df,
        ax=ax,
        cmap=my_cmap,
        linewidths=0.5,
        linecolor='grey',
        cbar_kws={'label': 'Percentage of simulation spent in contact'}    # Key: Add a title to the colorbar.
    )

    ax.set_title('Percentage of Simulation the Binary was in a Contact Phase, for Primaries of 20M$_\odot$', fontsize=16, pad=20)
    ax.set_xlabel('Mass Ratio (q = M$_2$/M$_1$)', fontsize=12)
    ax.set_ylabel('Binary Period [log(days)]', fontsize=12)

    plt.tight_layout()
    plt.show()

    # --- Saving the Figure ---
    if save_figure:
        save_directory = Path(savepath)
        save_directory.mkdir(parents=True, exist_ok=True)
        figure_path = save_directory / 'contact_grid_percent.pdf'
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {figure_path}")

    print('Grid completed!')



def create_contact_grid_kyr(path, savepath, save_figure=True):
    """
    Creates a grid showing how many kyrs in the simulation the stars were in contact.
    - Missing data (file not found) is colored black.
    - Simulations with no contact phase are colored dark red.
    - Simulations with a contact phase are colored on a logarithmic scale.

    Args:
        path (str): Root directory path containing simulation folders.
        savepath (str): Directory where the output figure should be saved.
        save_figure (bool): If True, saves the figure to a PDF. Defaults to True.
    """    
    data = []
    print(f"Scanning simulation folders in: {Path(path).resolve()}")

    # ... (Data scanning part remains unchanged) ...
    for parent_folder_name in os.listdir(path):
        parent_folder_path = Path(path) / parent_folder_name
        if not parent_folder_path.is_dir(): continue
        try:
            mass_ratio = float(parent_folder_name.split('_')[0])
        except (IndexError, ValueError):
            continue

        for sim_folder_name in os.listdir(parent_folder_path):
            sim_folder_path = parent_folder_path / sim_folder_name
            if not sim_folder_path.is_dir(): continue
            try:
                log_period = float(sim_folder_name.split('-')[-1])
            except (IndexError, ValueError):
                continue
            
            kyr_in_contact = np.nan 
            
            fail_file = sim_folder_path / 'FAIL.txt'
            contact_file = sim_folder_path / 'CONTACT_PHASE.txt'
            no_contact_file = sim_folder_path / 'NO_CONTACT_PHASE.txt'

            if fail_file.exists():
                kyr_in_contact = np.nan
            elif contact_file.exists():
                try:
                    with open(contact_file, 'r') as f:
                        for line in f:
                            if line.strip().startswith('Kiloyears spent in contact:'):
                                parts = line.strip().split(':', 1)
                                kyr_in_contact = float(parts[1].strip())
                except Exception as e:
                    print(f"An error occurred with file {contact_file}: {e}")
            elif no_contact_file.exists():
                kyr_in_contact = -1.0
            
            data.append({
                'mass_ratio': mass_ratio,
                'log_period': log_period,
                'kyr_in_contact': kyr_in_contact
            })

    if not data:
        print("Error: No valid simulation folders found in the specified path.")
        return

    # --- Data Structuring and Plotting ---
    df = pd.DataFrame(data)
    grid_df = df.pivot_table(index='log_period', columns='mass_ratio', values='kyr_in_contact')
    grid_df = grid_df.sort_index(ascending=False).sort_index(axis=1, ascending=True)

    # *** THIS IS THE CORRECTED LINE ***
    my_cmap = plt.colormaps['viridis'].copy()
    my_cmap.set_bad(color='black')
    my_cmap.set_under(color='darkred')
    
    # Prepare for Logarithmic Scale
    min_positive_val = grid_df[grid_df > 0].min().min()
    log_norm = None
    if pd.notna(min_positive_val):
        log_norm = LogNorm(vmin=min_positive_val, vmax=grid_df.max().max())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        grid_df,
        ax=ax,
        cmap=my_cmap,
        norm=log_norm,
        linewidths=0.5,
        linecolor='grey',
        cbar_kws={'label': 'Kyrs spent in contact [$10^3$yr]'}
    )

    ax.set_title('Duration of Contact Phase for Primaries of 20M$_\odot$', fontsize=16, pad=20)
    ax.set_xlabel('Mass Ratio (q = M$_2$/M$_1$)', fontsize=12)
    ax.set_ylabel('Binary Period [log(days)]', fontsize=12)
    
    # Fix Legend Positioning
    legend_elements = [
        Patch(facecolor='darkred', edgecolor='grey', label='No Contact Phase'),
        Patch(facecolor='black', edgecolor='grey', label='Failed / Missing Data')
    ]
    ax.legend(
        handles=legend_elements,
        title='Simulation Outcome',
        loc='lower left',
        bbox_to_anchor=(1.02, 1.0)
    )
    
    plt.tight_layout()
    plt.show()

    # --- Saving the Figure ---
    if save_figure:
        save_directory = Path(savepath)
        save_directory.mkdir(parents=True, exist_ok=True)
        figure_path = save_directory / 'contact_grid_kyr.pdf'
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {figure_path}")

    print('Grid completed!')



def create_max_mass_grid(path, savepath, save_figure=True, star_type='Secondary', mass_type='Total'):
    """
    Creates a grid showing the maximum mass of the chosen star type
    Missing data points (where a results file is not found) are represented as NaN
    and colored black on the plot to avoid skewing the main colormap.

    Args:
        path (str): Root directory path containing simulation folders.
        savepath (str): Directory where the output figure should be saved.
        save_figure (bool): If True, saves the figure to a PDF. Defaults to False.
        star_type (str): Either 'Primary' or 'Secondary'.
    """    
    if mass_type in ['total', 'Total', 'all', 'All']:
        mass_filename = 'MAX_MASS.txt'
        prim_start_line = 'Primary maximum mass (M_solar):'
        sec_start_line = 'Secondary maximum mass (M_solar):'
        He_grid = False
    elif mass_type in ['He', 'helium', 'Helium', 'helium mass']:
        mass_filename = 'He_MAX_MASS.txt'
        prim_start_line = 'Primary maximum He mass (M_solar):'
        sec_start_line = 'Primary maximum He mass (M_solar):'
        He_grid = True
    else:
        print('Invalid mass type. Please try either \'Total\' or \'Helium\'.')
        return

        
    data = []
    print(f"Scanning simulation folders in: {Path(path).resolve()}")

    # Scan nested directories, parsing mass ratio and log period from folder names.
    for parent_folder_name in os.listdir(path):
        parent_folder_path = Path(path) / parent_folder_name
        if not parent_folder_path.is_dir(): continue
        try:
            mass_ratio = float(parent_folder_name.split('_')[0])
        except (IndexError, ValueError):
            continue

        for sim_folder_name in os.listdir(parent_folder_path):
            sim_folder_path = parent_folder_path / sim_folder_name
            if not sim_folder_path.is_dir(): continue
            try:
                log_period = float(sim_folder_name.split('-')[-1])
            except (IndexError, ValueError):
                continue
            
            max_mass = np.nan # Default code for "Incomplete" or "Failed"
            
            fail_file = sim_folder_path / 'FAIL.txt'
            # merger_file = sim_folder_path / 'MERGER.txt'
            mass_file = sim_folder_path / mass_filename

            # Priority 1: Check for a failed simulation
            if fail_file.exists():
                max_mass = np.nan
            # Priority 2: Check for a contact phase result
            elif mass_file.exists():
                try:
                    with open(mass_file, 'r') as f:
                        for line in f:
                            # Now get the maximum mass for either the primary or the secondary
                            if star_type in ['secondary', 'Secondary']:
                                if line.strip().startswith(sec_start_line):
                                    parts = line.strip().split(':', 1)
                                    max_mass = float(parts[1].strip())
                            elif star_type in ['primary', 'Primary']:
                                if line.strip().startswith(prim_start_line):
                                    parts = line.strip().split(':', 1)
                                    max_mass = float(parts[1].strip())
                            else:
                                print('Invalid star type. Please choose either \'Primary\' or \'Secondary\'')
                                return
                except Exception as e:
                    print(f"An error occurred with file {mass_file}: {e}")
            
            # If none of these files exist, result_code remains -1
            
            data.append({
                'mass_ratio': mass_ratio,
                'log_period': log_period,
                'max_mass': max_mass
            })

    if not data:
        print("Error: No valid simulation folders found in the specified path.")
        return

    # --- Data Structuring and Plotting ---
    # Pivot the collected data into a 2D grid suitable for a heatmap.
    df = pd.DataFrame(data)
    grid_df = df.pivot_table(index='log_period', columns='mass_ratio', values='max_mass')
    
    # Sort axes for a clean, logical plot layout.
    grid_df = grid_df.sort_index(ascending=False).sort_index(axis=1, ascending=True)


    my_cmap = plt.colormaps['viridis'].copy()
    my_cmap.set_bad(color='black')  # Set the color for "bad" (NaN) values to black
    
    # Create the plot.
    fig, ax = plt.subplots(figsize=(12, 8))

    if He_grid:
        cbar_label = f'Maximum Helium mass accreted by the {star_type} ' + r'[M$_\odot$]'
        title_label = f'Maximum Helium mass accreted by the {star_type} in binary systems, for Primaries initially at' + r'20M$_\odot$'
        ax.set_title(title_label, fontsize=16, pad=20)
    
    else:
        cbar_label = f'Maximum mass of the {star_type} [M$_\\odot$]'
        ax.set_title(f'Maximum mass of the {star_type} in binary systems, for Primaries initially at 20M$_\\odot$', 
                 fontsize=16, pad=20)
    
    sns.heatmap(
        grid_df,
        ax=ax,
        cmap=my_cmap,
        linewidths=0.5,
        linecolor='grey',
        cbar_kws={'label': cbar_label}    # Key: Add a title to the colorbar.
    )

    ax.set_xlabel('Mass Ratio (q = M$_2$/M$_1$)', fontsize=12)
    ax.set_ylabel('Binary Period [log(days)]', fontsize=12)

    plt.tight_layout()
    plt.show()

    # --- Saving the Figure ---
    if save_figure:
        save_directory = Path(savepath)
        save_directory.mkdir(parents=True, exist_ok=True)
        figure_name = f'max_{star_type}_mass.pdf'
        if He_grid:
            figure_name = 'He_' + figure_name
        figure_path = save_directory / figure_name
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {figure_path}")

    print('Grid completed!')




def create_max_mass_merger_grid(path, savepath, save_figure=True, star_type='Secondary', mass_type='Total', flatten=True, log_scale=True, difference=False):
    """
    Creates a grid showing the maximum mass of the chosen star type.
    Missing data points (where a results file is not found) are represented as NaN
    and colored black on the plot to avoid skewing the main colormap.

    Additionally, simulation grid cells that ended in a merger (MERGER.txt exists)
    are marked with a red dot.

    Args:
        path (str): Root directory path containing simulation folders.
        savepath (str): Directory where the output figure should be saved.
        save_figure (bool): If True, saves the figure to a PDF. Defaults to True.
        star_type (str): Either 'Primary' or 'Secondary'.

    CURRENTLY HARD CODED FOR PRIMARY OF 20M!!!!!
    """    

    if mass_type in ['total', 'Total', 'all', 'All']:
        mass_filename = 'MAX_MASS.txt'
        prim_start_line = 'Primary maximum mass (M_solar):'
        sec_start_line = 'Secondary maximum mass (M_solar):'
        He_grid = False
    elif mass_type in ['He', 'helium', 'Helium', 'helium mass']:
        mass_filename = 'He_MAX_MASS.txt'
        prim_start_line = 'Primary maximum He mass (M_solar):'
        sec_start_line = 'Primary maximum He mass (M_solar):'
        He_grid = True
    else:
        print('Invalid mass type. Please try either \'Total\' or \'Helium\'.')
        return
    
    data = []
    print(f"Scanning simulation folders in: {Path(path).resolve()}")
        

    # Scan nested directories, parsing mass ratio and log period from folder names.
    for parent_folder_name in os.listdir(path):
        parent_folder_path = Path(path) / parent_folder_name
        
        if not parent_folder_path.is_dir(): 
            continue
        try:
            mass_ratio = float(parent_folder_name.split('_')[0])
        except (IndexError, ValueError):
            continue

        for sim_folder_name in os.listdir(parent_folder_path):
            sim_folder_path = parent_folder_path / sim_folder_name
            if not sim_folder_path.is_dir(): 
                continue
            try:
                log_period = float(sim_folder_name.split('-')[-1])
            except (IndexError, ValueError):
                continue
            
            max_mass = np.nan
            fail_file = sim_folder_path / 'FAIL.txt'
            merger_file = sim_folder_path / 'MERGER.txt'
            mass_file = sim_folder_path / mass_filename
            reverse_files = [
                sim_folder_path/'PASS_REVERSE.txt',
                sim_folder_path/'WEAK_PASS_REVERSE.txt',
                sim_folder_path/'PASS_REVERSE_NO_SUPERNOVA.txt'
            ]

            is_reverse = any(f.exists() for f in reverse_files)
            is_merger = merger_file.exists()

            if fail_file.exists():
                max_mass = np.nan
            elif mass_file.exists():
                try:
                    with open(mass_file, 'r') as f:
                        for line in f:
                            if star_type.lower() == 'secondary':
                                if line.strip().startswith(sec_start_line):
                                    parts = line.strip().split(':', 1)
                                    max_mass = float(parts[1].strip())
                            elif star_type.lower() == 'primary':
                                if line.strip().startswith(prim_start_line):
                                    parts = line.strip().split(':', 1)
                                    max_mass = float(parts[1].strip())
                            else:
                                print("Invalid star type. Please choose either 'Primary' or 'Secondary'")
                                return
                except Exception as e:
                    print(f"An error occurred with file {mass_file}: {e}")
            data.append({
                'mass_ratio': mass_ratio,
                'log_period': log_period,
                'max_mass': max_mass,
                'is_merger': is_merger,
                'is_reverse': is_reverse
            })

    if not data:
        print("Error: No valid simulation folders found in the specified path.")
        return

    df = pd.DataFrame(data)
    grid_df = df.pivot_table(index='log_period', columns='mass_ratio', values='max_mass')    
    merger_pts = df[df['is_merger'] & df['max_mass'].notna()]
    reverse_pts = df[df['is_reverse'] & df['max_mass'].notna()]
    grid_df = grid_df.sort_index(ascending=False).sort_index(axis=1, ascending=True)

    if flatten:
        grid_df[0.1] = grid_df[0.1] / 2
        grid_df[0.2] = grid_df[0.2] / 4
        grid_df[0.3] = grid_df[0.3] / 6
        grid_df[0.4] = grid_df[0.4] / 8
        grid_df[0.5] = grid_df[0.5] / 10
        grid_df[0.6] = grid_df[0.6] / 12
        grid_df[0.7] = grid_df[0.7] / 14
        grid_df[0.8] = grid_df[0.8] / 16
        grid_df[0.9] = grid_df[0.9] / 18

    elif difference:
        grid_df[0.1] = grid_df[0.1] - 2
        grid_df[0.2] = grid_df[0.2] - 4
        grid_df[0.3] = grid_df[0.3] - 6
        grid_df[0.4] = grid_df[0.4] - 8
        grid_df[0.5] = grid_df[0.5] - 10
        grid_df[0.6] = grid_df[0.6] - 12
        grid_df[0.7] = grid_df[0.7] - 14
        grid_df[0.8] = grid_df[0.8] - 16
        grid_df[0.9] = grid_df[0.9] - 18


    my_cmap = plt.colormaps['viridis'].copy()
    my_cmap.set_bad(color='black')

    # Set up the figure with extra width for legend/cbar
    fig, ax = plt.subplots(figsize=(13, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)

    if flatten:
        cbar_label = f'Ratio of maximum mass to initial mass of the {star_type} ' + r'[$M_{max}/M_i$]'
        ax.set_title(f'Maximum mass ratio of the {star_type} in binary systems, for Primaries initially at 20M$_\\odot$', 
                 fontsize=16, pad=20)
    elif difference:
        cbar_label = f'Maximum mass accreted by the {star_type} ' + r'[$M_{max} - M_i$]'
        ax.set_title(f'Maximum mass accreted by the {star_type} in binary systems, for Primaries initially at 20M$_\\odot$', 
                 fontsize=16, pad=20)

    elif He_grid:
        cbar_label = f'Maximum Helium mass accreted by the {star_type} ' + r'[M$_\odot$]'
        title_label = f'Maximum Helium mass accreted by the {star_type} in binary systems, for Primaries initially at' + r'20M$_\odot$'
        ax.set_title(title_label, fontsize=16, pad=20)
    
    else:
        cbar_label = f'Maximum mass of the {star_type} [M$_\\odot$]'
        ax.set_title(f'Maximum mass of the {star_type} in binary systems, for Primaries initially at 20M$_\\odot$', 
                 fontsize=16, pad=20)

    if log_scale:
        # Use LogNorm for log-scaled colorbar (ignoring NaNs)
        vmin = grid_df.min().min()
        vmax = grid_df.max().max()
        if vmin <= 0 or pd.isna(vmin):  # LogNorm requires strictly positive vmin
            vmin = 1e-3
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    
        # Draw heatmap on ax with colorbar at cax
        sns.heatmap(
            grid_df,
            ax=ax,
            cmap=my_cmap,
            linewidths=0.5,
            linecolor='grey',
            cbar_ax=cax,
            cbar_kws={'label': cbar_label, 'format': '%.0e'},
            norm=norm
        )
    else:
         # Draw heatmap on ax with colorbar at cax
        sns.heatmap(
            grid_df,
            ax=ax,
            cmap=my_cmap,
            linewidths=0.5,
            linecolor='grey',
            cbar_ax=cax,
            cbar_kws={'label': cbar_label}
        )

    # Get the colorbar and set tick/label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)        # tick font size
    cbar.ax.yaxis.label.set_size(12)         # label font size

    # Overlay white stars for "merger" runs
    merger_row_inds, merger_col_inds = [], []
    for _, row in merger_pts.iterrows():
        try:
            y = np.where(grid_df.index == row['log_period'])[0][0] + 0.5
            x = np.where(grid_df.columns == row['mass_ratio'])[0][0] + 0.5
            merger_row_inds.append(y)
            merger_col_inds.append(x)
        except:
            continue

    ax.scatter(
        merger_col_inds, merger_row_inds,
        s=80, color='white', marker='*',
        edgecolor='k', linewidths=1, zorder=10
    )

    # Overlay red dots for "reverse" runs
    reverse_row_inds, reverse_col_inds = [], []
    for _, row in reverse_pts.iterrows():
        try:
            y = np.where(grid_df.index == row['log_period'])[0][0] + 0.5
            x = np.where(grid_df.columns == row['mass_ratio'])[0][0] + 0.5
            reverse_row_inds.append(y)
            reverse_col_inds.append(x)
        except:
            continue

    ax.scatter(
        reverse_col_inds, reverse_row_inds,
        s=80, color='red', marker='o',
        edgecolor='k', linewidths=1, zorder=10
    )

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='k',
               markersize=10, linewidth=0, label='Reverse System'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white', markeredgecolor='k',
               markersize=10, linewidth=0, label='Merger'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markeredgecolor='grey',
               markersize=10, linewidth=0, label='Missing/Failed')
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.9),
        frameon=True
    )


    ax.set_xlabel('Mass Ratio (q = M$_2$/M$_1$)', fontsize=12)
    ax.set_ylabel('Binary Period [log(days)]', fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.88, 1])  # Leave space at right
    plt.show()

    if save_figure:
        save_directory = Path(savepath)
        save_directory.mkdir(parents=True, exist_ok=True)
        figure_name = f'max_{star_type}_mass.pdf'
        if He_grid:
            figure_name = 'He_' + figure_name
        if log_scale:
            figure_name = 'logscale_' + figure_name
        if flatten:
            figure_name = 'flattened_' + figure_name
        elif difference:
            figure_name = 'accreted_' + figure_name
        figure_path = save_directory / figure_name
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {figure_path}")

    print('Grid completed!')



def create_max_X_He_surface_merger_grid(path, savepath, save_figure=True, star_type='Secondary', log_scale=True):
    """
    Creates a grid showing the maximum surface Helium abundance of the chosen star type.
    Missing data points (where a results file is not found) are represented as NaN
    and colored black on the plot to avoid skewing the main colormap.

    Additionally, simulation grid cells that ended in a merger (MERGER.txt exists)
    are marked with a red dot.

    Args:
        path (str): Root directory path containing simulation folders.
        savepath (str): Directory where the output figure should be saved.
        save_figure (bool): If True, saves the figure to a PDF. Defaults to True.
        star_type (str): Either 'Primary' or 'Secondary'.

    CURRENTLY HARD CODED FOR PRIMARY OF 20M!!!!!
    """    

    data = []
    print(f"Scanning simulation folders in: {Path(path).resolve()}")

    # Scan nested directories, parsing mass ratio and log period from folder names.
    for parent_folder_name in os.listdir(path):
        parent_folder_path = Path(path) / parent_folder_name
        if not parent_folder_path.is_dir(): 
            continue
        try:
            mass_ratio = float(parent_folder_name.split('_')[0])
        except (IndexError, ValueError):
            continue

        for sim_folder_name in os.listdir(parent_folder_path):
            sim_folder_path = parent_folder_path / sim_folder_name
            if not sim_folder_path.is_dir(): 
                continue
            try:
                log_period = float(sim_folder_name.split('-')[-1])
            except (IndexError, ValueError):
                continue
            
            max_X_He = np.nan
            X_He_file = sim_folder_path / 'MAX_X_He_SURFACE.txt'
            
            fail_file = sim_folder_path / 'FAIL.txt'
            merger_file = sim_folder_path / 'MERGER.txt'            
            reverse_files = [
                sim_folder_path/'PASS_REVERSE.txt',
                sim_folder_path/'WEAK_PASS_REVERSE.txt',
                sim_folder_path/'PASS_REVERSE_NO_SUPERNOVA.txt'
            ]

            is_reverse = any(f.exists() for f in reverse_files)
            is_merger = merger_file.exists()

            if fail_file.exists():
                max_X_He = np.nan
            elif X_He_file.exists():
                try:
                    with open(X_He_file, 'r') as f:
                        for line in f:
                            if star_type.lower() == 'secondary':
                                if line.strip().startswith('Secondary maximum He surface abundance (%):'):
                                    parts = line.strip().split(':', 1)
                                    max_X_He = float(parts[1].strip())
                            elif star_type.lower() == 'primary':
                                if line.strip().startswith('Primary maximum He surface abundance (%):'):
                                    parts = line.strip().split(':', 1)
                                    max_X_He = float(parts[1].strip())
                            else:
                                print("Invalid star type. Please choose either 'Primary' or 'Secondary'")
                                return
                except Exception as e:
                    print(f"An error occurred with file {X_He_file}: {e}")
            data.append({
                'mass_ratio': mass_ratio,
                'log_period': log_period,
                'max_X_He': max_X_He,
                'is_merger': is_merger,
                'is_reverse': is_reverse
            })

    if not data:
        print("Error: No valid simulation folders found in the specified path.")
        return

    df = pd.DataFrame(data)
    grid_df = df.pivot_table(index='log_period', columns='mass_ratio', values='max_X_He')    
    merger_pts = df[df['is_merger'] & df['max_X_He'].notna()]
    reverse_pts = df[df['is_reverse'] & df['max_X_He'].notna()]
    grid_df = grid_df.sort_index(ascending=False).sort_index(axis=1, ascending=True)


    my_cmap = plt.colormaps['viridis'].copy()
    my_cmap.set_bad(color='black')

    # Set up the figure with extra width for legend/cbar
    fig, ax = plt.subplots(figsize=(13, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)

    ax.set_title(f'Maximum Surface Abundance of Helium for the {star_type} in Binary Systems, for Primaries Initially at 20M$_\\odot$', 
             fontsize=16, pad=20)

    if log_scale:
        grid_df = 1 - grid_df
        
        # Use LogNorm for log-scaled colorbar (ignoring NaNs)
        vmin = 1e-2#grid_df.min().min()
        vmax = 1#grid_df.max().max()
        if vmin <= 0 or pd.isna(vmin):  # LogNorm requires strictly positive vmin
            vmin = 1e-3
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        cbar_label = f'Maximum He surface abundance for the {star_type} ' + r'[$1-$max($X_{He,s}$)]'
        
        # Draw heatmap on ax with colorbar at cax
        sns.heatmap(
            grid_df,
            ax=ax,
            cmap=my_cmap,
            linewidths=0.5,
            linecolor='grey',
            cbar_ax=cax,
            cbar_kws={'label': cbar_label},#, 'format': '%.0e'},
            norm=norm
        )
    else:
         # Draw heatmap on ax with colorbar at cax
        sns.heatmap(
            grid_df,
            ax=ax,
            cmap=my_cmap,
            linewidths=0.5,
            linecolor='grey',
            cbar_ax=cax,
            cbar_kws={'label': cbar_label}
        )

    # Get the colorbar and set tick/label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)        # tick font size
    cbar.ax.yaxis.label.set_size(12)         # label font size

    # Overlay white stars for "merger" runs
    merger_row_inds, merger_col_inds = [], []
    for _, row in merger_pts.iterrows():
        try:
            y = np.where(grid_df.index == row['log_period'])[0][0] + 0.5
            x = np.where(grid_df.columns == row['mass_ratio'])[0][0] + 0.5
            merger_row_inds.append(y)
            merger_col_inds.append(x)
        except:
            continue

    ax.scatter(
        merger_col_inds, merger_row_inds,
        s=80, color='white', marker='*',
        edgecolor='k', linewidths=1, zorder=10
    )

    # Overlay red dots for "reverse" runs
    reverse_row_inds, reverse_col_inds = [], []
    for _, row in reverse_pts.iterrows():
        try:
            y = np.where(grid_df.index == row['log_period'])[0][0] + 0.5
            x = np.where(grid_df.columns == row['mass_ratio'])[0][0] + 0.5
            reverse_row_inds.append(y)
            reverse_col_inds.append(x)
        except:
            continue

    ax.scatter(
        reverse_col_inds, reverse_row_inds,
        s=80, color='red', marker='o',
        edgecolor='k', linewidths=1, zorder=10
    )

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='k',
               markersize=10, linewidth=0, label='Reverse System'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white', markeredgecolor='k',
               markersize=10, linewidth=0, label='Merger'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markeredgecolor='grey',
               markersize=10, linewidth=0, label='Missing/Failed')
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.9),
        frameon=True
    )


    ax.set_xlabel('Mass Ratio (q = M$_2$/M$_1$)', fontsize=12)
    ax.set_ylabel('Binary Period [log(days)]', fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.88, 1])  # Leave space at right
    plt.show()

    if save_figure:
        save_directory = Path(savepath)
        save_directory.mkdir(parents=True, exist_ok=True)
        figure_name = f'max_{star_type}_X_He.pdf'
        if log_scale:
            figure_name = 'logscale_' + figure_name
        figure_path = save_directory / figure_name
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {figure_path}")

    print('Grid completed!')



def thermohaline_lifetime_comparison_grid(path, savepath, main_sequence_lifetime=True, save_figure=True):
    """
    Creates a heatmap of stellar lifetime ratios from a directory of simulation results.

    Missing data points (where a results file is not found) are represented as NaN
    and colored black on the plot to avoid skewing the main colormap.

    Args:
        path (str): Root directory path containing simulation folders.
        savepath (str): Directory where the output figure should be saved.
        save_figure (bool): If True, saves the figure to a PDF. Defaults to False.
    """
    if main_sequence_lifetime:
        txt_to_find = '- MS lifetime ratio:'
        title = 'Main sequence lifetime comparison for thermohaline mixing (20M$_{\odot}$ Primary)'
    else:
        txt_to_find = '- Lifetime ratio:'
        title = 'Total lifetime comparison for thermohaline mixing (20M$_{\odot}$ Primary)'
    data = []
    print(f"Scanning simulation folders in: {Path(path).resolve()}")

    # Scan nested directories, parsing mass ratio and log period from folder names.
    for parent_folder_name in os.listdir(path):
        parent_folder_path = Path(path) / parent_folder_name
        if not parent_folder_path.is_dir(): continue
        try:
            mass_ratio = float(parent_folder_name.split('_')[0])
        except (IndexError, ValueError):
            continue

        for sim_folder_name in os.listdir(parent_folder_path):
            sim_folder_path = parent_folder_path / sim_folder_name
            if not sim_folder_path.is_dir(): continue
            try:
                log_period = float(sim_folder_name.split('-')[-1])
            except (IndexError, ValueError):
                continue
            
            lifetime_comparison_file = sim_folder_path / 'thermohaline_lifetime_comparison.txt'
            merger_file = sim_folder_path / 'MERGER.txt'
            reverse_files = [
                sim_folder_path/'PASS_REVERSE.txt',
                sim_folder_path/'WEAK_PASS_REVERSE.txt',
                sim_folder_path/'PASS_REVERSE_NO_SUPERNOVA.txt'
            ]

            is_reverse = any(f.exists() for f in reverse_files)
            is_merger = merger_file.exists()

            # Default to NaN for missing data
            lifetime_ratio = np.nan
        
            try:
                with open(lifetime_comparison_file, 'r') as f:
                    for line in f:
                        if line.strip().startswith(txt_to_find):
                            parts = line.strip().split(':', 1)
                            lifetime_ratio = float(parts[1].strip())
            except FileNotFoundError:
                # If the file doesn't exist, lifetime_ratio correctly remains NaN.
                pass
            except Exception as e:
                print(f"An error occurred with file {lifetime_comparison_file}: {e}")

            data.append({
                'mass_ratio': mass_ratio,
                'log_period': log_period,
                'lifetime_ratio': lifetime_ratio,
                'is_merger': is_merger,
                'is_reverse': is_reverse
            })

    if not data:
        print("Error: No valid simulation folders found.")
        return

    # --- Data Structuring and Plotting ---
    # Pivot the collected data into a 2D grid suitable for a heatmap.
    df = pd.DataFrame(data)
    grid_df = df.pivot_table(index='log_period', columns='mass_ratio', values='lifetime_ratio')
    
    # Sort axes for a clean, logical plot layout.
    grid_df = grid_df.sort_index(ascending=False).sort_index(axis=1, ascending=True)

    grid_df = 1-grid_df

    my_cmap = plt.colormaps['viridis'].copy()
    my_cmap.set_bad(color='black')  # Set the color for "bad" (NaN) values to black

    # Use LogNorm for log-scaled colorbar (ignoring NaNs)
    vmin = grid_df.min().min()
    vmax = grid_df.max().max()
    if vmin <= 0 or pd.isna(vmin):  # LogNorm requires strictly positive vmin
        vmin = 1e-3
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    
    # Create the plot.
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        grid_df,
        ax=ax,
        cmap=my_cmap,
        linewidths=0.5,
        linecolor='grey',
        cbar_kws={'label': 'Lifetime ratio [1 - $T_{th}$ / $T$]'},    # Key: Add a title to the colorbar.
        norm=norm   # <-- NEW: This enables log scaling
    )

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Mass Ratio (q = M$_2$/M$_1$)', fontsize=12)
    ax.set_ylabel('Binary Period [log(days)]', fontsize=12)

    plt.tight_layout()
    plt.show()

    # --- Saving the Figure ---
    if save_figure:
        save_directory = Path(savepath)
        save_directory.mkdir(parents=True, exist_ok=True)
        figure_path = save_directory / 'thermohaline_lifetime_grid.pdf'
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {figure_path}")

    print('Grid completed!')




# def plot_he_compositions(star2, star2_th, path, ext_path, final_ext_path, 
#                              file1, file2, mass_ratio, period, savepath, text_size=16, 
#                              save_figure=True, print_info=False, x_axis_unit='percent'):
#     """
#     Generates Helium composition plots for Standard vs Thermohaline mixing.
    
#     Parameters:
#     - x_axis_unit: 'percent' (default) for m/M_*, or 'mass' for m in M_sol.
#     """
    
#     # 1. Calculate Timesteps
#     star2_timestep = find_helium_flash_timestep(star2, plot_figure=False)
#     star2_th_timestep = find_helium_flash_timestep(star2_th, plot_figure=False)

#     if print_info:
#         print('Final MS timestep star2:', star2_timestep, 'Model:', (star2_timestep // 1000)*1000)
#         print('Final MS timestep star2 thermohaline:', star2_th_timestep, 'Model:', (star2_th_timestep // 1000)*1000)

#     # 2. Get out models (Read Data)
#     if mass_ratio == 0.5:
#         star2_first_sim_str, _, model_nums1_str = get_out_full_mesh_models(f'{path}{file2}', lines_after=499)
#     else:
#         star2_first_sim_str, _, model_nums1_str = get_out_full_mesh_models(f'{path}{file2}', lines_after=199)        
    
#     star2_first_sim, model_nums1 = convert_strings_to_plot_arrays(star2_first_sim_str, ['He4', 'm'], model_num_strs=model_nums1_str)

#     star2_second_sim_str, _, model_nums2_str = get_out_full_mesh_models(f'{ext_path}{file1}', lines_after=199)
#     star2_second_sim, model_nums2 = convert_strings_to_plot_arrays(star2_second_sim_str, ['He4', 'm'], model_num_strs=model_nums2_str)
    
#     star2_second_th_sim_str, _, model_nums2_th_str = get_out_full_mesh_models(f'{ext_path}{file2}', lines_after=199)
#     star2_second_th_sim, model_nums2_th = convert_strings_to_plot_arrays(star2_second_th_sim_str, ['He4', 'm'], model_num_strs=model_nums2_th_str)

#     star2_third_sim_str, _, model_nums3_str = get_out_full_mesh_models(f'{final_ext_path}{file1}', lines_after=499)
#     star2_third_sim, model_nums3 = convert_strings_to_plot_arrays(star2_third_sim_str, ['He4', 'm'], model_num_strs=model_nums3_str)

#     # Apply corrections
#     # REMOVED +100 to ensure ZAMS/Continuity is zero-gap based
#     correction = model_nums1[-1] 
#     model_nums2 = model_nums2 + correction
#     model_nums2_th = model_nums2_th + correction
#     model_nums3 = model_nums3 + correction

#     # 3. Organize Data Chunks
#     # Standard
#     all_models_std = np.concatenate([model_nums1, model_nums2, model_nums3])
#     data_chunks_std = [star2_first_sim, star2_second_sim, star2_third_sim]
    
#     # Thermohaline
#     all_models_th = np.concatenate([model_nums1, model_nums2_th])
#     data_chunks_th = [star2_first_sim, star2_second_th_sim]

#     # Helper: Get Data from Ragged Chunks
#     def get_model_data(target_val, all_models_arr, data_chunks):
#         global_idx = (np.abs(all_models_arr - target_val)).argmin()
#         actual_model_num = all_models_arr[global_idx]
        
#         current_count = 0
#         selected_data = None
#         for chunk in data_chunks:
#             chunk_len = len(chunk)
#             if global_idx < (current_count + chunk_len):
#                 local_idx = global_idx - current_count
#                 selected_data = chunk[local_idx]
#                 break
#             current_count += chunk_len
#         return selected_data, actual_model_num

#     # Helper: Generate Target Model Numbers (No Rounding)
#     def generate_targets(start_val, end_val, count):
#         return np.linspace(start_val, end_val, count)

#     # --- SELECTION LOGIC ---

#     # Phase 1: First Simulation
#     # Changed start_val from model_nums1[0] to 0 to assume ZAMS is zero
#     p1_raw = generate_targets(0, model_nums1[-1], 4)
#     phase1_targets = p1_raw[:3] # Keep indices 0, 1, 2
    
#     # Explicitly check for ZAMS label on the very first target
#     # If the file starts at 200, but we request 0, the nearest is 200.
#     # We force the label check later against phase1_targets[0]
    
#     # Phase 2: Extension
#     def get_phase2_targets(phase2_start_avail, phase2_end_tams):
#         return generate_targets(phase2_start_avail, phase2_end_tams, 5)

#     phase2_targets_std = get_phase2_targets(model_nums2[0], star2_timestep)
#     phase2_targets_th = get_phase2_targets(model_nums2_th[0], star2_th_timestep)

#     # 4. Build Plot Lists
#     def build_plot_list(targets, all_models, chunks, is_tams_val, phase_name):
#         plot_list = []
#         for t in targets:
#             data, actual_num = get_model_data(t, all_models, chunks)
            
#             # Label Logic
#             # If the target is 0 (or close to the start of our array), it's ZAMS
#             if t == phase1_targets[0]: 
#                 label = "ZAMS"
#             elif np.isclose(t, is_tams_val, atol=20): 
#                 label = "TAMS"
#             else:
#                 label = f"Mod. {int(actual_num)}"
            
#             plot_list.append({'data': data, 'label': label, 'phase': phase_name})
#         return plot_list

#     list_p1_std = build_plot_list(phase1_targets, all_models_std, data_chunks_std, star2_timestep, 'early')
#     list_p2_std = build_plot_list(phase2_targets_std, all_models_std, data_chunks_std, star2_timestep, 'late')
    
#     list_p1_th = build_plot_list(phase1_targets, all_models_th, data_chunks_th, star2_th_timestep, 'early')
#     list_p2_th = build_plot_list(phase2_targets_th, all_models_th, data_chunks_th, star2_th_timestep, 'late')

#     full_data_std = list_p1_std + list_p2_std
#     full_data_th = list_p1_th + list_p2_th

#     # 5. Plotting
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

#     def plot_track_styled(ax, data_dict_list, fs):
#         early_items = [x for x in data_dict_list if x['phase'] == 'early']
#         late_items = [x for x in data_dict_list if x['phase'] == 'late']
        
#         # --- Helper for X Axis ---
#         def get_x_data(model_data):
#             mass_coords = model_data[1]
#             if x_axis_unit == 'mass':
#                 return mass_coords
#             else: # percent
#                 if len(mass_coords) > 0 and mass_coords[-1] != 0:
#                     return mass_coords / mass_coords[-1]
#                 else:
#                     return mass_coords

#         # Phase 1: Winter Colormap
#         for i, item in enumerate(early_items):
#             model_data = item['data']
#             x_plot = get_x_data(model_data)

#             color_idx = 0.3 + (0.7 * (i / len(early_items)))
#             c = cm_mpl.winter(color_idx)
            
#             ax.plot(x_plot, model_data[0], label=item['label'], 
#                     color=c, linestyle='-', linewidth=2, alpha=0.8)

#         # Phase 2: Autumn Colormap
#         for i, item in enumerate(late_items):
#             model_data = item['data']
#             x_plot = get_x_data(model_data)

#             color_idx = 0.2 + (0.8 * (i / len(late_items)))
#             c = cm_mpl.autumn(color_idx)
            
#             ax.plot(x_plot, model_data[0], label=item['label'], 
#                     color=c, linestyle='-', linewidth=2, alpha=0.9)
        
#         # Text Sizing & Legend Placement
        
#         if x_axis_unit == 'mass':
#             ax.set_xlabel(r'Mass co-ordinate $m$ [M$_\odot$]', fontsize=fs)
#         else:
#             ax.set_xlabel(r'Mass co-ordinate [$m$/M$_*$]', fontsize=fs)
            
#         ax.tick_params(axis='both', which='major', labelsize=fs)
#         ax.grid(True, alpha=0.3)
        
#         # LEGEND MODIFICATION
#         ax.legend(fontsize=fs-5, 
#                   loc='lower center',       
#                   bbox_to_anchor=(0.5, 1.01), 
#                   ncol=3,                   
#                   borderaxespad=0)

#     # Plot No Thermohaline
#     plot_track_styled(ax1, full_data_std, text_size)
#     ax1.set_ylabel(r'He Concentration', fontsize=text_size)

#     # Plot Thermohaline
#     plot_track_styled(ax2, full_data_th, text_size)
    
#     plt.tight_layout()
    
#     if save_figure:
#         if not os.path.exists(savepath):
#             os.makedirs(savepath)
            
#         fname_suffix = "mass_coord" if x_axis_unit == 'mass' else "mass_percent"
#         plt.savefig(f'{savepath}/A20-{mass_ratio:.1f}-{period:.1f}_He_composition_graph_{fname_suffix}.pdf', dpi=150)
        
#     plt.show()



def plot_he_compositions(star2, star2_th, path, ext_path, final_ext_path, 
                         file1, file2, base_cutoff, mass_ratio, period, savepath, 
                         star1=None, # New optional input for Star 1 object
                         comparison_mode='star2_vs_th', # Options: 'star2_vs_th', 'star1_vs_star2', 'star1_vs_th'
                         text_size=16, save_figure=True, print_info=False, x_axis_unit='percent'):
    """
    Generates Helium composition plots for two compared models.
    
    Parameters:
    - star1: The star1 object (required only if comparison_mode involves star1).
    - comparison_mode: 
        'star2_vs_th' (default): Standard Star 2 vs Thermohaline Star 2.
        'star1_vs_star2': Star 1 vs Star 2.
        'star1_vs_th': Star 1 vs Thermohaline Star 2.
    - x_axis_unit: 'percent' (default) for m/M_*, or 'mass' for m in M_sol.
    """
    
    # --- HELPER: DATA LOADER ---
    def load_model_track(file_sequence, cutoff_lines):
        """
        Loads a sequence of model files, applies correction, and concatenates.
        file_sequence: list of full file paths [base, ext, final]
        cutoff_lines: list of 'lines_after' integers [base_cut, ext_cut, final_cut]
        """
        chunks = []
        model_nums_list = []
        correction = 0
        first_valid_found = False # Flag to track if we have found the base file yet
        
        for i, (fpath, cut) in enumerate(zip(file_sequence, cutoff_lines)):
            # Read data
            try:
                sim_str, _, nums_str = get_out_full_mesh_models(fpath, lines_after=cut)
                sim_data, nums = convert_strings_to_plot_arrays(sim_str, ['He4', 'm'], model_num_strs=nums_str)
            except Exception as e:
                print(f"Warning: Could not load {fpath}. Error: {e}")
                continue

            # CHECK: If data is empty, skip this file to avoid the IndexError
            if len(nums) == 0:
                print(f"Warning: File contained no data after cutoff: {fpath}")
                continue

            # Apply correction logic
            if not first_valid_found:
                # This is the first file with actual data
                correction = nums[-1]
                first_valid_found = True
            else:
                # Subsequent chunks get shifted
                nums = nums + correction
                
            chunks.append(sim_data)
            model_nums_list.append(nums)
        
        # Check if we found ANY data at all
        if not model_nums_list:
            raise ValueError("No data could be loaded from any files in the sequence.")

        all_models = np.concatenate(model_nums_list)
        return all_models, chunks

    # --- 1. SETUP CONFIGURATIONS ---
    
    # Define file paths and cutoffs based on your original logic
    # Note: Adjust these lists if Star 1 has a different file structure in your directory

    
    # Configuration Dictionaries
    # STAR 2 STANDARD
    config_star2 = {
        'obj': star2,
        'label_prefix': 'Star 2 (Std)',
        'files': [f'{path}{file2}', f'{ext_path}{file1}', f'{final_ext_path}{file1}'],
        'cutoffs': [base_cutoff, 199, 499] 
    }

    # STAR 2 THERMOHALINE
    config_star2_th = {
        'obj': star2_th,
        'label_prefix': 'Star 2 (TH)',
        'files': [f'{path}{file2}', f'{ext_path}{file2}'],
        'cutoffs': [base_cutoff, 199, 499]
    }

    # STAR 1 (Assumed standard paths using file1 as base)
    config_star1 = {
        'obj': star1,
        'label_prefix': 'Star 1',
        # Assuming Star 1 uses file1, and extends with file1. Adjust if Star 1 has different extension files.
        'files': [f'{path}{file1}', f'{ext_path}{file1}', f'{final_ext_path}{file1}'], 
        'cutoffs': [base_cutoff, 199, 499]
    }

    # Select Comparison Mode
    if comparison_mode == 'star2_vs_th':
        left_config = config_star2
        right_config = config_star2_th
        fname_tag = "He_composition"
    elif comparison_mode == 'star1_vs_star2':
        if star1 is None: raise ValueError("star1 object must be provided for this mode")
        left_config = config_star1
        right_config = config_star2
        fname_tag = "S1_vs_S2_He"
    elif comparison_mode == 'star1_vs_th':
        if star1 is None: raise ValueError("star1 object must be provided for this mode")
        left_config = config_star1
        right_config = config_star2_th
        fname_tag = "S1_vs_TH_He"
    else:
        raise ValueError(f"Unknown comparison_mode: {comparison_mode}")

    # --- 2. CALCULATE TIMESTEPS (TAMS) ---
    tams_left = find_helium_flash_timestep(left_config['obj'], plot_figure=False)
    tams_right = find_helium_flash_timestep(right_config['obj'], plot_figure=False)

    if print_info:
        print(f"Left Model ({left_config['label_prefix']}) TAMS: {tams_left}")
        print(f"Right Model ({right_config['label_prefix']}) TAMS: {tams_right}")

    # --- 3. LOAD DATA ---
    all_models_left, chunks_left = load_model_track(left_config['files'], left_config['cutoffs'])
    all_models_right, chunks_right = load_model_track(right_config['files'], right_config['cutoffs'])

    # --- 4. ORGANIZE PLOT DATA ---
    
    # Helper: Get Data from Ragged Chunks (Reused)
    def get_model_data(target_val, all_models_arr, data_chunks):
        global_idx = (np.abs(all_models_arr - target_val)).argmin()
        actual_model_num = all_models_arr[global_idx]
        current_count = 0
        selected_data = None
        for chunk in data_chunks:
            chunk_len = len(chunk)
            if global_idx < (current_count + chunk_len):
                local_idx = global_idx - current_count
                selected_data = chunk[local_idx]
                break
            current_count += chunk_len
        return selected_data, actual_model_num

        # Helper: Build Plot Lists
    def build_plot_sequence(all_models, chunks, tams_val):
        # --- FIX STARTS HERE ---
        
        # 1. Identify the boundary between the first file (Phase 1) and extensions (Phase 2)
        # chunks[0] contains the data for the first file.
        # We need the *count* of models in the first file to find the correct Model Number in all_models.
        count_chunk0 = len(chunks[0])
        
        if count_chunk0 > 0:
            # The model number of the last entry in the first file
            end_val_p1 = all_models[count_chunk0 - 1]
        else:
            end_val_p1 = all_models[0]

        # Phase 1 Targets: From Model 0 to end of first file
        # We assume ZAMS is roughly 0 or the first available model
        p1_targets = np.linspace(0, end_val_p1, 4)[:3]
        
        # 2. Determine Start of Phase 2
        if len(chunks) > 1 and len(chunks[1]) > 0:
            # The start of the second file is the index immediately following chunk 0
            start_val_p2 = all_models[count_chunk0]
        else:
            # Fallback if there are no extension files, just continue from p1
            start_val_p2 = end_val_p1

        # Phase 2 Targets: From start of second file to TAMS
        p2_targets = np.linspace(start_val_p2, tams_val, 5)

        # --- FIX ENDS HERE ---

        # Extraction Logic
        def extract(targets, phase_name):
            res_list = []
            for t in targets:
                data, actual_num = get_model_data(t, all_models, chunks)
                
                if data is None: continue # Safety skip if data is missing
                
                if t == p1_targets[0]: label = "ZAMS"
                elif np.isclose(t, tams_val, atol=20): label = "TAMS"
                else: label = f"Mod. {int(actual_num)}"
                
                res_list.append({'data': data, 'label': label, 'phase': phase_name})
            return res_list

        return extract(p1_targets, 'early') + extract(p2_targets, 'late')

    data_left = build_plot_sequence(all_models_left, chunks_left, tams_left)
    data_right = build_plot_sequence(all_models_right, chunks_right, tams_right)

    # --- 5. PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    def plot_track_styled(ax, data_dict_list, fs, title):
        early_items = [x for x in data_dict_list if x['phase'] == 'early']
        late_items = [x for x in data_dict_list if x['phase'] == 'late']
        
        def get_x_data(model_data):
            mass_coords = model_data[1]
            if x_axis_unit == 'mass': return mass_coords
            else: return (mass_coords / mass_coords[-1]) if (len(mass_coords)>0 and mass_coords[-1]!=0) else mass_coords

        # Winter Colormap (Early)
        for i, item in enumerate(early_items):
            x_plot = get_x_data(item['data'])
            c = cm_mpl.winter(0.3 + (0.7 * (i / len(early_items))))
            ax.plot(x_plot, item['data'][0], label=item['label'], color=c, ls='-', lw=2, alpha=0.8)

        # Autumn Colormap (Late)
        for i, item in enumerate(late_items):
            x_plot = get_x_data(item['data'])
            c = cm_mpl.autumn(0.2 + (0.8 * (i / len(late_items))))
            ax.plot(x_plot, item['data'][0], label=item['label'], color=c, ls='-', lw=2, alpha=0.9)
        
        # Styling
        if x_axis_unit == 'mass': ax.set_xlabel(r'Mass co-ordinate $m$ [M$_\odot$]', fontsize=fs)
        else: ax.set_xlabel(r'Mass co-ordinate [$m$/M$_*$]', fontsize=fs)
            
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=fs+2, pad=20)
        ax.legend(fontsize=fs-5, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, borderaxespad=0)

    # Execute Plot
    plot_track_styled(ax1, data_left, text_size, left_config['label_prefix'])
    ax1.set_ylabel(r'He Concentration', fontsize=text_size)
    
    plot_track_styled(ax2, data_right, text_size, right_config['label_prefix'])
    
    plt.tight_layout()
    
    if save_figure:
        if not os.path.exists(savepath): os.makedirs(savepath)
        suffix = "mass_coord" if x_axis_unit == 'mass' else "mass_percent"
        plt.savefig(f'{savepath}/A20-{mass_ratio:.1f}-{period:.1f}_{fname_tag}_{suffix}.pdf', dpi=150)
        
    plt.show()


    
#===============================================================================================================================
# pdf Functions
#===============================================================================================================================
def create_heading_page(folder_name, status_text):
    """
    Creates a PDF page with the folder name and status message
    centered vertically and horizontally.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    # Set fonts
    title_font = "Helvetica-Bold"
    status_font = "Helvetica"
    title_size = 28
    status_size = 18

    # Calculate positions
    page_width, page_height = A4
    spacing = 2 * cm

    # Measure text heights
    total_height = title_size + spacing + status_size

    # Y-coordinate for the center block
    center_y = (page_height + total_height) / 2

    # Draw folder title
    c.setFont(title_font, title_size)
    c.drawCentredString(page_width / 2, center_y, folder_name)

    # Draw status text below
    c.setFont(status_font, status_size)
    c.drawCentredString(page_width / 2, center_y - title_size - spacing, status_text)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer



def combine_folder_pdfs(root_dir, output_pdf_path):
    """
    Combines all PDFs from subdirectories of root_dir into a single PDF file,
    with headings and pass/fail status in between.
    """
    merger = PdfMerger()

    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue  # Skip files

        # Check for pass/fail file
        status = "Unknown"
        for fname in os.listdir(folder_path):
            if fname.upper() == "PASS.TXT":
                status = "This simulation PASSED"
                break
            if fname.upper() == "WEAK_PASS.TXT":
                status = "This simulation PASSED WEAK \nDid not reach carbon burning"
                break
            elif fname.upper() == "MERGER.TXT":
                status = "This simulation was a MERGER"
                break
            elif fname.upper() == "PASS_NO_SUPERNOVA.TXT":
                status = "This simulation PASSED - NO SUPERNOVA"
                break
            elif fname.upper() == "PASS_REVERSE.TXT":
                status = "This simulation REVERSED"
                break
            if fname.upper() == "WEAK_PASS_REVERSE.TXT":
                status = "This simulation REVERSED WEAK \nDid not reach carbon burning"
                break
            elif fname.upper() == "PASS_REVERSE_NO_SUPERNOVA.TXT":
                status = "This simulation REVERSED - NO SUPERNOVA"
                break
            elif fname.upper() == "FAIL.TXT":
                status = "This simulation FAILED"
                break

        # Add heading page
        heading_pdf = create_heading_page(folder, status)
        merger.append(heading_pdf)

        # Add PDFs in folder
        for file in sorted(os.listdir(folder_path)):
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(folder_path, file)
                merger.append(pdf_path)

    # Write final output
    merger.write(os.path.join(root_dir, output_pdf_path))
    merger.close()
    print(f"Combined PDF saved as: {output_pdf_path}")



def determine_binary_result(out_star1, out_star2):
    """
    Determines the final result of the binary simulation

   Codes:
        1: PASS
        2: WEAK_PASS
        3: PASS_NO_SUPERNOVA
        4: MERGER
        11: PASS_REVERSE (PASS + 10)
        12: WEAK_PASS_REVERSE (WEAK_PASS + 10)
        13: PASS_REVERSE_NO_SUPERNOVA (PASS_NO_SUPERNOVA + 10)
        0: FAIL
    """
    res1, H1, He1, temp1, psi1 = out_star1
    res2, H2, He2, temp2, psi2 = out_star2
    
    # Default to a FAIL state using the primary's data.
    final_res = 0
    data_to_write = (H1, He1, temp1, psi1)
    
    # The logic checks for the highest priority outcomes first (standard pass),
    # then moves to weaker passes. For each pass type, it prioritizes the primary star.

    # Priority 1: Standard Pass (Code 1)
    if res1 == 1:
        final_res = 1  # PASS
        data_to_write = (H1, He1, temp1, psi1)
    elif res2 == 1:
        final_res = 11  # PASS_REVERSE
        data_to_write = (H2, He2, temp2, psi2)
    
    # Priority 2: Merger (Code 4) - only checked if no standard pass occurred
    elif res1 == 4 or res2 == 4:
        final_res = 4  # MERGER
        data_to_write = (H1, He1, temp1, psi1)
    
    # Priority 3: Weak Pass (Code 2) - only checked if no standard or merger pass occurred
    elif res1 == 2:
        final_res = 2  # WEAK_PASS
        data_to_write = (H1, He1, temp1, psi1)
    elif res2 == 2:
        final_res = 12  # WEAK_PASS_REVERSE
        data_to_write = (H2, He2, temp2, psi2)

    # Priority 4: No Supernova Pass (Code 3) - only checked if no other pass occurred
    elif res1 == 3:
        final_res = 3  # PASS_NO_SUPERNOVA
        data_to_write = (H1, He1, temp1, psi1)
    elif res2 == 3:
        final_res = 13  # PASS_REVERSE_NO_SUPERNOVA
        data_to_write = (H2, He2, temp2, psi2)
        
    # If none of the above conditions are met, the default 'FAIL' (0) is returned.
    return final_res, data_to_write

#===============================================================================================================================
# Run Functions
#===============================================================================================================================


def make_all_plots(path, star1, star2, savepath, th_comparison=False, star_th=pd.DataFrame(), reverse=False, save_figures=True):
    """
    Creates plots of data from the STARS plot files
    """
    res1, H1, He1, temp1, psi1 = get_final_out(star1, star2, path, savepath, 'primary')
    res2, H2, He2, temp2, psi2 = get_final_out(star1, star2, path, savepath, 'secondary')
    
    out_star1 = (res1, H1, He1, temp1, psi1)
    out_star2 = (res2, H2, He2, temp2, psi2)
    
    final_res, data_to_write = determine_binary_result(out_star1, out_star2)
    write_pass_fail_file(final_res, *data_to_write, savepath)
    
    plot_HR_diagrams(star1, star2, savepath, add_thermohaline=th_comparison, th_star=star_th, save_figure=save_figures)
    plot_change_in_mass(star1, star2, savepath, add_thermohaline=th_comparison, th_star=star_th, save_figure=save_figures)
    if len(star1[0]) == len(star2[0]):
        plot_radii(star1, star2, savepath, save_figure=save_figures)
        

    if th_comparison:
        if reverse:
            kippenhahn_reverse(star2, star1, star_th, savepath, save_figure=save_figures)
        else:
            kippenhahn(star1, star2, star_th, savepath, save_figure=save_figures)
    