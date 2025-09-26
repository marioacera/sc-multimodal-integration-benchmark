

import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
rcParams['figure.figsize'] = (12,12)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import matplotlib.patheffects as path_effects


sc.settings.verbosity =0

def adjust_labels(coordinates, label_size, scale_factor,categories, max_iterations=5, displacement=1, neighborhood_radius=1.0):
    """
    Adjust label positions to minimize overlaps by moving labels towards areas of lower density.
    Once a label's position is adjusted and it has no overlaps, its position is fixed.

    :param coordinates: numpy array of shape (n_labels, 2) with initial label coordinates.
    :param label_sizes: numpy array of shape (n_labels, 2) with width and height of each label.
    :param scale_factor: scaling factor to adjust label sizes to UMAP coordinate space.
    :param max_iterations: maximum number of iterations for adjustment.
    :param displacement: step size for moving labels.
    :param neighborhood_radius: radius to consider for neighborhood density.
    :return: adjusted coordinates.
    """
    fixed_labels = [False] * len(coordinates)  # Track whether a label's position is fixed

    for _ in range(max_iterations):
        for i in range(len(coordinates)):
            if fixed_labels[i]:
                continue  # Skip this label if its position is already fixed

            direction = np.zeros(2)
            has_overlap = False
            for j in range(len(coordinates)):
                if not fixed_labels[j] and i != j and check_overlap(coordinates[i], coordinates[j],label_size, scale_factor, pad = label_size +20):
                    # print(categories[i],' ', categories[j], 'OVERLAP' )
                    has_overlap = True
                    direction += coordinates[i] - coordinates[j]

            if has_overlap:
                # Normalize and move label
                if np.linalg.norm(direction) != 0:
                    direction /= np.linalg.norm(direction)
                    less_dense_direction = find_less_dense_direction(coordinates, i, neighborhood_radius)
                    direction += less_dense_direction
                    direction /= np.linalg.norm(direction)
                    coordinates[i] += direction * displacement
            else:
                fixed_labels[i] = True  # Mark this label as fixed

        # Check if all labels are fixed, and if so, break the loop
        if all(fixed_labels):
            break

    return coordinates

def check_overlap(coord1, coord2, size, scale_factor=1.0, pad = 25):
    """
    Check if two labels overlap, accounting for scale differences between UMAP coordinates and label sizes.

    :param coord1: coordinates of the first label.
    :param coord2: coordinates of the second label.
    :param size1: size of the first label.
    :param size2: size of the second label.
    :param scale_factor: scaling factor to adjust label sizes to UMAP coordinate space.
    :return: True if labels overlap, False otherwise.
    """
    distance = np.linalg.norm(coord1 - coord2) - pad/scale_factor
    adjusted_size = size / scale_factor
    return distance < adjusted_size

def find_less_dense_direction(coordinates, index, radius):
    """
    Find a direction towards a less dense area for a label.

    :param coordinates: numpy array of label coordinates.
    :param index: index of the label to move.
    :param radius: radius to consider for neighborhood density.
    :return: a unit vector pointing towards a less dense area.
    """
    center = coordinates[index]
    density_vector = np.zeros(2)
    for i, coord in enumerate(coordinates):
        if i != index and np.linalg.norm(coord - center) < radius:
            density_vector += center - coord

    if np.linalg.norm(density_vector) == 0:
        return np.random.rand(2) - 0.5  # Random direction if density is uniform
    else:
        return density_vector / np.linalg.norm(density_vector)
def label_size_to_font_size(label_size):
    # This is a placeholder function. You need to define how to convert label size to font size
    return label_size / 2 

def shorten_line(start, end, shorten_by):
    direction = end - start
    norm_direction = direction / np.linalg.norm(direction)
    return end - norm_direction * shorten_by

# VARIABLES




def umap_refined(adata,
                umap,
                var,
                 size=10,
                label_size = 50,
                width_in_inches = 18,
                height_in_inches = 18,
                max_iterations=20,
                displacement=0.5):
    x_min, y_min = np.min(adata.obsm[umap], axis=0)
    x_max, y_max = np.max(adata.obsm[umap], axis=0)

    umap_width = x_max - x_min
    umap_height = y_max - y_min


    fig = plt.figure(figsize=(width_in_inches, height_in_inches))
    dpi = fig.dpi  # Get the DPI of the figure

    # Convert figure size to pixels
    plot_width_pixels = width_in_inches * dpi
    plot_height_pixels = height_in_inches * dpi

    # You might need to adjust for margins and paddings
    # These are approximate values and might need tweaking
    left_margin = 0.1 * plot_width_pixels
    right_margin = 0.1 * plot_width_pixels
    top_margin = 0.1 * plot_height_pixels
    bottom_margin = 0.1 * plot_height_pixels

    effective_plot_width = plot_width_pixels - (left_margin + right_margin)
    effective_plot_height = plot_height_pixels - (top_margin + bottom_margin)


    scale_factor_x = effective_plot_width / umap_width
    scale_factor_y = effective_plot_height / umap_height

    # Use the smaller scale factor to maintain aspect ratio
    scale_factor = min(scale_factor_x, scale_factor_y)
    print('Figure size=',(width_in_inches, height_in_inches),'dpi=',dpi, 'Scale factor =', scale_factor)

    ax = plt.subplot()
    sc.pl.embedding(adata, basis = umap, color = [var], frameon = False, size = size, show=False, legend_fontoutline=2, ax=ax )


    label_colors = adata.uns[f'{var}_colors']
    # Get initial label positions (centroids of clusters)
    categories = adata.obs[var].cat.categories  # Replace with your category column
    initial_coords = np.array([np.median(adata.obsm[umap][adata.obs[var] == cat], axis=0) for cat in categories])
    # for i in range(len(categories)):
        # print(categories[i], initial_coords[i])
  # Replace with your estimated sizes

    adjusted_coords = adjust_labels(initial_coords.copy(), label_size, scale_factor = scale_factor,categories = categories, max_iterations=2, displacement=0.5)

    scaled_initial_coords = initial_coords 
    scaled_adjusted_coords = adjusted_coords 

    for i, cat in enumerate(categories):
        start_coord = scaled_initial_coords[i]
        end_coord = scaled_adjusted_coords[i]

        # Calculate new endpoint shortened by half the font size
        font_size = label_size_to_font_size(label_size / 2)
        new_end_coord = shorten_line(start_coord, end_coord, font_size/scale_factor)

        # Draw line
        ax.plot([start_coord[0], new_end_coord[0]], 
                [start_coord[1], new_end_coord[1]], 
                color='black', linestyle='-', linewidth=1)

        # Plot label
        text = ax.text(end_coord[0], end_coord[1], cat, 
                       ha='center', va='center', 
                       fontsize=font_size, color=label_colors[i])

        # Add black outline to text
        text.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                               path_effects.Stroke(linewidth=2, foreground='black'),
                               path_effects.Normal()])
        




import seaborn as sns
def comparison_heatmap(adata, key1, key2, label_1=None, label_2=None, cmap = 'Reds', annot = True, figsize=(7,7)):
    if label_1==None:
        label_1=key1
    if label_2==None:
        label_2=key2
    expected_df = adata.obs[[key1,key2]].groupby(by=[key2,key1]).size().reset_index(name = 'count')
    counts = np.array(expected_df['count'].tolist())
    df = pd.DataFrame(counts.reshape(((len(adata.obs[key2].cat.categories),len(adata.obs[key1].cat.categories)))), index = expected_df[key2].unique(), columns = expected_df[key1].unique())
    if annot ==True:
        annot_ = df.astype(int)
        sc.settings.set_figure_params(figsize=figsize, color_map='inferno')
    else:
        annot_=None
        sc.settings.set_figure_params(figsize=figsize, color_map='inferno')
    s = sns.heatmap(df/np.sum(df,axis = 0), cbar_kws={'label': '% cell shared between annotations',"shrink": .5}, cmap=cmap, vmax=1, vmin=0, annot = annot_,  fmt='.7g',center=0.5,square=True, linewidths=.5)
    s.set_ylabel(label_2, fontsize=12)
    s.set_xlabel(label_1, fontsize = 12)
    # plt.show()
    return df

def comparison_heatmap_percent(adata, key1, key2, label_1=None, label_2=None, cmap = 'Reds', annot = True, figsize=(7,7)):
    if label_1==None:
        label_1=key1
    if label_2==None:
        label_2=key2
    expected_df = adata.obs[[key1,key2]].groupby(by=[key2,key1]).size().reset_index(name = 'count')
    counts = np.array(expected_df['count'].tolist())
    df = pd.DataFrame(counts.reshape(((len(adata.obs[key2].cat.categories),len(adata.obs[key1].cat.categories)))), index = expected_df[key2].unique(), columns = expected_df[key1].unique())
    if annot ==True:
        annot_ = df/np.sum(df,axis = 0)
        sc.settings.set_figure_params(figsize=figsize, color_map='inferno')
    else:
        annot_=None
        sc.settings.set_figure_params(figsize=figsize, color_map='inferno')
    s = sns.heatmap(df/np.sum(df,axis = 0), cbar_kws={'label': '% cell shared between annotations',"shrink": .5}, cmap=cmap, vmax=1, vmin=0, annot = annot_,  fmt='.1f',center=0.5,square=True, linewidths=.5)
    s.set_ylabel(label_2, fontsize=12)
    s.set_xlabel(label_1, fontsize = 12)
    # plt.show()
    return df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def custom_dotplot(
    adata,
    var_names,
    groupby,
    padd_list_x,
    category_order=None,
    figsize=(12, 6),
    cmap="coolwarm",
    short_padding=0.1,
    long_padding=0.5,
    x_padding_interval=3,
    dot_min=0.05,
    dot_max=1.0,
    only_expressed=False,
    vmax=None,
    vmin=None
):
    """
    Custom dot plot with legends adjusted and uniform padding, but with axes switched:
    Genes on the X-axis, Categories on the Y-axis.
    """
    # --- 1) Set up categories in the desired order ---
    categories = adata.obs[groupby].unique()
    if category_order:
        categories_order = category_order
    else:
        categories_order = sorted(categories)

    # --- 2) Compute percentage of cells expressing each gene, grouped by category ---
    percentages = pd.DataFrame(
        {
            cat: (adata[adata.obs[groupby] == cat, var_names].X > 0).mean(axis=0).A1 * 100
            for cat in categories_order
        },
        index=var_names,
    )

    # --- 3) Compute mean expression values (either non-zero only or full mean) ---
    if only_expressed:
        mean_expression = pd.DataFrame(index=var_names)
        for cat in categories_order:
            mean_values = []
            for gene in var_names:
                gene_data = adata[adata.obs[groupby] == cat, gene].raw.X
                expressed_values = gene_data[gene_data > 0]
                if expressed_values.size > 0.0:
                    mean_values.append(expressed_values.mean())
                else:
                    mean_values.append(0)
            mean_expression[cat] = mean_values
    else:
        mean_expression = pd.DataFrame(
            {
                cat: adata[adata.obs[groupby] == cat, var_names].X.mean(axis=0).A1
                for cat in categories_order
            },
            index=var_names,
        )

    # --- 4) Create a melted DataFrame for plotting ---
    data_for_plot = percentages.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Percentage"
    )
    data_for_plot["Mean Expression"] = mean_expression.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Expression"
    )["Expression"]

    # ----------------------------------------------------------------------------
    #                     MAIN AXIS-SWITCHING CHANGES BELOW
    # ----------------------------------------------------------------------------

    # Genes (var_names) -> X-axis
    # We'll assign each gene a numeric position (with no special spacing, but you can add your own logic if needed).
    x_positions = np.arange(len(var_names))
    counter=0
    i_save=0
    for i in range(1, len(x_positions)):
        if (i-i_save) % padd_list_x[counter] == 0:
            x_positions[i:] = x_positions[i:] + long_padding
            print(i, counter, padd_list_x[counter])
            counter +=1
            i_save = i
        else:
            x_positions[i:] = x_positions[i:] + short_padding

    gene_to_x = dict(zip(var_names, x_positions))
    data_for_plot["x"] = data_for_plot["index"].map(gene_to_x)

    # Categories -> Y-axis
    # We'll apply the short/long padding logic here.
    y_positions = np.arange(len(categories_order))
    for i in range(1, len(y_positions)):
        if i % x_padding_interval == 0:
            y_positions[i:] = y_positions[i:] + long_padding
        else:
            y_positions[i:] = y_positions[i:] + short_padding

    category_to_y = dict(zip(categories_order, y_positions))
    data_for_plot["y"] = data_for_plot["Category"].map(category_to_y)

    # ----------------------------------------------------------------------------
    #                                 PLOTTING
    # ----------------------------------------------------------------------------
    plt.figure(figsize=figsize)
    print(data_for_plot)
    scatter_plot = sns.scatterplot(
        data=data_for_plot,
        x="x",
        y="y",
        size="Percentage",
        sizes=(dot_min, dot_max),
        hue="Mean Expression",
        palette=cmap,
        edgecolor="black",
        linewidth=0.1,
    )

    # --- 5) Configure axis ticks & labels ---
    # X-axis: Genes
    scatter_plot.set_xticks(x_positions)
    scatter_plot.set_xticklabels(var_names, rotation=90)
    scatter_plot.set_xlabel("Genes")

    # Y-axis: Category
    scatter_plot.set_yticks(y_positions)
    scatter_plot.set_yticklabels(categories_order)
    scatter_plot.set_ylabel(groupby)
    scatter_plot.invert_yaxis()

    # --- 6) Adjust the dot size legend (percentage) ---
    handles, labels = scatter_plot.get_legend_handles_labels()
    # The size handles are usually the first few, but exact slicing can vary
    # depending on Seaborn version. Adjust as necessary.
    # Typically, size handles appear after all color handles. We look for "Percentage" or dot-min/dot-max in the labels.
    # A simple approach is to find where "Mean Expression" label starts and slice up to that point.
    try:
        mean_expr_index = labels.index("Mean Expression")
    except ValueError:
        mean_expr_index = len(labels)  # fallback if not found
    size_legend_handles = handles[:mean_expr_index]
    size_legend_labels = labels[:mean_expr_index]

    # Remove the default legend altogether
    scatter_plot.legend_.remove()

    # Re-add the legend for dot size
    plt.legend(
        handles=size_legend_handles,
        labels=size_legend_labels,
        title="Dot Size (%)",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
    )

    # --- 7) Add a colorbar for Mean Expression ---
    # If you want dynamic vmin/vmax, compute them from data; here, we assume user-specified or a default range.
    if vmin is None:
        vmin = data_for_plot["Mean Expression"].min()
    if vmax is None:
        vmax = data_for_plot["Mean Expression"].max()

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=scatter_plot,
        orientation="horizontal",  # You can switch to 'vertical' if desired
        pad=0.17,
        aspect=50
    )
    cbar.set_label("Mean Expression")

    plt.tight_layout()
    # plt.show()
    return data_for_plot
