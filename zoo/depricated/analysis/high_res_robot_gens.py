import canonical_toolkit as ctk
import numpy as np

def high_res_robot_gens(gen_df, config, top_n=3, n_samples=5) -> ctk.GridPlotter:
    """
    Plot top/bottom robots across sampled generations.

    Args:
        gen_df: DataFrame with MultiIndex (gen, rank)
        config: Config object with IS_MAXIMISATION, NUM_GENERATIONS
        top_n: Positive for best, negative for worst
        n_samples: Number of generations to sample

    Returns:
        GridPlotter object (call .plot() to display)
    """
    generations = np.linspace(0, config.NUM_GENERATIONS, n_samples, dtype=int)
    n = abs(top_n)

    data_2d = [[] for _ in range(n)]
    titles_2d = [[] for _ in range(n)]

    df_sorted = gen_df.sort_values(
        by=['gen', 'fitness'],
        ascending=[True, not config.IS_MAXIMISATION]
    )

    for gen in generations:
        gen_data = df_sorted.loc[gen]
        selection = gen_data.head(n) if top_n > 0 else gen_data.tail(n)

        for j, row in enumerate(selection.itertuples()):
            img = ctk.quick_view(
                ctk.node_from_string(row.ctk_string).to_graph(),
                return_img=True,
                white_background=True
            )
            data_2d[j].append(img)
            titles_2d[j].append(f"Gen {gen} | ID {row.id} | fit={row.fitness:.3f}")

    plotter = ctk.GridPlotter()
    plotter.config.title_size = 5
    plotter.config.margin = (0.3, 0, 0, 0)
    plotter.config.col_space = 0.23
    plotter.config.dpi = 300
    plotter.add_2D_image_data(data_2d, titles_2d=titles_2d)

    label = "Fittest" if (top_n > 0) == config.IS_MAXIMISATION else "Least Fit"
    plotter.suptitle(f"{label} {n} Robots Across Generations", font_size=8)

    return plotter


def high_res_robot_gens(gen_df, config, top_n=3, n_samples=5, by='fitness') -> ctk.GridPlotter:
    """
    Plot top/bottom robots across sampled generations based on a specific metric.

    Args:
        gen_df: DataFrame with MultiIndex (gen, rank)
        config: Config object with IS_MAXIMISATION, NUM_GENERATIONS
        top_n: Positive for best, negative for worst
        n_samples: Number of generations to sample
        by: The column name to sort and display (e.g., 'fitness', 'speed', 'novelty')

    Returns:
        GridPlotter object
    """
    generations = np.linspace(0, config.NUM_GENERATIONS, n_samples, dtype=int)
    n = abs(top_n)

    data_2d = [[] for _ in range(n)]
    titles_2d = [[] for _ in range(n)]

    # Sort by generation and the chosen metric (by)
    df_sorted = gen_df.sort_values(
        by=['gen', by],
        ascending=[True, not config.IS_MAXIMISATION]
    )

    for gen in generations:
        gen_data = df_sorted.loc[gen]
        selection = gen_data.head(n) if top_n > 0 else gen_data.tail(n)

        for j, row in enumerate(selection.itertuples()):
            img = ctk.quick_view(
                ctk.node_from_string(row.ctk_string).to_graph(),
                return_img=True,
                white_background=True
            )
            data_2d[j].append(img)

            # Dynamically get the value for the chosen metric
            metric_val = getattr(row, by)
            titles_2d[j].append(f"Gen {gen} | ID {row.id} | {by}={metric_val:.3f}")

    plotter = ctk.GridPlotter()
    plotter.config.title_size = 10
    plotter.config.margin = (0.3, 0, 0, 0)
    plotter.config.col_space = 0.23
    plotter.config.dpi = 300
    plotter.add_2D_image_data(data_2d, titles_2d=titles_2d)

    label = "Best" if (top_n > 0) == config.IS_MAXIMISATION else "Worst"
    plotter.suptitle(f"{label} {n} Robots by {by.capitalize()} Across Generations", font_size=8)

    return plotter
