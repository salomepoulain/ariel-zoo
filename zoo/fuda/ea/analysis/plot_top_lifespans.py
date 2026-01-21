import matplotlib.pyplot as plt                                                                                                                   
import matplotlib.cm as cm                                                                                                                        
import numpy as np                                                                                                                                
                    

def plot_top_lifespans(gen_df,*, is_maximalisation: bool = True, column='fitness', top_x=5, title=None):                                                                    
    """Plot lifespan of individuals who were ever in top X of any generation."""                                                                                                                                                                                                          
    df = gen_df.reset_index()                                                                                                                     
                                                                                                                                                
    # For each generation, get the top X individual IDs                                                                                           
    top_per_gen = (df                                                                                                                             
        .sort_values([column], ascending=not is_maximalisation)                                                                              
        .groupby('gen')                                                                                                                           
        .head(top_x)                                                                                                                              
    )                                                                                                                                             
                                                                                                                                                
    # Get all unique individuals who were ever in top X                                                                                           
    top_individuals = top_per_gen['id'].unique()                                                                                       
                                                                                                                                                
    # Get their full lifespan data                                                                                                                
    lifespan_data = (df[df['id'].isin(top_individuals)]                                                                                
        .groupby('id')                                                                                                                 
        .agg({                                                                                                                                    
            column: 'first',                                                                                                                      
            'gen': ['min', 'max']                                                                                                                 
        })                                                                                                                                        
    )                                                                                                                                             
    lifespan_data.columns = [column, 'birth', 'death']                                                                                            
    lifespan_data = lifespan_data.sort_values(column, ascending=not is_maximalisation)                                                       
                                                                                                                                                
    # Plot                                                                                                                                        
    fig, ax = plt.subplots(figsize=(20, 8))                                                                                                       
                                                                                                                                                
    # Color map                                                                                                                                   
    n = len(lifespan_data)                                                                                                                        
    colors = cm.viridis(np.linspace(0, 1, n))                                                                                                     
                                                                                                                                                
    for i, (ind_id, row) in enumerate(lifespan_data.iterrows()):                                                                                  
        # Horizontal line from birth to death                                                                                                     
        ax.hlines(y=row[column], xmin=row['birth'], xmax=row['death'],                                                                            
                color=colors[i], linewidth=2, alpha=0.7)                                                                                        
                                                                                                                                                
        # Markers                                                                                                                                 
        # ax.scatter(row['birth'], row[column], color=colors[i], s=20, marker='o', zorder=0)                                                        
        # ax.scatter(row['death'], row[column], color=colors[i], s=20, marker='X', zorder=0)                                                        
                                                                                                                                                
        # Label                                                                                                                                   
        ax.annotate(f'{int(ind_id)}', (row['death'] + 0.2, row[column]), fontsize=7, va='center')                                                 
                                                                                                                                                
    # Mark which generations each was in top X                                                                                                    
    for gen in df['gen'].unique():                                                                                                                
        gen_top = top_per_gen[top_per_gen['gen'] == gen]['id'].values                                                                  
        for ind_id in gen_top:                                                                                                                    
            fit = lifespan_data.loc[ind_id, column]                                                                                               
            ax.scatter(gen, fit, color='red', s=20, marker='s', alpha=0.5, zorder=4)                                                              
                                                                                                                                                
    ax.set_xlabel('Generation')                                                                                                                   
    ax.set_ylabel(column)                                                                                                                         
    ax.set_title(title or f'Individuals Ever in Top {top_x} (in top {top_x} that gen)')                                                     
    ax.grid(True, alpha=0.3)                                                                                                                      
                                                                                                                                                
    from matplotlib.lines import Line2D                                                                                                           
    legend_elements = [                                                                                                                           
        # Line2D([0], [0], marker='o', color='gray', label='Birth', markersize=8, linestyle=''),                                                    
        # Line2D([0], [0], marker='X', color='gray', label='Death', markersize=8, linestyle=''),                                                    
        Line2D([0], [0], marker='s', color='red', label=f'In top {top_x}', markersize=8, linestyle='', alpha=0.5),                                
    ]                                                                                                                                             
    ax.legend(handles=legend_elements, loc='lower right')                                                                                         
    fig.dpi = 300                                                                                                                                                
    plt.tight_layout()                                                                                                                            
    return fig, ax   
