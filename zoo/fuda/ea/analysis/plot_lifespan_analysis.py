import matplotlib.pyplot as plt                                                                                                                   
                                                                                                                             
                    

def plot_lifespan_analysis(gen_df, fitness_xlim=None):
    # 1. Prepare the data (handling the MultiIndex by resetting)
    temp_df = gen_df.reset_index()
    
    # 2. Identify the max generation present in the data
    max_gen = temp_df['gen'].max()
    
    # 3. Aggregate individual history
    individuals = temp_df.groupby('id').agg({
        'fitness': 'first',
        'gen': ['min', 'max']
    })
    
    individuals.columns = ['fitness', 'birth', 'death']
    individuals['lifespan'] = individuals['death'] - individuals['birth']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot 1: Lifespan distribution ---
    ax1.hist(individuals['lifespan'], bins=range(int(max_gen) + 2), 
             edgecolor='black', color='skyblue', alpha=0.7)
    ax1.set_xlabel('Lifespan (generations)')
    ax1.set_ylabel('Count (Number of Individuals)')
    ax1.set_title('Distribution of Individual Longevity')
    
    # --- Plot 2: Fitness vs lifespan ---
    ax2.scatter(individuals['fitness'], individuals['lifespan'], 
                alpha=0.4, s=20, label='Individual Data')
    
    # Add horizontal lines for Gen 0 and Max Gen
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Birth (Gen 0)', zorder=0)
    ax2.axhline(y=max_gen, color='green', linestyle='--', linewidth=1, label=f'End (Gen {max_gen})', zorder=0)
    
    ax2.set_xlabel('Fitness')
    ax2.set_ylabel('Lifespan (Total Generations Alive)')
    ax2.set_title('Fitness vs Lifespan')
    
    # Place legend to the side or inside
    ax2.legend(loc='upper left', frameon=True)
    
    if fitness_xlim:
        ax2.set_xlim(fitness_xlim)
        
    plt.tight_layout()
    plt.show()
