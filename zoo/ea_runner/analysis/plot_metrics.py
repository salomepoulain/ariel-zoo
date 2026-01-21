import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(gen_df, metrics=list[str], is_max=True):                                                                                          
    gen_grouped = gen_df.groupby(level='gen')                                                                                                                            
    x = np.arange(gen_grouped.ngroups)                                                                                                                                                                                                                                                                                       
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3*len(metrics)), sharex=True)                                                                                                                                                                                                                                    
    for ax, key in zip(axes, metrics):                                                                                                                                   
        mean, std = gen_grouped[key].mean().values, gen_grouped[key].std().values                                                                                        
        best = gen_grouped[key].max().values if is_max else gen_grouped[key].min().values                                                                                
                                                                                                                                                                        
        ax.plot(x, mean, 'b-', lw=2, label='Mean')                                                                                                                       
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color='blue')                                                                                              
        ax.plot(x, best, 'g--', lw=1.5, label='Best')                                                                                                                    
        ax.set_ylabel(key.capitalize())                                                                                                                                  
        ax.legend(loc='upper right')                                                                                                                                     
        # ax.grid(alpha=0.3)                                                                                                                                               
                                                                                                                                                                        
    axes[-1].set_xlabel('Generation')                                                                                                                                    
    plt.tight_layout()     
    fig.dpi = 300                                                                                                                                              
    plt.show()                                                                                                                                                           
                