import matplotlib.pyplot as plt
import numpy as np
import random

# IUPAC ambiguity codes dictionary
IUPAC_CODES = {
    'A': 'A', 'T': 'T', 'C': 'C', 'G': 'G',
    'R': 'AG', 'Y': 'CT', 'S': 'GC', 'W': 'AT',
    'K': 'GT', 'M': 'AC', 'B': 'CGT', 'D': 'AGT',
    'H': 'ACT', 'V': 'ACG', 'N': 'ACGT'
}

def generate_random_sequence(length=80, ambiguity_prob=0.1):
    """Generate a random DNA sequence with occasional IUPAC ambiguity codes"""
    bases = ['A', 'T', 'C', 'G']
    ambiguous_bases = ['R', 'Y', 'S', 'W', 'K', 'M', 'N']
    
    sequence = []
    for i in range(length):
        if random.random() < ambiguity_prob:
            sequence.append(random.choice(ambiguous_bases))
        else:
            sequence.append(random.choice(bases))
    
    return ''.join(sequence)

def create_clean_electro(primary_sequence, secondary_sequence=None, output_file="clean_electro.png"):
    # Use shorter sequences for better visualization
    short_primary = primary_sequence[:80] if len(primary_sequence) > 80 else primary_sequence
    
    # Generate random secondary sequence if none provided
    if secondary_sequence is None:
        secondary_sequence = generate_random_sequence(len(short_primary), ambiguity_prob=0.1)
    short_secondary = secondary_sequence[:len(short_primary)]
    
    # Create figure with sequence track and combined electropherogram
    fig, (ax_seq, ax_electro) = plt.subplots(2, 1, figsize=(25, 10), 
                                            gridspec_kw={'height_ratios': [1, 4]})
    
    # High resolution for smooth curves
    x_hr = np.linspace(0, len(short_primary)-1, len(short_primary)*20)
    
    # Initialize traces for both signals
    traces_primary = {'A': np.zeros(len(x_hr)), 'T': np.zeros(len(x_hr)), 
                     'C': np.zeros(len(x_hr)), 'G': np.zeros(len(x_hr))}
    traces_secondary = {'A': np.zeros(len(x_hr)), 'T': np.zeros(len(x_hr)), 
                       'C': np.zeros(len(x_hr)), 'G': np.zeros(len(x_hr))}
    
    colors = {'A': 'green', 'T': 'red', 'C': 'blue', 'G': 'black'}
    ambiguity_colors = {'R': 'orange', 'Y': 'purple', 'S': 'cyan', 'W': 'pink',
                       'K': 'brown', 'M': 'olive', 'B': 'magenta', 'D': 'navy',
                       'H': 'teal', 'V': 'coral', 'N': 'gray'}
    
    def create_random_peaks(sequence, traces_dict, min_amplitude=0.7, max_amplitude=1.0):
        """Create clean Gaussian peaks with random amplitude between min and max"""
        for i, base in enumerate(sequence):
            center = i
            
            # Generate random amplitude factor for this base position
            random_amp = random.uniform(min_amplitude, max_amplitude)
            
            # Handle IUPAC ambiguity codes
            if base in IUPAC_CODES and len(IUPAC_CODES[base]) > 1:
                possible_bases = IUPAC_CODES[base]
                for amb_base in possible_bases:
                    sigmas = {'A': 0.35, 'T': 0.4, 'C': 0.38, 'G': 0.36}
                    sigma = sigmas[amb_base]
                    
                    amplitudes = {'A': 0.85, 'T': 0.95, 'C': 0.88, 'G': 0.82}
                    amplitude = amplitudes[amb_base] * random_amp * 0.6
                    
                    main_peak = amplitude * np.exp(-0.5 * ((x_hr - center) / sigma) ** 2)
                    shoulder_left = 0.15 * amplitude * np.exp(-0.5 * ((x_hr - center + 0.8) / (sigma*0.7)) ** 2)
                    shoulder_right = 0.1 * amplitude * np.exp(-0.5 * ((x_hr - center - 0.8) / (sigma*0.7)) ** 2)
                    
                    full_peak = main_peak + shoulder_left + shoulder_right
                    traces_dict[amb_base] += full_peak
            else:
                # For standard bases
                sigmas = {'A': 0.35, 'T': 0.4, 'C': 0.38, 'G': 0.36}
                sigma = sigmas.get(base, 0.38)
                
                amplitudes = {'A': 0.85, 'T': 0.95, 'C': 0.88, 'G': 0.82}
                amplitude = amplitudes.get(base, 0.9) * random_amp
                
                main_peak = amplitude * np.exp(-0.5 * ((x_hr - center) / sigma) ** 2)
                shoulder_left = 0.15 * amplitude * np.exp(-0.5 * ((x_hr - center + 0.8) / (sigma*0.7)) ** 2)
                shoulder_right = 0.1 * amplitude * np.exp(-0.5 * ((x_hr - center - 0.8) / (sigma*0.7)) ** 2)
                
                full_peak = main_peak + shoulder_left + shoulder_right
                if base in traces_dict:
                    traces_dict[base] += full_peak
    
    # Create peaks for primary signal with random amplitudes between 0.7-1.0
    create_random_peaks(short_primary, traces_primary, min_amplitude=0.7, max_amplitude=1.0)
    
    # Create peaks for secondary signal with fixed low amplitude
    create_random_peaks(short_secondary, traces_secondary, min_amplitude=0.05, max_amplitude=0.05)
    
    # ===== COMPLETELY TRANSPARENT SEQUENCE TRACK =====
    for i, base in enumerate(short_primary):
        color = ambiguity_colors.get(base, colors.get(base, 'black'))
        
        # Just the sequence letters - no background, no borders
        ax_seq.text(i + 0.5, 0.5, base, ha='center', va='center', 
                   color=color, fontweight='bold', fontsize=12,
                   fontfamily='monospace')
    
    ax_seq.set_xlim(0, len(short_primary))
    ax_seq.set_ylim(0, 1)
    ax_seq.set_title('Sequence', fontsize=14, fontweight='bold', pad=10)
    
    # Remove everything: no ticks, no labels, no grid, no borders
    ax_seq.set_xticks([])
    ax_seq.set_yticks([])
    ax_seq.set_xticklabels([])
    ax_seq.set_yticklabels([])
    ax_seq.grid(False)
    
    # Make ALL spines completely invisible
    for spine in ax_seq.spines.values():
        spine.set_visible(False)
    
    # ===== CLEAN ELECTROPHEROGRAM =====
    # Plot primary signals (solid lines, random intensity 0.7-1.0)
    for base in 'ATCG':
        ax_electro.plot(x_hr, traces_primary[base], color=colors[base], 
                       linewidth=2.0, label=f'{base}', alpha=0.9)
    
    # Plot secondary signals (solid lines, 5% intensity)
    for base in 'ATCG':
        ax_electro.plot(x_hr, traces_secondary[base], color=colors[base], 
                       linewidth=1.5, alpha=0.6)
    
    ax_electro.set_title('Electropherogram', fontsize=14, fontweight='bold', pad=10)
    
    # Remove x-axis elements
    ax_electro.set_xlabel('')
    ax_electro.set_xticks([])
    ax_electro.set_xticklabels([])
    
    ax_electro.set_ylabel('Signal Intensity', fontsize=12, fontweight='bold')
    
    # Simplified legend
    ax_electro.legend(fontsize=11, loc='upper right')
    
    ax_electro.grid(True, alpha=0.1)
    ax_electro.set_ylim(-0.05, 1.0)
    ax_electro.set_xlim(0, len(short_primary)-1)
    
    # Clean styling for electropherogram
    ax_electro.spines['top'].set_visible(False)
    ax_electro.spines['right'].set_visible(False)
    ax_electro.spines['bottom'].set_visible(False)
    ax_electro.spines['left'].set_visible(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Generated clean electropherogram for {len(short_primary)} bases")
    print(f"Primary intensity: Random between 0.7-1.0 for each base")
    print(f"Secondary intensity: Fixed at 5%")

# Your primary sequence
primary_sequence = "AATTTAGGGTCGGCATCAAAAGCAAYATCAGCACCAACAGAAACAACCTGARTAGCGGCGTTGACAGATGTATCCATCTGA"

# Clean version with transparent sequence and random intensities
print("\n=== Clean Electropherogram - Transparent Sequence ===")
create_clean_electro(primary_sequence, output_file="clean_transparent_seq.png")
