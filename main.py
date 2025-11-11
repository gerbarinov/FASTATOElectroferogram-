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

def phred_to_prob(phred_char):
    """Convert Phred quality character to error probability"""
    phred_score = ord(phred_char) - 33  # Convert from ASCII to Phred score
    return 10 ** (-phred_score / 10)

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

def create_clean_electro(primary_sequence, quality_string=None, secondary_sequence=None, 
                        use_quality_metrics=False, output_file="clean_electro.png"):
    # Use shorter sequences for better visualization
    short_primary = primary_sequence[:80] if len(primary_sequence) > 80 else primary_sequence
    
    # Truncate quality string if provided
    if quality_string and len(quality_string) > len(short_primary):
        short_quality = quality_string[:len(short_primary)]
    else:
        short_quality = quality_string
    
    # Generate random secondary sequence if none provided
    if secondary_sequence is None:
        secondary_sequence = generate_random_sequence(len(short_primary), ambiguity_prob=0.1)
    short_secondary = secondary_sequence[:len(short_primary)]
    
    # Create figure with sequence track and combined electropherogram
    if use_quality_metrics and short_quality:
        fig, (ax_seq, ax_qual, ax_electro) = plt.subplots(3, 1, figsize=(25, 12), 
                                                        gridspec_kw={'height_ratios': [1, 0.5, 4]})
    else:
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
    
    def create_quality_peaks(sequence, traces_dict, quality_string=None, min_amplitude=0.7, max_amplitude=1.0):
        """Create Gaussian peaks with quality-based amplitude adjustment"""
        for i, base in enumerate(sequence):
            center = i
            
            # Base amplitude factor
            base_amp_factor = random.uniform(min_amplitude, max_amplitude)
            
            # Quality-based adjustment if quality metrics are provided and used
            quality_amp_factor = 1.0
            if use_quality_metrics and quality_string and i < len(quality_string):
                error_prob = phred_to_prob(quality_string[i])
                # High quality (low error prob) = higher amplitude, low quality = lower amplitude
                quality_amp_factor = 1.0 - (error_prob * 2)  # Scale factor based on error probability
                quality_amp_factor = max(0.3, min(1.0, quality_amp_factor))  # Clamp between 0.3-1.0
            
            final_amplitude_factor = base_amp_factor * quality_amp_factor
            
            # Handle IUPAC ambiguity codes
            if base in IUPAC_CODES and len(IUPAC_CODES[base]) > 1:
                possible_bases = IUPAC_CODES[base]
                for amb_base in possible_bases:
                    sigmas = {'A': 0.35, 'T': 0.4, 'C': 0.38, 'G': 0.36}
                    sigma = sigmas[amb_base]
                    
                    amplitudes = {'A': 0.85, 'T': 0.95, 'C': 0.88, 'G': 0.82}
                    amplitude = amplitudes[amb_base] * final_amplitude_factor * 0.6
                    
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
                amplitude = amplitudes.get(base, 0.9) * final_amplitude_factor
                
                main_peak = amplitude * np.exp(-0.5 * ((x_hr - center) / sigma) ** 2)
                shoulder_left = 0.15 * amplitude * np.exp(-0.5 * ((x_hr - center + 0.8) / (sigma*0.7)) ** 2)
                shoulder_right = 0.1 * amplitude * np.exp(-0.5 * ((x_hr - center - 0.8) / (sigma*0.7)) ** 2)
                
                full_peak = main_peak + shoulder_left + shoulder_right
                if base in traces_dict:
                    traces_dict[base] += full_peak
    
    # Create peaks for primary signal with optional quality-based adjustment
    if use_quality_metrics and short_quality:
        create_quality_peaks(short_primary, traces_primary, short_quality, 0.7, 1.0)
    else:
        create_quality_peaks(short_primary, traces_primary, min_amplitude=0.7, max_amplitude=1.0)
    
    # Create peaks for secondary signal with fixed low amplitude
    create_quality_peaks(short_secondary, traces_secondary, min_amplitude=0.05, max_amplitude=0.05)
    
    # ===== COMPLETELY TRANSPARENT SEQUENCE TRACK =====
    for i, base in enumerate(short_primary):
        color = ambiguity_colors.get(base, colors.get(base, 'black'))
        
        # Adjust alpha based on quality if quality metrics are used
        alpha = 1.0
        if use_quality_metrics and short_quality and i < len(short_quality):
            error_prob = phred_to_prob(short_quality[i])
            alpha = 1.0 - (error_prob * 3)  # Lower quality = more transparent
            alpha = max(0.4, min(1.0, alpha))
        
        ax_seq.text(i + 0.5, 0.5, base, ha='center', va='center', 
                   color=color, fontweight='bold', fontsize=12,
                   fontfamily='monospace', alpha=alpha)
    
    ax_seq.set_xlim(0, len(short_primary))
    ax_seq.set_ylim(0, 1)
    ax_seq.set_title('Sequence' + (' (Quality Adjusted)' if use_quality_metrics else ''), 
                    fontsize=14, fontweight='bold', pad=10)
    
    # Remove everything: no ticks, no labels, no grid, no borders
    ax_seq.set_xticks([])
    ax_seq.set_yticks([])
    ax_seq.set_xticklabels([])
    ax_seq.set_yticklabels([])
    ax_seq.grid(False)
    
    # Make ALL spines completely invisible
    for spine in ax_seq.spines.values():
        spine.set_visible(False)
    
    # ===== QUALITY METRICS TRACK (if enabled) =====
    if use_quality_metrics and short_quality:
        # Calculate quality scores for each position
        qual_scores = []
        for i, qual_char in enumerate(short_quality):
            if i < len(short_primary):
                phred_score = ord(qual_char) - 33
                qual_scores.append(phred_score)
        
        # Plot quality scores
        positions = range(len(qual_scores))
        bars = ax_qual.bar(positions, qual_scores, color='skyblue', alpha=0.7, width=0.8)
        
        # Color bars based on quality
        for j, (score, bar) in enumerate(zip(qual_scores, bars)):
            if score >= 30:
                bar.set_color('green')  # High quality
            elif score >= 20:
                bar.set_color('orange')  # Medium quality
            else:
                bar.set_color('red')  # Low quality
        
        ax_qual.set_ylabel('Q Score', fontsize=10, fontweight='bold')
        ax_qual.set_ylim(0, 40)
        ax_qual.set_xlim(0, len(short_primary))
        ax_qual.set_xticks([])
        ax_qual.grid(True, alpha=0.2, axis='y')
        
        # Clean styling for quality track
        for spine in ax_qual.spines.values():
            spine.set_visible(False)
    
    # ===== CLEAN ELECTROPHEROGRAM =====
    # Plot primary signals (solid lines, with optional quality adjustment)
    for base in 'ATCG':
        ax_electro.plot(x_hr, traces_primary[base], color=colors[base], 
                       linewidth=2.0, label=f'{base}', alpha=0.9)
    
    # Plot secondary signals (solid lines, 5% intensity)
    for base in 'ATCG':
        ax_electro.plot(x_hr, traces_secondary[base], color=colors[base], 
                       linewidth=1.5, alpha=0.6)
    
    quality_status = " (Quality Based)" if use_quality_metrics else ""
    ax_electro.set_title(f'Electropherogram{quality_status}', fontsize=14, fontweight='bold', pad=10)
    
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
    print(f"Quality metrics: {'ENABLED' if use_quality_metrics else 'DISABLED'}")
    if use_quality_metrics and short_quality:
        avg_quality = np.mean([ord(q) - 33 for q in short_quality])
        print(f"Average quality score: {avg_quality:.1f}")
    print(f"Primary intensity: Random between 0.7-1.0" + (" (quality adjusted)" if use_quality_metrics else ""))
    print(f"Secondary intensity: Fixed at 5%")

# Your sequences and quality data
primary_sequence = "TAAACATCATAGGCAGTCGGGAGGGTAGTCGGAACCGAAGAAGACTCAAAGCGAACCAAACAGGCAAAAAATTTAGGGTCGGCATCAAAAGCAATATCAGCACCAACAGAAACAACCTGATTAGCGGCGTTGACAGATGTATCCATCTGA"
quality_string = "EEEEEEEEEEEEEEEE@EE@@EEEE@EE@@EE@E@EEEEE@EE@@E@EEEEEE@E@EEEEEEEEEEEEEEEEEEEEEEEEEE6EE@EEEEEEEEEEEEEEEEEEEEEEEEEE@EEEEEEEEEEEEEE@E@@E@:EEEEEEEEEEEEEEEE"

# Option 1: With quality metrics enabled
print("\n=== WITH Quality Metrics ===")
create_clean_electro(primary_sequence, quality_string, use_quality_metrics=True, 
                    output_file="with_quality_metrics.png")

# Option 2: Without quality metrics (default behavior)
print("\n=== WITHOUT Quality Metrics ===")
create_clean_electro(primary_sequence, quality_string, use_quality_metrics=False,
                    output_file="without_quality_metrics.png")
