import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_character_card(ax, name, role, traits, color_theme, position):
    # Background card
    rect = patches.FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle="round,pad=0.02", lw=2, edgecolor=color_theme, facecolor='#f0f0f0', transform=ax.transAxes)
    ax.add_patch(rect)
    
    # Header area
    header = patches.Rectangle((0.1, 0.75), 0.8, 0.15, facecolor=color_theme, alpha=0.8, transform=ax.transAxes)
    ax.add_patch(header)
    
    # Text: Name and Role
    ax.text(0.5, 0.85, name, fontsize=16, fontweight='bold', color='white', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.78, role, fontsize=10, color='white', ha='center', transform=ax.transAxes)
    
    # Traits
    y_pos = 0.65
    for trait in traits:
        ax.text(0.2, y_pos, f"■ {trait}", fontsize=11, color='#333333', transform=ax.transAxes)
        y_pos -= 0.08

    # Symbol Placeholder (Stylized icon)
    if "Shibata" in name:
        # Precision icon (Grid/Network)
        for i in range(3):
            ax.plot([0.65, 0.85], [0.3+i*0.05, 0.3+i*0.05], color=color_theme, alpha=0.3, transform=ax.transAxes)
            ax.plot([0.7+i*0.05, 0.7+i*0.05], [0.25, 0.45], color=color_theme, alpha=0.3, transform=ax.transAxes)
        ax.text(0.75, 0.35, "10^300", fontsize=8, color=color_theme, ha='center', transform=ax.transAxes)
    else:
        # BEP icon (X crossing)
        ax.plot([0.65, 0.85], [0.25, 0.45], color='red', lw=2, label='Cost', transform=ax.transAxes)
        ax.plot([0.65, 0.85], [0.45, 0.25], color='green', lw=2, label='Revenue', transform=ax.transAxes)
        ax.scatter([0.75], [0.35], color='gold', s=100, zorder=5, transform=ax.transAxes)

    ax.set_axis_off()

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    fig.patch.set_facecolor('#2c3e50')
    
    # Shibata Atsushi Data
    create_character_card(
        ax1, 
        "Atsushi Shibata", 
        "Visionary CEO & Engineer", 
        [
            "Precision: long double (10^300)",
            "Core: Deep Learning from Scratch",
            "Weapon: Numerical Simulation (C/Go)",
            "Motto: Precision is Truth"
        ],
        "#2980b9", # Blue theme
        "left"
    )
    
    # BEP-chan Data
    create_character_card(
        ax2, 
        "BEP-chan", 
        "The Guardian of Equilibrium", 
        [
            "Role: Profit/Loss Management",
            "Origin: break_even_pid.c",
            "Special: PID Control for Business",
            "Aura: Golden Intersection"
        ],
        "#d35400", # Orange theme
        "right"
    )
    
    plt.tight_layout()
    output_file = "character_profiles.png"
    plt.savefig(output_file, facecolor=fig.get_facecolor())
    print(f"Character profiles saved to {output_file}")

if __name__ == "__main__":
    main()
