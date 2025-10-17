"""
Phase 6C Diagnostic Analysis - Test Log Analysis with Visualizations

Analyzes the training_data.log to extract detection performance and generate
visual diagnostics for understanding failure modes.

Usage:
    python -m analysis.phase6c_diagnostics.analyze_test_log
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class TestLogAnalyzer:
    """Analyzes training_data.log to extract detection performance."""

    def __init__(self, log_path: str):
        """
        Initialize analyzer.

        Args:
            log_path: Path to training_data.log file
        """
        self.log_path = log_path
        self.transects = []
        self.detections = []
        self.rejections = []

    def parse_log(self):
        """Parse log file to extract detection information."""
        print(f"Parsing log file: {self.log_path}")

        with open(self.log_path, 'r') as f:
            lines = f.readlines()

        current_transect = None

        for line in lines:
            # Extract transect ID
            match = re.search(r'\[(\d+)/10\] Processing transect (\d+)', line)
            if match:
                current_transect = int(match.group(2))
                self.transects.append({
                    'transect_id': current_transect,
                    'detected': False,
                    'detection_mode': None,
                    'distance': None,
                    'confidence': None,
                    'nir_drop': None,
                    'var_ratio': None,
                    'num_boundaries_phase2': 0,
                    'num_filtered': 0
                })

            # Extract shell line detection
            if 'Selected shell line' in line and current_transect is not None:
                match = re.search(
                    r'Selected shell line at ([\d.]+)m \(mode=(\w+), confidence=([\d.]+), '
                    r'rank=\d+/\d+ in zone, NIR_drop=([\d.]+), var_ratio=([\d.]+)\)',
                    line
                )
                if match:
                    self.transects[-1]['detected'] = True
                    self.transects[-1]['distance'] = float(match.group(1))
                    self.transects[-1]['detection_mode'] = match.group(2)
                    self.transects[-1]['confidence'] = float(match.group(3))
                    self.transects[-1]['nir_drop'] = float(match.group(4))
                    self.transects[-1]['var_ratio'] = float(match.group(5))

            # Extract Phase 2 boundary count
            if 'Detected' in line and 'zone boundaries (Phase 2)' in line and current_transect is not None:
                match = re.search(r'Detected (\d+) zone boundaries', line)
                if match:
                    self.transects[-1]['num_boundaries_phase2'] = int(match.group(1))

            # Extract filtering info
            if 'Filtered' in line and 'transitions' in line and current_transect is not None:
                match = re.search(r'Filtered (\d+) -> (\d+) transitions', line)
                if match:
                    before = int(match.group(1))
                    after = int(match.group(2))
                    self.transects[-1]['num_filtered'] = before - after

            # Extract rejection reasons
            if 'Rejected at' in line and current_transect is not None:
                match = re.search(
                    r'Rejected at ([\d.]+)m: (.+)',
                    line
                )
                if match:
                    distance = float(match.group(1))
                    reasons = match.group(2).strip()
                    self.rejections.append({
                        'transect_id': current_transect,
                        'distance': distance,
                        'reasons': reasons
                    })

        print(f"Parsed {len(self.transects)} transects")
        print(f"Found {len(self.rejections)} rejection records")

        return pd.DataFrame(self.transects), pd.DataFrame(self.rejections)

    def visualize_detection_overview(self, df: pd.DataFrame, output_dir: Path):
        """Create overview visualization of detection results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phase 6C Shell Line Detection - Performance Overview', fontsize=16, fontweight='bold')

        # 1. Detection success rate
        ax = axes[0, 0]
        detected_count = df['detected'].sum()
        not_detected_count = len(df) - detected_count
        colors = ['#2ecc71', '#e74c3c']
        ax.pie([detected_count, not_detected_count],
               labels=[f'Detected ({detected_count})', f'Not Detected ({not_detected_count})'],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title(f'Detection Success Rate: {detected_count}/{len(df)} ({detected_count/len(df)*100:.1f}%)')

        # 2. Detection mode breakdown
        ax = axes[0, 1]
        detected_df = df[df['detected'] == True]
        if not detected_df.empty:
            mode_counts = detected_df['detection_mode'].value_counts()
            colors_mode = {'strict': '#3498db', 'fallback': '#f39c12', 'strict_anywhere': '#9b59b6'}
            bars = ax.bar(range(len(mode_counts)), mode_counts.values,
                         color=[colors_mode.get(m, '#95a5a6') for m in mode_counts.index])
            ax.set_xticks(range(len(mode_counts)))
            ax.set_xticklabels(mode_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Count')
            ax.set_title('Detection Mode Distribution')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Detection Mode Distribution')

        # 3. Confidence distribution
        ax = axes[1, 0]
        if not detected_df.empty and 'confidence' in detected_df.columns:
            ax.hist(detected_df['confidence'].dropna(), bins=10, color='#3498db', alpha=0.7, edgecolor='black')
            ax.axvline(detected_df['confidence'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {detected_df["confidence"].mean():.2f}')
            ax.axvline(0.75, color='green', linestyle=':', linewidth=2, label='Strict threshold (0.75)')
            ax.axvline(0.50, color='orange', linestyle=':', linewidth=2, label='Fallback threshold (0.50)')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Count')
            ax.set_title('Confidence Score Distribution (Detected Shell Lines)')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Confidence Score Distribution')

        # 4. NIR drop vs variability ratio scatter
        ax = axes[1, 1]
        if not detected_df.empty:
            valid_data = detected_df[detected_df['nir_drop'] > 0]
            if not valid_data.empty:
                scatter = ax.scatter(valid_data['nir_drop'], valid_data['var_ratio'],
                                   c=valid_data['confidence'], cmap='viridis',
                                   s=100, alpha=0.7, edgecolors='black')
                ax.axvline(39, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='NIR threshold (39)')
                ax.axhline(1.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Var ratio threshold (1.5)')
                ax.set_xlabel('NIR Drop (absolute)')
                ax.set_ylabel('Variability Ratio')
                ax.set_title('Detection Characteristics (color = confidence)')
                ax.legend()
                ax.grid(alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Confidence')
            else:
                ax.text(0.5, 0.5, 'No valid detection data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Detection Characteristics')
        else:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Detection Characteristics')

        plt.tight_layout()
        output_path = output_dir / 'detection_overview.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved detection overview to {output_path}")
        plt.close()

    def visualize_rejections(self, rejections_df: pd.DataFrame, output_dir: Path):
        """Analyze and visualize rejection patterns."""
        if rejections_df.empty:
            print("No rejections to analyze")
            return

        # Parse rejection reasons
        rejection_categories = defaultdict(int)

        for reasons_str in rejections_df['reasons']:
            if 'NIR_drop=' in reasons_str:
                rejection_categories['NIR drop too small'] += 1
            if 'brightness=' in reasons_str:
                rejection_categories['Brightness too low'] += 1
            if 'NIR_before=' in reasons_str:
                rejection_categories['NIR before too low'] += 1
            if 'var_ratio=' in reasons_str:
                rejection_categories['Variability ratio too low'] += 1

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Rejection Analysis - Why Candidates Failed', fontsize=16, fontweight='bold')

        # 1. Rejection reasons breakdown
        ax = axes[0]
        if rejection_categories:
            categories = list(rejection_categories.keys())
            counts = list(rejection_categories.values())

            bars = ax.barh(categories, counts, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Number of Rejections')
            ax.set_title('Rejection Reasons (Multiple reasons per candidate possible)')
            ax.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{int(width)}',
                       ha='left', va='center', fontweight='bold')

        # 2. Rejections per transect
        ax = axes[1]
        rejections_per_transect = rejections_df.groupby('transect_id').size()
        ax.bar(rejections_per_transect.index, rejections_per_transect.values,
               color='#e67e22', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Transect ID')
        ax.set_ylabel('Number of Rejected Candidates')
        ax.set_title('Rejected Candidates per Transect')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / 'rejection_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved rejection analysis to {output_path}")
        plt.close()

        # Print rejection summary
        print("\n" + "="*80)
        print("REJECTION SUMMARY")
        print("="*80)
        print(f"Total rejections: {len(rejections_df)}")
        print(f"\nRejection reasons breakdown:")
        for reason, count in sorted(rejection_categories.items(), key=lambda x: -x[1]):
            percentage = count / len(rejections_df) * 100
            print(f"  {reason:30s}: {count:3d} ({percentage:5.1f}%)")

    def visualize_phase_progression(self, df: pd.DataFrame, output_dir: Path):
        """Visualize detection pipeline progression (Phase 2 -> Filtering -> Final)."""
        fig, ax = plt.subplots(figsize=(12, 6))

        transect_ids = df['transect_id'].values
        phase2_counts = df['num_boundaries_phase2'].values
        filtered_counts = df['num_filtered'].values
        detected_counts = df['detected'].astype(int).values

        x = np.arange(len(transect_ids))
        width = 0.25

        bars1 = ax.bar(x - width, phase2_counts, width, label='Phase 2 Detected', color='#3498db', alpha=0.7)
        bars2 = ax.bar(x, filtered_counts, width, label='Filtered Out', color='#e74c3c', alpha=0.7)
        bars3 = ax.bar(x + width, detected_counts, width, label='Final Detection', color='#2ecc71', alpha=0.7)

        ax.set_xlabel('Transect ID')
        ax.set_ylabel('Count')
        ax.set_title('Detection Pipeline Progression (Phase 2 -> Filtering -> Final)')
        ax.set_xticks(x)
        ax.set_xticklabels(transect_ids, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / 'pipeline_progression.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved pipeline progression to {output_path}")
        plt.close()

    def generate_summary_report(self, df: pd.DataFrame, rejections_df: pd.DataFrame, output_dir: Path):
        """Generate text summary report."""
        report = []
        report.append("=" * 80)
        report.append("PHASE 6C DIAGNOSTIC ANALYSIS - TEST LOG SUMMARY")
        report.append("=" * 80)
        report.append("")

        # Overall performance
        detected_count = df['detected'].sum()
        total_count = len(df)
        detection_rate = detected_count / total_count if total_count > 0 else 0

        report.append("OVERALL PERFORMANCE")
        report.append("-" * 80)
        report.append(f"Total transects:       {total_count}")
        report.append(f"Detected shell lines:  {detected_count} ({detection_rate*100:.1f}%)")
        report.append(f"Failed detections:     {total_count - detected_count} ({(1-detection_rate)*100:.1f}%)")
        report.append("")

        # Detection mode breakdown
        if detected_count > 0:
            report.append("DETECTION MODE BREAKDOWN")
            report.append("-" * 80)
            mode_counts = df[df['detected'] == True]['detection_mode'].value_counts()
            for mode, count in mode_counts.items():
                percentage = count / detected_count * 100
                report.append(f"{mode:20s}: {count:2d} ({percentage:5.1f}%)")
            report.append("")

        # Confidence statistics
        if detected_count > 0:
            confidences = df[df['detected'] == True]['confidence'].dropna()
            report.append("CONFIDENCE STATISTICS")
            report.append("-" * 80)
            report.append(f"Mean:       {confidences.mean():.3f}")
            report.append(f"Median:     {confidences.median():.3f}")
            report.append(f"Std Dev:    {confidences.std():.3f}")
            report.append(f"Min:        {confidences.min():.3f}")
            report.append(f"Max:        {confidences.max():.3f}")
            report.append("")

        # Pipeline attrition
        report.append("PIPELINE ATTRITION ANALYSIS")
        report.append("-" * 80)
        total_phase2 = df['num_boundaries_phase2'].sum()
        total_filtered = df['num_filtered'].sum()
        report.append(f"Total boundaries detected (Phase 2):  {total_phase2}")
        report.append(f"Total boundaries filtered out:        {total_filtered}")
        report.append(f"Final detections:                     {detected_count}")
        if total_phase2 > 0:
            attrition_rate = (total_phase2 - detected_count) / total_phase2 * 100
            report.append(f"Attrition rate:                       {attrition_rate:.1f}%")
        report.append("")

        # Rejection analysis
        if not rejections_df.empty:
            report.append("REJECTION ANALYSIS")
            report.append("-" * 80)
            report.append(f"Total rejected candidates: {len(rejections_df)}")
            report.append(f"Average rejections per transect: {len(rejections_df) / total_count:.1f}")
            report.append("")

        report.append("="  * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)
        report.append("")

        if detection_rate < 0.85:
            report.append("[!] Detection rate below target (85%). Issues identified:")
            report.append("")

            if detected_count > 0:
                fallback_count = (df[df['detected'] == True]['detection_mode'] == 'fallback').sum()
                if fallback_count / detected_count > 0.5:
                    report.append("  1. FALLBACK MODE OVERUSE")
                    report.append(f"     - {fallback_count}/{detected_count} detections required fallback")
                    report.append("     - Indicates strict thresholds are too harsh")
                    report.append("     - Action: Relax Phase 6C thresholds further")
                    report.append("")

            if not rejections_df.empty:
                report.append("  2. HIGH REJECTION RATE")
                report.append(f"     - {len(rejections_df)} candidates rejected")
                report.append("     - Action: Analyze rejection reasons and implement soft scoring")
                report.append("")

            if total_filtered > detected_count * 2:
                report.append("  3. EXCESSIVE FILTERING")
                report.append(f"     - {total_filtered} boundaries filtered out vs {detected_count} kept")
                report.append("     - Action: Review _filter_transitions() logic")
                report.append("")

        else:
            report.append("[OK] Detection rate meets target!")
            report.append("")

        report.append("=" * 80)

        # Write report
        report_text = "\n".join(report)
        output_path = output_dir / 'diagnostic_summary.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\nSaved summary report to {output_path}")
        print("\n" + report_text)


def main():
    """Main diagnostic workflow."""
    base_dir = Path(__file__).parent.parent.parent
    log_path = base_dir / "analysis" / "training_output" / "training_data.log"
    output_dir = base_dir / "analysis" / "phase6c_diagnostics"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("PHASE 6C DIAGNOSTIC ANALYSIS - TEST LOG ANALYZER")
    print("=" * 80)
    print(f"\nLog file: {log_path}")
    print(f"Output directory: {output_dir}")
    print("")

    if not log_path.exists():
        print(f"ERROR: Log file not found at {log_path}")
        return 1

    # Parse log
    analyzer = TestLogAnalyzer(str(log_path))
    df, rejections_df = analyzer.parse_log()

    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.visualize_detection_overview(df, output_dir)
    analyzer.visualize_rejections(rejections_df, output_dir)
    analyzer.visualize_phase_progression(df, output_dir)

    # Generate summary report
    analyzer.generate_summary_report(df, rejections_df, output_dir)

    # Save dataframes
    df.to_csv(output_dir / 'detection_results.csv', index=False)
    if not rejections_df.empty:
        rejections_df.to_csv(output_dir / 'rejection_details.csv', index=False)

    print("\n[DONE] Diagnostic analysis complete!")
    print(f"  - Visualizations saved to: {output_dir}")
    print(f"  - Detection results CSV: {output_dir / 'detection_results.csv'}")
    print(f"  - Summary report: {output_dir / 'diagnostic_summary.txt'}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
