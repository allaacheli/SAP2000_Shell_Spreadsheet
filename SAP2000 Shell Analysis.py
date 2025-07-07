import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SAP2000ShellAnalyzer:
    def __init__(self, excel_file_path):
        """
        Initialize the SAP2000 Shell Analyzer
        
        Parameters:
        excel_file_path (str): Path to the Excel file
        """
        self.file_path = excel_file_path
        self.data = None
        self.load_cases = []
        self.envelope_data = None
        
    def read_excel_data(self, sheet_name=0):
        """
        Read Excel data with SAP2000 format:
        Row 1: Table title
        Row 2: Column names
        Row 3: Units
        Row 4+: Data
        """
        try:
            # Read the raw data
            raw_data = pd.read_excel(self.file_path, sheet_name=sheet_name, header=None)
            
            # Extract components
            table_title = raw_data.iloc[0, 0] if not pd.isna(raw_data.iloc[0, 0]) else "SAP2000 Shell Data"
            column_names = raw_data.iloc[1, :].values
            units = raw_data.iloc[2, :].values
            
            # Create DataFrame with proper headers
            data = raw_data.iloc[3:, :].copy()
            data.columns = column_names
            
            # Clean up the data
            data = data.dropna(how='all')  # Remove completely empty rows
            data = data.reset_index(drop=True)
            
            # Convert numeric columns
            numeric_columns = ['As1Top', 'As2Top', 'As1Bot', 'As2Bot', 'Asw/s', 'Asw1/s', 'Asw2/s', 
                             'TopLayThick', 'BotLayThick', 'JointThick', 'CoverTop1', 'CoverTop2', 
                             'CoverBot1', 'CoverBot2', 'RebarPct', 'F11', 'F22', 'F12']
            
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            self.data = data
            self.table_title = table_title
            self.units = dict(zip(column_names, units))
            
            print(f"✓ Successfully loaded: {table_title}")
            print(f"✓ Data shape: {data.shape}")
            print(f"✓ Available columns: {list(data.columns)}")
            
            # Identify unique load cases
            if 'OutputCase' in data.columns:
                self.load_cases = data['OutputCase'].unique().tolist()
                print(f"✓ Found {len(self.load_cases)} load cases: {self.load_cases}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error reading Excel file: {str(e)}")
            return False
    
    def create_reinforcement_envelope(self):
        """
        Create envelope of maximum reinforcement for top and bottom
        """
        if self.data is None:
            print("❌ No data loaded. Please run read_excel_data() first.")
            return None
        
        try:
            # Group by element and find maximum reinforcement
            envelope_cols = ['Area', 'ElemType', 'ShellType']
            rebar_cols = ['As1Top', 'As2Top', 'As1Bot', 'As2Bot']
            
            # Add any other columns you want to keep
            other_cols = ['TopLayThick', 'BotLayThick', 'JointThick']
            
            all_cols = envelope_cols + rebar_cols + other_cols
            available_cols = [col for col in all_cols if col in self.data.columns]
            
            # Create envelope by taking maximum values for each element
            envelope = self.data.groupby('Area').agg({
                **{col: 'first' for col in envelope_cols if col in self.data.columns},
                **{col: 'max' for col in rebar_cols if col in self.data.columns},
                **{col: 'first' for col in other_cols if col in self.data.columns},
                'OutputCase': lambda x: ', '.join(x.astype(str))  # Combine all load cases
            }).reset_index()
            
            # Calculate envelope reinforcement
            envelope['MaxTopRebar'] = envelope[['As1Top', 'As2Top']].max(axis=1)
            envelope['MaxBotRebar'] = envelope[['As1Bot', 'As2Bot']].max(axis=1)
            envelope['MaxOverallRebar'] = envelope[['MaxTopRebar', 'MaxBotRebar']].max(axis=1)
            
            # Add direction indicators
            envelope['TopDirection'] = envelope[['As1Top', 'As2Top']].idxmax(axis=1)
            envelope['BotDirection'] = envelope[['As1Bot', 'As2Bot']].idxmax(axis=1)
            
            self.envelope_data = envelope
            
            print(f"✓ Envelope created with {len(envelope)} elements")
            return envelope
            
        except Exception as e:
            print(f"❌ Error creating envelope: {str(e)}")
            return None
    
    def get_summary_statistics(self):
        """
        Get summary statistics of the reinforcement
        """
        if self.envelope_data is None:
            print("❌ No envelope data. Please run create_reinforcement_envelope() first.")
            return None
        
        stats = {}
        
        # Overall statistics
        rebar_cols = ['As1Top', 'As2Top', 'As1Bot', 'As2Bot', 'MaxTopRebar', 'MaxBotRebar', 'MaxOverallRebar']
        available_rebar_cols = [col for col in rebar_cols if col in self.envelope_data.columns]
        
        for col in available_rebar_cols:
            stats[col] = {
                'min': self.envelope_data[col].min(),
                'max': self.envelope_data[col].max(),
                'mean': self.envelope_data[col].mean(),
                'std': self.envelope_data[col].std(),
                'count': self.envelope_data[col].count()
            }
        
        return stats
    
    def find_critical_elements(self, top_n=10):
        """
        Find elements with highest reinforcement requirements
        """
        if self.envelope_data is None:
            print("❌ No envelope data. Please run create_reinforcement_envelope() first.")
            return None
        
        # Sort by maximum overall reinforcement
        critical = self.envelope_data.nlargest(top_n, 'MaxOverallRebar')
        
        return critical[['Area', 'ElemType', 'MaxTopRebar', 'MaxBotRebar', 'MaxOverallRebar', 'OutputCase']]
    
    def plot_reinforcement_distribution(self, figsize=(15, 10)):
        """
        Create comprehensive plots of reinforcement distribution
        """
        if self.envelope_data is None:
            print("❌ No envelope data. Please run create_reinforcement_envelope() first.")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('SAP2000 Shell Reinforcement Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top vs Bottom Reinforcement Scatter
        axes[0,0].scatter(self.envelope_data['MaxTopRebar'], self.envelope_data['MaxBotRebar'], 
                         alpha=0.6, s=20)
        axes[0,0].set_xlabel('Max Top Reinforcement')
        axes[0,0].set_ylabel('Max Bottom Reinforcement')
        axes[0,0].set_title('Top vs Bottom Reinforcement')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add diagonal line
        max_val = max(self.envelope_data['MaxTopRebar'].max(), self.envelope_data['MaxBotRebar'].max())
        axes[0,0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal reinforcement')
        axes[0,0].legend()
        
        # 2. Overall Reinforcement Distribution
        axes[0,1].hist(self.envelope_data['MaxOverallRebar'], bins=30, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Max Overall Reinforcement')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Overall Reinforcement Distribution')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Top Reinforcement by Direction
        if 'As1Top' in self.envelope_data.columns and 'As2Top' in self.envelope_data.columns:
            axes[0,2].scatter(self.envelope_data['As1Top'], self.envelope_data['As2Top'], alpha=0.6, s=20)
            axes[0,2].set_xlabel('As1 Top Reinforcement')
            axes[0,2].set_ylabel('As2 Top Reinforcement')
            axes[0,2].set_title('Top Reinforcement Directions')
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. Bottom Reinforcement by Direction
        if 'As1Bot' in self.envelope_data.columns and 'As2Bot' in self.envelope_data.columns:
            axes[1,0].scatter(self.envelope_data['As1Bot'], self.envelope_data['As2Bot'], alpha=0.6, s=20)
            axes[1,0].set_xlabel('As1 Bottom Reinforcement')
            axes[1,0].set_ylabel('As2 Bottom Reinforcement')
            axes[1,0].set_title('Bottom Reinforcement Directions')
            axes[1,0].grid(True, alpha=0.3)
        
        # 5. Element-wise Maximum Reinforcement
        element_subset = self.envelope_data.head(50)  # Show first 50 elements
        axes[1,1].bar(range(len(element_subset)), element_subset['MaxOverallRebar'])
        axes[1,1].set_xlabel('Element Index (first 50)')
        axes[1,1].set_ylabel('Max Reinforcement')
        axes[1,1].set_title('Max Reinforcement by Element')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Box plot comparison
        rebar_data = []
        labels = []
        for col in ['As1Top', 'As2Top', 'As1Bot', 'As2Bot']:
            if col in self.envelope_data.columns:
                rebar_data.append(self.envelope_data[col].dropna())
                labels.append(col)
        
        if rebar_data:
            axes[1,2].boxplot(rebar_data, labels=labels)
            axes[1,2].set_ylabel('Reinforcement')
            axes[1,2].set_title('Reinforcement Comparison')
            axes[1,2].grid(True, alpha=0.3)
            plt.setp(axes[1,2].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def export_results(self, output_path="sap2000_envelope_results.xlsx"):
        """
        Export envelope results to Excel
        """
        if self.envelope_data is None:
            print("❌ No envelope data to export.")
            return False
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main envelope data
                self.envelope_data.to_excel(writer, sheet_name='Envelope_Data', index=False)
                
                # Summary statistics
                stats = self.get_summary_statistics()
                if stats:
                    stats_df = pd.DataFrame(stats).T
                    stats_df.to_excel(writer, sheet_name='Summary_Statistics')
                
                # Critical elements
                critical = self.find_critical_elements(20)
                if critical is not None:
                    critical.to_excel(writer, sheet_name='Critical_Elements', index=False)
            
            print(f"✓ Results exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting results: {str(e)}")
            return False

# Example usage
def main():
    """
    Example of how to use the SAP2000ShellAnalyzer
    """
    # Initialize analyzer
    analyzer = SAP2000ShellAnalyzer("your_sap2000_file.xlsx")
    
    # Step 1: Read the data
    if analyzer.read_excel_data():
        
        # Step 2: Create reinforcement envelope
        envelope = analyzer.create_reinforcement_envelope()
        
        if envelope is not None:
            # Step 3: Display summary
            print("\n" + "="*60)
            print("REINFORCEMENT ENVELOPE SUMMARY")
            print("="*60)
            
            stats = analyzer.get_summary_statistics()
            if stats:
                for rebar_type, stat_dict in stats.items():
                    print(f"\n{rebar_type}:")
                    print(f"  Min: {stat_dict['min']:.3f}")
                    print(f"  Max: {stat_dict['max']:.3f}")
                    print(f"  Mean: {stat_dict['mean']:.3f}")
                    print(f"  Std: {stat_dict['std']:.3f}")
            
            # Step 4: Show critical elements
            print(f"\n" + "="*60)
            print("TOP 10 CRITICAL ELEMENTS")
            print("="*60)
            critical = analyzer.find_critical_elements(10)
            if critical is not None:
                print(critical.to_string(index=False))
            
            # Step 5: Create plots
            analyzer.plot_reinforcement_distribution()
            
            # Step 6: Export results
            analyzer.export_results("sap2000_envelope_analysis.xlsx")

if __name__ == "__main__":
    # To use this script:
    # 1. Replace "your_sap2000_file.xlsx" with your actual file path
    # 2. Run the script
    
    print("SAP2000 Shell Reinforcement Envelope Analyzer")
    print("=" * 50)
    print("Instructions:")
    print("1. Update the file path in the main() function")
    print("2. Run the script")
    print("3. The script will:")
    print("   - Read your SAP2000 Excel data")
    print("   - Create reinforcement envelopes")
    print("   - Find maximum and minimum values")
    print("   - Generate plots")
    print("   - Export results to Excel")
    print("\nTo run: python sap2000_analysis.py")
    
    # Uncomment the next line to run the analysis
    # main()