import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Use basic matplotlib style to avoid seaborn conflicts
plt.style.use('default')

class SAP2000AnalyzerSimple:
    def __init__(self, file_path):
        """
        SAP2000 Shell Analyzer - Simplified version without seaborn dependencies
        """
        self.file_path = file_path
        self.raw_data = None
        self.data = None
        self.load_cases = []
        self.envelope_data = None
        self.table_title = ""
        self.units = {}
        
    def read_data(self, sheet_name=0):
        """Read SAP2000 data file"""
        try:
            print("🔍 Reading SAP2000 data file...")
            
            if self.file_path.lower().endswith('.csv'):
                raw_data = pd.read_csv(self.file_path, header=None)
            else:
                raw_data = pd.read_excel(self.file_path, sheet_name=sheet_name, header=None)
            
            self.table_title = raw_data.iloc[0, 0] if not pd.isna(raw_data.iloc[0, 0]) else "SAP2000 Shell Data"
            column_names = raw_data.iloc[1, :].values
            units = raw_data.iloc[2, :].values
            
            self.raw_data = raw_data.copy()
            
            data = raw_data.iloc[3:, :].copy()
            data.columns = column_names
            data = data.dropna(how='all').reset_index(drop=True)
            
            self.units = dict(zip(column_names, units))
            
            print(f"✅ Successfully loaded: {self.table_title}")
            print(f"📊 Raw data shape: {data.shape}")
            print(f"📋 Available columns: {list(data.columns)}")
            
            # Convert numeric columns
            numeric_columns = [
                'Area', 'As1Top', 'As2Top', 'As1Bot', 'As2Bot', 
                'Asw/s', 'Asw1/s', 'Asw2/s', 'TopLayThick', 'BotLayThick', 
                'JointThick', 'CoverTop1', 'CoverTop2', 'CoverBot1', 'CoverBot2', 
                'F11', 'F22', 'F12', 'M11', 'M22', 'M12', 'V13', 'V23'
            ]
            
            conversion_success = []
            for col in numeric_columns:
                if col in data.columns:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        conversion_success.append(col)
                    except:
                        pass
            
            print(f"🔢 Converted {len(conversion_success)} columns to numeric")
            
            if 'Area' in data.columns:
                original_length = len(data)
                data = data.dropna(subset=['Area'])
                removed_rows = original_length - len(data)
                if removed_rows > 0:
                    print(f"🧹 Removed {removed_rows} rows with missing Area values")
            
            self.data = data
            
            if 'OutputCase' in data.columns:
                self.load_cases = sorted(data['OutputCase'].unique().tolist())
                print(f"📂 Found {len(self.load_cases)} load cases: {self.load_cases}")
            
            if 'Area' in data.columns:
                unique_areas = sorted(data['Area'].unique())
                print(f"🏗️ Found {len(unique_areas)} unique elements: {unique_areas}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error reading file: {str(e)}")
            return False
    
    def create_reinforcement_envelope(self):
        """Create reinforcement envelope"""
        if self.data is None:
            print("❌ No data loaded. Please run read_data() first.")
            return None
        
        try:
            print("\n🎯 Creating REINFORCEMENT ENVELOPE...")
            print("=" * 50)
            
            required_rebar_cols = ['As1Top', 'As2Top', 'As1Bot', 'As2Bot']
            available_rebar_cols = [col for col in required_rebar_cols if col in self.data.columns]
            
            if len(available_rebar_cols) == 0:
                print("❌ No reinforcement columns found!")
                return None
            
            print(f"🔍 Available reinforcement columns: {available_rebar_cols}")
            
            envelope_agg = {'OutputCase': lambda x: ', '.join(x.astype(str))}
            
            info_cols = ['ElemType', 'ShellType', 'DesignType']
            for col in info_cols:
                if col in self.data.columns:
                    envelope_agg[col] = 'first'
            
            for col in available_rebar_cols:
                envelope_agg[col] = 'max'
            
            other_cols = ['TopLayThick', 'BotLayThick', 'JointThick']
            for col in other_cols:
                if col in self.data.columns:
                    envelope_agg[col] = 'first'
            
            envelope = self.data.groupby('Area').agg(envelope_agg).reset_index()
            
            if 'As1Top' in envelope.columns and 'As2Top' in envelope.columns:
                envelope['MaxTopRebar'] = envelope[['As1Top', 'As2Top']].max(axis=1)
                envelope['TopDirection'] = envelope[['As1Top', 'As2Top']].idxmax(axis=1)
            
            if 'As1Bot' in envelope.columns and 'As2Bot' in envelope.columns:
                envelope['MaxBotRebar'] = envelope[['As1Bot', 'As2Bot']].max(axis=1)
                envelope['BotDirection'] = envelope[['As1Bot', 'As2Bot']].idxmax(axis=1)
            
            if 'MaxTopRebar' in envelope.columns and 'MaxBotRebar' in envelope.columns:
                envelope['MaxOverallRebar'] = envelope[['MaxTopRebar', 'MaxBotRebar']].max(axis=1)
                envelope['GoverningLocation'] = envelope[['MaxTopRebar', 'MaxBotRebar']].idxmax(axis=1)
            
            if 'MaxTopRebar' in envelope.columns and 'MaxBotRebar' in envelope.columns:
                envelope['TopBottomRatio'] = envelope['MaxTopRebar'] / envelope['MaxBotRebar']
                envelope['ReinforcementClass'] = pd.cut(envelope['MaxOverallRebar'], 
                                                      bins=[0, 0.2, 0.5, 1.0, float('inf')], 
                                                      labels=['Light', 'Moderate', 'Heavy', 'Very Heavy'])
            
            self.envelope_data = envelope
            
            print(f"✅ Envelope created for {len(envelope)} elements")
            print(f"📊 Envelope columns: {list(envelope.columns)}")
            
            return envelope
            
        except Exception as e:
            print(f"❌ Error creating envelope: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_full_data(self):
        """Comprehensive analysis of all data"""
        if self.data is None:
            print("❌ No data loaded.")
            return None
        
        print("\n📈 COMPREHENSIVE DATA ANALYSIS")
        print("=" * 50)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        print(f"📊 Analyzing {len(numeric_cols)} numeric columns")
        
        if 'OutputCase' in self.data.columns:
            print(f"\n📂 LOAD CASE ANALYSIS:")
            for case in self.load_cases:
                case_data = self.data[self.data['OutputCase'] == case]
                print(f"   {case}: {len(case_data)} elements")
        
        if 'Area' in self.data.columns:
            print(f"\n🏗️ ELEMENT ANALYSIS:")
            element_counts = self.data['Area'].value_counts().sort_index()
            for area, count in element_counts.items():
                print(f"   Element {area}: {count} load case results")
        
        force_cols = [col for col in ['F11', 'F22', 'F12', 'M11', 'M22', 'M12', 'V13', 'V23'] 
                     if col in self.data.columns]
        
        if force_cols:
            print(f"\n⚡ FORCES/MOMENTS SUMMARY:")
            for col in force_cols[:3]:
                values = self.data[col].dropna()
                if len(values) > 0:
                    print(f"   {col}: Min={values.min():.1f}, Max={values.max():.1f}, Mean={values.mean():.1f}")
        
        return True
    
    def get_envelope_statistics(self):
        """Detailed statistics for reinforcement envelope"""
        if self.envelope_data is None:
            print("❌ No envelope data. Run create_reinforcement_envelope() first.")
            return None
        
        print("\n📊 ENVELOPE STATISTICS")
        print("=" * 50)
        
        stats = {}
        rebar_cols = ['As1Top', 'As2Top', 'As1Bot', 'As2Bot', 'MaxTopRebar', 'MaxBotRebar', 'MaxOverallRebar']
        available_cols = [col for col in rebar_cols if col in self.envelope_data.columns]
        
        for col in available_cols:
            values = self.envelope_data[col].dropna()
            stats[col] = {
                'min': values.min(),
                'max': values.max(),
                'mean': values.mean(),
                'std': values.std(),
                'count': len(values)
            }
            
            print(f"\n{col} ({self.units.get(col, 'units')}):")
            print(f"   Min:  {stats[col]['min']:.3f}")
            print(f"   Max:  {stats[col]['max']:.3f}")
            print(f"   Mean: {stats[col]['mean']:.3f}")
            print(f"   Std:  {stats[col]['std']:.3f}")
        
        return stats
    
    def find_critical_elements(self, top_n=5):
        """Find elements with highest reinforcement"""
        if self.envelope_data is None:
            print("❌ No envelope data available.")
            return None
        
        print(f"\n🔥 TOP {top_n} CRITICAL ELEMENTS")
        print("=" * 50)
        
        if 'MaxOverallRebar' not in self.envelope_data.columns:
            print("❌ MaxOverallRebar not available.")
            return None
        
        critical = self.envelope_data.nlargest(top_n, 'MaxOverallRebar')
        
        display_cols = ['Area', 'MaxTopRebar', 'MaxBotRebar', 'MaxOverallRebar', 
                       'TopDirection', 'BotDirection', 'GoverningLocation']
        available_display_cols = [col for col in display_cols if col in critical.columns]
        
        result = critical[available_display_cols]
        print(result.to_string(index=False, float_format='%.3f'))
        
        return result
    
    def create_simple_plots(self, figsize=(16, 10)):
        """Create simple, reliable plots using only matplotlib"""
        if self.envelope_data is None:
            print("❌ No envelope data for plotting.")
            return None
        
        print("\n🎨 CREATING VISUALIZATIONS...")
        print("=" * 50)
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            fig.suptitle(f'{self.table_title}\nReinforcement Envelope Analysis', 
                        fontsize=16, fontweight='bold')
            
            # Colors for consistency
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            
            # Plot 1: Top vs Bottom Reinforcement
            ax1 = axes[0, 0]
            if 'MaxTopRebar' in self.envelope_data.columns and 'MaxBotRebar' in self.envelope_data.columns:
                ax1.scatter(self.envelope_data['MaxTopRebar'], self.envelope_data['MaxBotRebar'], 
                           s=120, alpha=0.7, color=colors[0])
                ax1.set_xlabel('Max Top Reinforcement (mm²/mm)')
                ax1.set_ylabel('Max Bottom Reinforcement (mm²/mm)')
                ax1.set_title('Top vs Bottom Reinforcement', fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                # Add diagonal line
                max_val = max(self.envelope_data['MaxTopRebar'].max(), self.envelope_data['MaxBotRebar'].max())
                ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal reinforcement')
                ax1.legend()
                
                # Add element labels
                for i, row in self.envelope_data.iterrows():
                    ax1.annotate(f'E{int(row["Area"])}', 
                               (row['MaxTopRebar'], row['MaxBotRebar']),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            # Plot 2: Overall Reinforcement by Element
            ax2 = axes[0, 1]
            if 'MaxOverallRebar' in self.envelope_data.columns:
                bars = ax2.bar(self.envelope_data['Area'].astype(str), 
                              self.envelope_data['MaxOverallRebar'], 
                              color=colors[:len(self.envelope_data)])
                ax2.set_xlabel('Element Area')
                ax2.set_ylabel('Max Overall Reinforcement (mm²/mm)')
                ax2.set_title('Max Reinforcement by Element', fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, value in zip(bars, self.envelope_data['MaxOverallRebar']):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 3: Reinforcement Distribution
            ax3 = axes[0, 2]
            if 'MaxOverallRebar' in self.envelope_data.columns:
                ax3.hist(self.envelope_data['MaxOverallRebar'], bins=5, alpha=0.7, 
                        color=colors[1], edgecolor='black', linewidth=1.5)
                ax3.set_xlabel('Max Overall Reinforcement (mm²/mm)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Reinforcement Distribution', fontweight='bold')
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Top Reinforcement Directions
            ax4 = axes[1, 0]
            if 'As1Top' in self.envelope_data.columns and 'As2Top' in self.envelope_data.columns:
                ax4.scatter(self.envelope_data['As1Top'], self.envelope_data['As2Top'], 
                           s=120, alpha=0.7, color=colors[2])
                ax4.set_xlabel('As1 Top Reinforcement (mm²/mm)')
                ax4.set_ylabel('As2 Top Reinforcement (mm²/mm)')
                ax4.set_title('Top Reinforcement Directions', fontweight='bold')
                ax4.grid(True, alpha=0.3)
                
                # Add element labels
                for i, row in self.envelope_data.iterrows():
                    ax4.annotate(f'E{int(row["Area"])}', 
                               (row['As1Top'], row['As2Top']),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            # Plot 5: Bottom Reinforcement Directions
            ax5 = axes[1, 1]
            if 'As1Bot' in self.envelope_data.columns and 'As2Bot' in self.envelope_data.columns:
                ax5.scatter(self.envelope_data['As1Bot'], self.envelope_data['As2Bot'], 
                           s=120, alpha=0.7, color=colors[3])
                ax5.set_xlabel('As1 Bottom Reinforcement (mm²/mm)')
                ax5.set_ylabel('As2 Bottom Reinforcement (mm²/mm)')
                ax5.set_title('Bottom Reinforcement Directions', fontweight='bold')
                ax5.grid(True, alpha=0.3)
                
                # Add element labels
                for i, row in self.envelope_data.iterrows():
                    ax5.annotate(f'E{int(row["Area"])}', 
                               (row['As1Bot'], row['As2Bot']),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            # Plot 6: Load Case Contributions
            ax6 = axes[1, 2]
            if len(self.load_cases) > 1:
                case_max_values = {}
                for case in self.load_cases:
                    case_data = self.data[self.data['OutputCase'] == case]
                    if 'MaxOverallRebar' in self.envelope_data.columns:
                        # Get the maximum reinforcement for this load case
                        case_max = 0
                        for _, row in case_data.iterrows():
                            if 'As1Top' in case_data.columns and 'As2Top' in case_data.columns:
                                top_max = max(row.get('As1Top', 0), row.get('As2Top', 0))
                                bot_max = max(row.get('As1Bot', 0), row.get('As2Bot', 0))
                                overall_max = max(top_max, bot_max)
                                case_max = max(case_max, overall_max)
                        case_max_values[str(case)] = case_max
                
                if case_max_values and sum(case_max_values.values()) > 0:
                    wedges, texts, autotexts = ax6.pie(case_max_values.values(), 
                                                      labels=case_max_values.keys(), 
                                                      autopct='%1.1f%%', startangle=90,
                                                      colors=colors[:len(case_max_values)])
                    ax6.set_title('Load Case Max Contributions', fontweight='bold')
                else:
                    ax6.text(0.5, 0.5, 'Load Case\nData Not Available', 
                            ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            else:
                ax6.text(0.5, 0.5, 'Single Load Case\nNo Comparison', 
                        ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Visualizations created successfully!")
            return fig
            
        except Exception as e:
            print(f"❌ Error creating plots: {str(e)}")
            print("📊 Continuing with text-based analysis...")
            return None
    
    def export_results(self, output_path="SAP2000_Analysis_Results.xlsx"):
        """Export results to Excel"""
        if self.envelope_data is None:
            print("❌ No envelope data to export.")
            return False
        
        try:
            print(f"\n💾 EXPORTING RESULTS...")
            print("=" * 50)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Reinforcement Envelope
                self.envelope_data.to_excel(writer, sheet_name='Reinforcement_Envelope', index=False)
                print("✅ Reinforcement envelope exported")
                
                # Full Data
                self.data.to_excel(writer, sheet_name='Full_Data', index=False)
                print("✅ Full data exported")
                
                # Statistics
                stats = self.get_envelope_statistics()
                if stats:
                    stats_df = pd.DataFrame(stats).T
                    stats_df.to_excel(writer, sheet_name='Statistics')
                    print("✅ Statistics exported")
                
                # Critical Elements
                critical = self.find_critical_elements(10)
                if critical is not None:
                    critical.to_excel(writer, sheet_name='Critical_Elements', index=False)
                    print("✅ Critical elements exported")
            
            print(f"🎉 Results exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting: {str(e)}")
            return False
    
    def run_complete_analysis(self):
        """Run complete analysis workflow"""
        print("🚀 STARTING SAP2000 ANALYSIS")
        print("=" * 60)
        
        if not self.read_data():
            return False
        
        self.analyze_full_data()
        
        envelope = self.create_reinforcement_envelope()
        if envelope is None:
            return False
        
        self.get_envelope_statistics()
        self.find_critical_elements()
        self.create_simple_plots()
        self.export_results()
        
        print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return True

# Main execution
if __name__ == "__main__":
    analyzer = SAP2000AnalyzerSimple("sap2000_sample_data.csv")
    analyzer.run_complete_analysis()
