# Color Correction Test Pipeline

Code repository for "A Color Correction Pipeline for Images in Controlled Environments." Contains scripts for running color correction experiments, extracting validation metrics, aggregating results, and generating visualizations.

📦 **Full Dataset:** [HuggingFace](https://huggingface.co/datasets/CollinsW/ColorCorrectionPipeline_Dataset)

**This was a rough workflow to test different scenarios for the pipeline.** A more finished Python pipeline can be found at [ColorCorrectionPackage](https://github.com/collinswakholi/ColorCorrectionPackage).

## Quick Start

### Requirements
- **Python:** 3.8 or higher
- **OS:** Tested with Windows 11

### Installation
```bash
cd cc_test_pipeline
pip install -r requirements.txt
```

### Basic Workflow
1. **Configure** - Edit parameters at the top of each script
2. **Run Pipeline** - Execute `1.x_*.py` to apply color correction
3. **Extract Data and Metrics** - Run `5.0.x_*.py` to extract chart values metrics from already corrected images
4. **Aggregate Data** - Run `5.1.x_*.py` to consolidate results
5. **Visualize** - Execute `6.x_*.py` to generate plots and statistics

**Note:** Sample data included for demonstration. Download full dataset from HuggingFace for complete reproduction.

## Scripts Overview

### Core Components (Do NOT Change!)
- `ColorCorrection.py` - Main pipeline class
- `key_functions.py` - Utility functions
- `kwargs.py` - Configuration classes, Edit this file for configuration you'd want to run
- `FFC/FF_correction.py` - Flat-field correction
- `utils/` - Logging and metrics
- `best.pt` - YOLO model for chart detection

### Pipeline Execution (`1.x`)
| Script | Dataset | Applies |
|--------|---------|---------|
| `1.1_final_pipeline.py` | Light Temperatures | FFC → GC → WB → CC |
| `1.2_final_pipeline_LP.py` | Light Positions | FFC → GC → WB → CC |
| `1.3_final_pipeline_BG_Walls.py` | Backgrounds/Walls | FFC → GC → WB → CC |
| `1.4_final_pipeline_Sat.py` | Saturated Images | FFC → GC → WB → CC |

### Validation (`5.0.x`)
| Script | Dataset | Output |
|--------|---------|--------|
| `5.0_extract_validation.py` | Light Temperatures | ΔE validation metrics |
| `5.0.1_extract_validation_BG_Walls.py` | Backgrounds/Walls | ΔE validation metrics |
| `5.0.2_extract_validation_LP.py` | Light Positions | ΔE validation metrics |
| `5.0.2_extract_charts_LP.py` | Light Positions | Color charts + hex values |

### Data Aggregation (`5.1.x`)
Consolidates CSV results into Excel files for analysis.
- `5.1_extract_csv_data.py` - Light Temperatures
- `5.1.1_extract_csv_data_BG_Walls.py` - Backgrounds/Walls
- `5.1.2_extract_csv_data_LP.py` - Light Positions

### Visualization (`6.x`)
Generates raincloud plots and statistical analyses.
- `6_Graphing_Stats.py` - Main results (Light Temperatures)
- `6.1_Graph_Impact_GC_WB.py` - Comparative analysis (GC+WB impact)
- `6.2_Graphing_Stats_LP.py` - Light Positions
- `6.3_Graphing_Stats_BG_Walls.py` - Backgrounds/Walls

### Utilities
- `8.1_check_combo_charts.py` - Extracts data and metrics from multi-chart images
- `test_FFC.py` - Standalone FFC testing


## Execution
### Dataset-Specific Scripts
Each dataset follows the same 4-step workflow with dataset-specific scripts:

| Dataset | Step 1 | Step 2 | Step 3 | Step 4 |
|---------|--------|--------|--------|--------|
| **Light Temperatures** | `1.1_*.py` | `5.0_*.py` | `5.1_*.py` | `6_*.py` |
| **Light Positions** | `1.2_*_LP.py` | `5.0.2_*_LP.py` | `5.1.2_*_LP.py` | `6.2_*_LP.py` |
| **Backgrounds/Walls** | `1.3_*_BG_Walls.py` | `5.0.1_*_BG_Walls.py` | `5.1.1_*_BG_Walls.py` | `6.3_*_BG_Walls.py` |
| **Saturated Images** | `1.4_*_Sat.py` | — | — | — |

### Configuration
Edit configuration variables at the top of each script:
```python
# Example from 1.1_final_pipeline.py
DATA_FOLDER = 'Data' # data directory
TEST_GROUP = 'Light_Temperatures' # Data folder to run tests
FOCUS_ON = {'D50', 'D65', 'Amazon', 'Dalatin'}  # illuminants 1-4 
SEQUENCES = [[True, True, True, True, True]]  # FFC, Sat, GC, WB, CC flags. Whether or not to enable them
Degrees = [1, 2, 3, 4, 5]  # Polynomial degrees to test
color_method = 'conv'  # 'conv' or 'ours'. 'conv' is conventional, 'ours' is for all the other ML methods
```
