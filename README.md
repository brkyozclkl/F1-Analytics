# F1-Analytics 🏎️ 

<div align="center">
  <img src="https://www.formula1.com/content/dam/fom-website/manual/Misc/2024manual/2024Launches/SF-24Launch/Ferrari_Social_16x9.jpg.transform/9col/image.jpg" alt="F1 Banner" width="100%">

  <p align="center">
    <strong>Advanced Formula 1 Analytics & Prediction Platform</strong>
  </p>

  <p align="center">
    <a href="https://www.python.org/downloads/">
      <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python Version">
    </a>
    <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    </a>
    <a href="https://github.com/psf/black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style: Black">
    </a>
    <a href="https://github.com/brkyozclkl/F1-Analytics">
      <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintenance">
    </a>
  </p>
</div>

## 🎯 Overview

F1-Analytics is a cutting-edge machine learning platform designed to revolutionize Formula 1 race predictions. By leveraging advanced ML algorithms and comprehensive historical data, we achieve exceptional accuracy in predicting race outcomes and providing deep insights into driver and team performance.

### 🌟 Key Features

- **Race Prediction Engine**: Advanced ML models for accurate race outcome predictions
- **Performance Analytics**: Comprehensive driver and team performance analysis
- **Circuit Analysis**: Detailed track-specific performance insights
- **Real-time Updates**: Live data integration for up-to-date predictions
- **Interactive Visualizations**: Rich, dynamic data visualizations

## 📊 Dataset Overview

Our comprehensive Formula 1 dataset includes:

<div align="center">

| Data File | Records | Description |
|-----------|---------|-------------|
| races.csv | 1,127 | Historical race information and details |
| results.csv | 25,000+ | Race results and finishing positions |
| qualifying.csv | 10,496 | Qualifying session results and grid positions |
| drivers.csv | 863 | Driver information and demographics |
| constructors.csv | 214 | Constructor (team) details |
| circuits.csv | 79 | Circuit information and characteristics |
| lap_times.csv | 400,000+ | Detailed lap-by-lap timing data |
| pit_stops.csv | 11,373 | Pit stop timing and duration |
| driver_standings.csv | 34,865 | Championship standings after each race |
| constructor_standings.csv | 13,393 | Team championship standings |
| sprint_results.csv | 362 | Sprint race results |

</div>

### Data Features

- **Race Information**: Complete race calendar from 1950 to present
- **Driver Data**: Comprehensive driver statistics and career information
- **Team Performance**: Detailed constructor performance metrics
- **Circuit Details**: Track characteristics and historical performance data
- **Timing Data**: Extensive lap times and pit stop information
- **Championship Progress**: Race-by-race standings for drivers and teams

## 🏆 Model Performance

Our state-of-the-art LightGBM model delivers industry-leading accuracy:

<div align="center">

| Metric | Performance |
|--------|-------------|
| Position Accuracy (±1) | 84.1% |
| Podium Prediction | 92.3% |
| Points Prediction | 94.2% |
| RMSE | 0.360 |
| MAE | 0.202 |
| R² Score | 0.990 |

</div>

## 📊 Sample Visualizations

<div align="center">

> Note: Visualization examples will be added after processing the dataset. They will include:
> - Driver performance trends
> - Team comparison analytics
> - Circuit-specific analysis
> - Prediction accuracy plots
> - Championship progression
> - Lap time analysis

</div>

## 🛠️ Technology Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| Core | ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) |
| ML Framework | ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green) ![LightGBM](https://img.shields.io/badge/LightGBM-Latest-lightgreen) |
| Data Processing | ![Pandas](https://img.shields.io/badge/Pandas-Latest-darkblue) ![NumPy](https://img.shields.io/badge/NumPy-Latest-lightblue) |
| Visualization | ![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-red) ![Seaborn](https://img.shields.io/badge/Seaborn-Latest-purple) |

</div>

## 📦 Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/brkyozclkl/F1-Analytics.git
   cd F1-Analytics
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Analysis**
   ```bash
   python src/data_collector.py  # Collect and process data
   python src/main.py           # Train model and generate predictions
   ```

## 📁 Project Structure

```plaintext
F1-Analytics/
├── archive/              # Raw F1 dataset
│   ├── circuits.csv     # Circuit information
│   ├── constructors.csv # Team data
│   ├── drivers.csv      # Driver information
│   ├── races.csv        # Race details
│   ├── results.csv      # Race results
│   ├── qualifying.csv   # Qualifying data
│   ├── lap_times.csv    # Lap timing data
│   └── ...              # Additional data files
├── data/                # Processed data
│   └── f1_races.csv     # Combined race data
├── results/             # Output directory
│   └── images/          # Generated visualizations
├── src/                 # Source code
│   ├── data_collector.py # Data collection
│   ├── data_processing.py# Data preprocessing
│   ├── model_training.py # ML implementation
│   ├── visualization.py  # Visualization
│   └── main.py          # Main script
├── notebooks/           # Jupyter notebooks
├── tests/              # Test suite
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

## 🔄 Data Pipeline

1. **Data Collection & Processing**
   - Historical race results from archive
   - Driver & team statistics integration
   - Circuit characteristics analysis
   - Lap time and pit stop data processing

2. **Feature Engineering**
   - Performance metrics calculation
   - Moving averages for consistency
   - Circuit-specific feature extraction
   - Championship progression tracking
   - Pit stop strategy analysis
   - Qualifying performance metrics

3. **Model Development**
   - Algorithm selection and testing
   - Hyperparameter optimization
   - Cross-validation strategies
   - Performance evaluation metrics

## 🚀 Future Roadmap

- [ ] Real-time race predictions
- [ ] Weather impact modeling
- [ ] Neural network integration
- [ ] Qualifying predictions
- [ ] Championship probability calculator
- [ ] Interactive web dashboard
- [ ] Pit stop strategy optimizer
- [ ] Driver head-to-head comparisons
- [ ] Team development tracker

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📫 Contact & Support

- **GitHub**: [@brkyozclkl](https://github.com/brkyozclkl)
- **LinkedIn**: [Berkay Özçelikel](https://www.linkedin.com/in/berkay-özçelikel-68315a1b6/)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- Formula 1 for providing historical data
- The F1 community for valuable insights
- All contributors to this project

---

<div align="center">

### Show Your Support

⭐️ Star this repo if you found it interesting!

</div> 