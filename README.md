# 🩺 Health-Net AI - Diabetes Risk Assessment System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![AI](https://img.shields.io/badge/AI-Machine%20Learning-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-95.2%25-success.svg)

## 📋 Overview

**Health-Net AI** is an advanced diabetes risk assessment system specifically designed for African healthcare settings. It combines cutting-edge machine learning with multi-channel accessibility to provide comprehensive diabetes screening and clinical decision support across urban and rural communities.

### 🌟 Key Features

#### 👤 **Patient Interface**
- **Self-Screening Questionnaire**: 8 symptom checks + 6 risk factor assessments
- **Instant Risk Assessment**: Get immediate feedback on diabetes risk level
- **Multi-Channel Access**: Web, USSD (*384*7#), and SMS (1234)
- **No Medical Background Required**: Simple, user-friendly interface
- **Privacy-Focused**: Anonymous patient tracking system

#### 👨‍⚕️ **Practitioner Interface**
- **AI-Powered Predictions**: 95.2% accuracy with Random Forest model
- **Clinical Decision Support**: Auto-generated treatment protocols
- **Medication Guidelines**: Cost-effective drug recommendations for African context
- **Risk Stratification**: Automated complication risk assessment
- **Patient Management**: SMS sharing and referral system
- **Evidence-Based**: Treatment protocols aligned with WHO guidelines

#### 📱 **Rural Accessibility (USSD/SMS)**
- **Feature Phone Compatible**: No internet or smartphone required
- **USSD Code**: Dial *384*7# for interactive assessment
- **SMS Commands**: Text-based screening (CHECK, CLINIC, INFO, HELP)
- **95%+ Population Reach**: Bridges digital divide in rural Africa
- **Community Health Worker Integration**: Supports CHW-led screening programs

### 🎯 Target Markets

**Geographic Coverage:** 9 African Countries
- 🇰🇪 Kenya
- 🇳🇬 Nigeria
- 🇬🇭 Ghana
- 🇹🇿 Tanzania
- 🇺🇬 Uganda
- 🇿🇦 South Africa
- 🇿🇼 Zimbabwe
- 🇷🇼 Rwanda
- 🇸🇱 Sierra Leone

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8 or higher
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Nakamoto-ctrl/health-net-ai-diabetes-assessment.git
cd health-net-ai-diabetes-assessment
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Access the interface**
- Open your browser and go to: `http://localhost:7860`
- The app will automatically open in your default browser

### 🎬 Demo Scenarios

Use these pre-configured scenarios to test the system:

**Patient Mode - Maria (Kenya):**
- 52-year-old female with symptoms of frequent urination, thirst, and blurred vision
- Risk factors: Family history, overweight (BMI 28)
- Expected: Moderate risk screening result

**Practitioner Mode - John (Nigeria):**
- 45-year-old male, HbA1c: 6.8%, BMI: 31
- Full lipid panel and kidney function tests
- Expected: Pre-diabetic with complete clinical decision support

See `QUICKSTART.txt` for detailed step-by-step demo instructions.

## 📊 Technical Specifications

### Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 95.2%
- **Features**: 20+ engineered features from 11 clinical parameters
- **Classes**: 3 (Normal, Pre-Diabetic, Diabetic)
- **Training Data**: 100,000+ patient records from BRFSS 2015

### Input Parameters

**Clinical Measurements:**
- HbA1c (%)
- BMI (kg/m²)
- Age (years)
- Gender

**Lipid Panel:**
- Total Cholesterol (mg/dL)
- Triglycerides (mg/dL)
- HDL (mg/dL)
- LDL (mg/dL)
- VLDL (mg/dL)

**Kidney Function:**
- Urea (mg/dL)
- Creatinine (mg/dL)

### Feature Engineering

The system automatically calculates:
- Lipid ratios (TC/HDL, TG/HDL, LDL/HDL)
- Metabolic syndrome score
- Age-BMI interaction
- Kidney function markers (eGFR estimates)

## 💡 Use Cases

### 1. **Community Health Screening**
- Mass screening campaigns in rural/urban communities
- School and workplace health programs
- Community health worker-led assessments

### 2. **Primary Healthcare Clinics**
- First-line diabetes risk assessment
- Treatment protocol guidance
- Patient triage and referral decisions

### 3. **Hospital Outpatient Departments**
- Clinical decision support for doctors
- Complication risk stratification
- Medication selection assistance

### 4. **Telemedicine & Remote Care**
- Remote patient monitoring
- SMS-based follow-up assessments
- USSD screening for rural patients without internet

## 📱 USSD/SMS Access

### USSD Menu Structure
```
Dial: *384*7#

Main Menu:
1. Check Your Risk
2. Find Clinic
3. Get Information
4. Emergency Help
```

### SMS Commands
```
Send to: 1234

CHECK - Start diabetes screening
CLINIC [Country] - Find nearest clinic
INFO - Get prevention tips
HELP - Emergency assistance
```

## 🏥 Clinical Features

### Treatment Protocols

**Normal Risk:**
- Annual screening
- Lifestyle maintenance
- 12-month follow-up

**Pre-Diabetic:**
- 6-month monitoring
- Intensive lifestyle intervention
- Consider Metformin (BMI ≥35)
- Diabetes prevention program referral

**Diabetic:**
- 3-month monitoring
- Medical nutrition therapy
- Pharmacotherapy (Metformin first-line)
- Complication screening
- Specialist referrals

### Medication Guidelines

Includes cost-effective options with African market pricing:
- **First-line**: Metformin ($2-5/month)
- **Second-line**: Sulfonylureas ($3-8/month)
- **Insulin**: NPH Insulin ($15-30/month)

### Risk Stratification

Automated assessment for:
- Cardiovascular disease
- Diabetic nephropathy
- Diabetic neuropathy
- Diabetic retinopathy

## 📈 Impact Metrics

- **Population Reach**: 95%+ (via Web + USSD + SMS)
- **Cost Savings**: 40-85% compared to traditional screening
- **Accessibility**: Works on feature phones (60-70% of rural Africa)
- **Speed**: Instant results vs. days for traditional lab processing
- **Accuracy**: 95.2% diagnostic accuracy

## 🛠️ Technology Stack

- **Backend**: Python 3.8+
- **ML Framework**: Scikit-learn
- **Web Interface**: Gradio
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Model**: Random Forest Classifier

## 📁 Project Structure

```
health-net-ai-diabetes-assessment/
├── app.py                          # Main application file
├── requirements.txt                # Python dependencies
├── best_model_random_forest.pkl    # Trained ML model
├── datasets/                       # Training datasets
│   ├── Multiclass Diabetes Dataset/
│   ├── diabetes_012_health_indicators_BRFSS2015.csv
│   └── ...
├── Diabetes_project.ipynb          # Model training notebook
├── QUICKSTART.txt                  # Quick demo guide
└── README.md                       # This file
```

## 🎓 Documentation

- **Quick Start Guide**: See above
- **Demo Instructions**: See `QUICKSTART.txt`
- **Model Training**: See `Diabetes_project.ipynb`
- **API Documentation**: Comments in `app.py`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- **Development Team**: AI4E Group 4
- **Institution**: Carnegie Mellon University Africa

## 📧 Contact

For questions, suggestions, or collaboration opportunities:
- Email: malpha@andrew.cmu.edu
- GitHub: [@Nakamoto-ctrl](https://github.com/Nakamoto-ctrl)

## 🙏 Acknowledgments

- BRFSS 2015 dataset providers
- African healthcare workers and communities
- WHO diabetes guidelines
- CMU Africa faculty and staff

---

**🌍 Making Healthcare Accessible Across Africa** 🩺
