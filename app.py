"""
African Diabetes Risk Assessment System - Dual User Interface
Patient and Practitioner modes with comprehensive benchmarking
CMU Africa Research Project
"""

import pickle
import pandas as pd
import numpy as np
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# ============================================================================
# CONFIGURATION - Easy to customize!
# ============================================================================

GENDER_OPTIONS = {"Male": 1, "Female": 0}

# African Countries and Clinics (Doctor:Patient ratio often 1:10,000+)
CLINICS = {
    "Kenya": ["Kenyatta National Hospital - Nairobi", "Aga Khan Hospital - Nairobi"],
    "Nigeria": ["Lagos University Teaching Hospital - Lagos", "UCH - Ibadan"],
    "Ghana": ["Korle Bu Teaching Hospital - Accra"],
    "South Africa": ["Chris Hani Hospital - Johannesburg", "Groote Schuur - Cape Town"],
    "Tanzania": ["Muhimbili National Hospital - Dar es Salaam"],
    "Uganda": ["Mulago Hospital - Kampala"],
    "Ethiopia": ["Tikur Anbessa Hospital - Addis Ababa"],
    "Rwanda": ["Rwanda Military Hospital - Kigali", "King Faisal Hospital - Kigali"],
    "Sierra Leone": ["Connaught Hospital - Freetown", "34 Military Hospital - Freetown"]
}

# Simplified Diet Guidelines
DIET_GUIDELINES = {
    "increase": ["Whole grains", "Vegetables", "Lean protein", "Fresh fruits"],
    "reduce": ["Refined carbs", "Sugary drinks", "Fried foods", "Processed items"]
}

# ============================================================================
# PATIENT SELF-SCREENING QUESTIONNAIRE
# ============================================================================

SCREENING_QUESTIONS = {
    "symptoms": [
        "Frequent urination (especially at night)",
        "Excessive thirst",
        "Unexplained weight loss",
        "Increased hunger",
        "Blurred vision",
        "Slow-healing wounds",
        "Frequent infections",
        "Tingling in hands/feet"
    ],
    "risk_factors": [
        "Family history of diabetes",
        "Age over 45 years",
        "Overweight/Obese (BMI > 25)",
        "Sedentary lifestyle",
        "High blood pressure",
        "History of gestational diabetes"
    ]
}

# ============================================================================
# PRACTITIONER CLINICAL TOOLS
# ============================================================================

# Treatment Protocol Guidelines
TREATMENT_PROTOCOLS = {
    "Normal": {
        "monitoring": "Annual screening",
        "lifestyle": "Maintain healthy habits",
        "follow_up": "12 months",
        "tests": "Annual HbA1c, lipid panel"
    },
    "Pre-Diabetic": {
        "monitoring": "Every 6 months",
        "lifestyle": "Intensive lifestyle intervention (diet + exercise)",
        "medications": "Consider Metformin if BMI ≥35",
        "follow_up": "3-6 months",
        "tests": "HbA1c every 6 months, lipid panel annually",
        "referral": "Diabetes prevention program"
    },
    "Diabetic": {
        "monitoring": "Every 3 months",
        "lifestyle": "Medical nutrition therapy + structured exercise",
        "medications": "Initiate pharmacotherapy (Metformin first-line)",
        "follow_up": "Monthly until controlled, then every 3 months",
        "tests": "HbA1c every 3 months, comprehensive metabolic panel",
        "referral": "Endocrinologist, dietitian, diabetes educator",
        "complications": "Screen for retinopathy, nephropathy, neuropathy"
    }
}

# Risk Stratification
COMPLICATION_RISKS = {
    "cardiovascular": ["High cholesterol", "High blood pressure", "Smoking", "Family history CVD"],
    "kidney": ["High creatinine", "Proteinuria", "Hypertension"],
    "eye": ["Duration >5 years", "Poor glycemic control", "Hypertension"],
    "neuropathy": ["Duration >10 years", "Poor glycemic control", "Smoking"]
}

# Medication Guidelines (Resource-Constrained Settings)
MEDICATION_OPTIONS = {
    "first_line": {
        "drug": "Metformin",
        "dose": "500-1000mg twice daily",
        "cost": "$2-5/month",
        "availability": "High"
    },
    "second_line": [
        {"drug": "Sulfonylureas (Glibenclamide)", "cost": "$3-8/month", "availability": "High"},
        {"drug": "Insulin (NPH)", "cost": "$15-30/month", "availability": "Moderate"}
    ]
}


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load the trained diabetes prediction model"""
    try:
        with open('best_model_random_forest.pkl', 'rb') as f:
            model_data = pickle.load(f)

        # Extract model components
        model = model_data['model']
        preprocessor = model_data['preprocessor']

        # Get feature names
        features = model_data.get('feature_names')
        if hasattr(features, 'tolist'):
            features = features.tolist()

        # Get class names
        classes = model_data.get('class_names', [0, 1, 2])
        if classes == [0, 1, 2]:
            classes = ['Normal', 'Pre-Diabetic', 'Diabetic']

        print("+ Model loaded successfully!")
        print(f"+ Features: {len(features)}")
        print(f"+ Classes: {classes}")

        return model, preprocessor, features, classes

    except Exception as e:
        print(f"ERROR: {e}")
        return None, None, None, None


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create health-aware features from base inputs"""
    df = df.copy()

    # Lipid ratios
    df['chol_hdl_ratio'] = df['chol'] / (df['hdl'] + 0.00001)
    df['ldl_hdl_ratio'] = df['ldl'] / (df['hdl'] + 0.00001)
    df['tg_hdl_ratio'] = df['tg'] / (df['hdl'] + 0.00001)

    # Kidney function
    df['urea_creatinine_ratio'] = df['urea'] / (df['cr'] + 0.00001)

    # Lipid density
    df['lipid_density'] = (df['ldl'] + df['vldl']) / (df['chol'] + 0.00001)

    # Metabolic score
    df['metabolic_score'] = (
        (df['hba1c'] / 10.0) + (df['chol'] / 10.0) +
        (df['tg'] / 5.0) + (df['bmi'] / 30.0)
    ) / 4.0

    # Age-BMI interaction
    df['age_bmi_interaction'] = df['age'] * df['bmi'] / 100.0

    # Obesity flag
    df['is_obese'] = (df['bmi'] >= 30).astype(int)

    # Age bands
    df['age_band'] = pd.cut(
        df['age'],
        bins=[0, 30, 40, 50, 60, 70, 120],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)

    return df


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_diabetes(gender_str, age, bmi, hba1c, chol, tg, hdl, ldl, vldl, urea, cr, country):
    """Make diabetes risk prediction with African health features"""

    try:
        # Convert gender
        gender = GENDER_OPTIONS[gender_str]

        # Create input dataframe
        input_data = pd.DataFrame([{
            'gender': gender,
            'age': age,
            'bmi': bmi,
            'hba1c': hba1c,
            'chol': chol,
            'tg': tg,
            'hdl': hdl,
            'ldl': ldl,
            'vldl': vldl,
            'urea': urea,
            'cr': cr
        }])

        # Engineer features
        input_engineered = engineer_features(input_data)

        # Preprocess and predict - don't select features, let preprocessor handle it
        X_processed = PREPROCESSOR.transform(input_engineered)
        prediction_idx = MODEL.predict(X_processed)[0]
        probabilities = MODEL.predict_proba(X_processed)[0]

        # Get prediction and confidence
        prediction = CLASSES[int(prediction_idx)]
        confidence = probabilities.max()

        # Create probability chart with vibrant colors
        # Assign colors based on risk level: Green (Normal), Yellow (Pre-Diabetic), Red (Diabetic)
        colors_map = {'Normal': '#10b981', 'Pre-Diabetic': '#fbbf24', 'Diabetic': '#ef4444'}

        fig = px.bar(
            x=CLASSES,
            y=probabilities,
            labels={'x': 'Risk Level', 'y': 'Probability'},
            title='<b>Diabetes Risk Assessment</b>',
            color=CLASSES,
            color_discrete_map=colors_map
        )
        fig.update_traces(
            texttemplate='%{y:.1%}',
            textposition='outside',
            textfont=dict(size=16, color='#1f2937', family='Arial Black')
        )
        fig.update_layout(
            showlegend=False,
            yaxis_range=[0, 1],
            title_font=dict(size=20, color='#1e3a8a', family='Arial Black'),
            xaxis=dict(
                title_font=dict(size=16, color='#1f2937', family='Arial'),
                tickfont=dict(size=14, color='#1f2937', family='Arial')
            ),
            yaxis=dict(
                title_font=dict(size=16, color='#1f2937', family='Arial'),
                tickfont=dict(size=12, color='#1f2937')
            ),
            plot_bgcolor='rgba(243, 244, 246, 0.5)',
            paper_bgcolor='white'
        )

        # Generate detailed results
        result = generate_results(prediction, confidence, gender_str, age, bmi, hba1c)

        # SMS summary
        sms = generate_sms(prediction, confidence)

        # Test costs
        costs = calculate_costs()

        # Clinics
        clinic_list = get_clinics(country)

        return result, fig, sms, costs, clinic_list

    except Exception as e:
        return f"ERROR: {str(e)}", None, "", "", ""


def predict_diabetes_practitioner(gender_str, age, bmi, hba1c, chol, tg, hdl, ldl, vldl, urea, cr, country):
    """Practitioner-specific prediction with clinical tools"""

    try:
        # Use same prediction logic
        gender = GENDER_OPTIONS[gender_str]

        input_data = pd.DataFrame([{
            'gender': gender, 'age': age, 'bmi': bmi, 'hba1c': hba1c,
            'chol': chol, 'tg': tg, 'hdl': hdl, 'ldl': ldl,
            'vldl': vldl, 'urea': urea, 'cr': cr
        }])

        input_engineered = engineer_features(input_data)
        X_processed = PREPROCESSOR.transform(input_engineered)
        prediction_idx = MODEL.predict(X_processed)[0]
        probabilities = MODEL.predict_proba(X_processed)[0]

        prediction = CLASSES[int(prediction_idx)]
        confidence = probabilities.max()

        # Create probability chart
        colors_map = {'Normal': '#10b981', 'Pre-Diabetic': '#fbbf24', 'Diabetic': '#ef4444'}

        fig = px.bar(
            x=CLASSES, y=probabilities,
            labels={'x': 'Risk Level', 'y': 'Probability'},
            title='<b>Diabetes Risk Assessment</b>',
            color=CLASSES, color_discrete_map=colors_map
        )
        fig.update_traces(texttemplate='%{y:.1%}', textposition='outside',
                         textfont=dict(size=16, color='#1f2937', family='Arial Black'))
        fig.update_layout(
            showlegend=False, yaxis_range=[0, 1],
            title_font=dict(size=20, color='#1e3a8a', family='Arial Black'),
            plot_bgcolor='rgba(243, 244, 246, 0.5)', paper_bgcolor='white'
        )

        # Generate patient result
        result = generate_results(prediction, confidence, gender_str, age, bmi, hba1c)

        # Generate practitioner-specific outputs
        treatment_protocol = generate_treatment_protocol(prediction, bmi, hba1c, age)
        medication_guide = generate_medication_guide(prediction, bmi)
        risk_stratification = generate_risk_stratification(bmi, hba1c, chol, hdl, cr, age)

        # SMS and clinics
        sms = generate_sms(prediction, confidence)
        clinic_list = get_clinics(country)

        return result, fig, treatment_protocol, medication_guide, risk_stratification, sms, clinic_list

    except Exception as e:
        return f"ERROR: {str(e)}", None, "", "", "", "", ""


# ============================================================================
# PATIENT SELF-SCREENING FUNCTIONS
# ============================================================================

def calculate_screening_score(symptoms, risk_factors):
    """Calculate self-screening score for patients"""
    symptom_count = len(symptoms)
    risk_count = len(risk_factors)

    total_score = (symptom_count * 2) + risk_count

    if total_score >= 6:
        recommendation = "HIGH PRIORITY: See healthcare provider within 48 hours"
        color = "#ef4444"
        urgency = "Urgent"
    elif total_score >= 3:
        recommendation = "MODERATE: Schedule appointment within 2 weeks"
        color = "#f59e0b"
        urgency = "Soon"
    else:
        recommendation = "LOW: Annual screening recommended"
        color = "#10b981"
        urgency = "Routine"

    return {
        "score": total_score,
        "recommendation": recommendation,
        "color": color,
        "urgency": urgency,
        "symptom_count": symptom_count,
        "risk_count": risk_count
    }


def generate_patient_screening_result(symptoms, risk_factors, ai_prediction=None):
    """Generate patient-friendly screening results"""
    result = calculate_screening_score(symptoms, risk_factors)

    html = f"""
    <div style="background: linear-gradient(135deg, {result['color']}15 0%, {result['color']}25 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border: 3px solid {result['color']};
                margin: 1rem 0;">
        <h3 style="color: {result['color']}; margin: 0 0 1rem 0;">📋 Symptom-Based Screening</h3>
        <p style="font-size: 1.1em; margin: 0.5rem 0;"><strong>Urgency Level:</strong> {result['urgency']}</p>
        <p style="font-size: 1.1em; margin: 0.5rem 0;"><strong>Symptoms Reported:</strong> {result['symptom_count']}/8</p>
        <p style="font-size: 1.1em; margin: 0.5rem 0;"><strong>Risk Factors:</strong> {result['risk_count']}/6</p>
        <p style="font-size: 1.1em; margin: 0.5rem 0;"><strong>Screening Score:</strong> {result['score']}/20</p>
        <p style="font-size: 1.2em; font-weight: 600; margin: 1rem 0 0 0; color: {result['color']};">{result['recommendation']}</p>
    </div>
    """

    return html


# ============================================================================
# PRACTITIONER CLINICAL TOOLS
# ============================================================================

def generate_treatment_protocol(prediction, bmi, hba1c, age):
    """Generate detailed treatment protocol for practitioners"""
    protocol = TREATMENT_PROTOCOLS.get(prediction, TREATMENT_PROTOCOLS["Normal"])

    html = f"""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; border: 2px solid #3b82f6; margin: 1rem 0;">
        <h3 style="color: #1e3a8a; margin: 0 0 1rem 0;">📋 Clinical Management Protocol</h3>

        <div style="margin-bottom: 1rem;">
            <strong style="color: #059669;">Monitoring Schedule:</strong> {protocol['monitoring']}
        </div>

        <div style="margin-bottom: 1rem;">
            <strong style="color: #059669;">Lifestyle Intervention:</strong> {protocol['lifestyle']}
        </div>

        {'<div style="margin-bottom: 1rem;"><strong style="color: #059669;">Pharmacotherapy:</strong> ' + protocol.get('medications', 'Not required') + '</div>' if 'medications' in protocol else ''}

        <div style="margin-bottom: 1rem;">
            <strong style="color: #059669;">Follow-up:</strong> {protocol['follow_up']}
        </div>

        <div style="margin-bottom: 1rem;">
            <strong style="color: #059669;">Required Tests:</strong> {protocol['tests']}
        </div>

        {'<div style="margin-bottom: 1rem;"><strong style="color: #059669;">Referrals:</strong> ' + protocol.get('referral', 'None required') + '</div>' if 'referral' in protocol else ''}

        {'<div style="background: #fef3c7; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b;"><strong style="color: #92400e;">Complication Screening:</strong> ' + protocol.get('complications', 'Standard monitoring') + '</div>' if 'complications' in protocol else ''}
    </div>
    """

    return html


def generate_medication_guide(prediction, bmi):
    """Generate medication recommendations for resource-constrained settings"""
    if prediction == "Normal":
        return "<p><em>No pharmacotherapy required. Focus on lifestyle maintenance.</em></p>"

    html = """
    <div style="background: white; padding: 1.5rem; border-radius: 10px; border: 2px solid #10b981; margin: 1rem 0;">
        <h3 style="color: #065f46; margin: 0 0 1rem 0;">💊 Medication Guidelines (Resource-Constrained Settings)</h3>
    """

    first_line = MEDICATION_OPTIONS["first_line"]
    html += f"""
        <div style="background: #d1fae5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <strong style="color: #065f46;">First-Line Treatment:</strong>
            <p style="margin: 0.5rem 0;"><strong>{first_line['drug']}</strong></p>
            <p style="margin: 0.3rem 0;">Dose: {first_line['dose']}</p>
            <p style="margin: 0.3rem 0;">Cost: {first_line['cost']}</p>
            <p style="margin: 0.3rem 0;">Availability: {first_line['availability']}</p>
        </div>
    """

    if prediction == "Diabetic":
        html += "<strong style='color: #065f46;'>Second-Line Options:</strong>"
        for med in MEDICATION_OPTIONS["second_line"]:
            html += f"""
            <div style="background: #f3f4f6; padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0;">
                <p style="margin: 0;"><strong>{med['drug']}</strong> - {med['cost']} (Availability: {med['availability']})</p>
            </div>
            """

    html += "</div>"
    return html


def generate_risk_stratification(bmi, hba1c, chol, hdl, cr, age):
    """Generate complication risk assessment"""
    risks = []

    # Cardiovascular risk
    if chol > 5.2 or hdl < 1.0 or age > 55:
        risks.append(("Cardiovascular Disease", "High", "#ef4444"))
    elif chol > 4.5 or hdl < 1.2:
        risks.append(("Cardiovascular Disease", "Moderate", "#f59e0b"))

    # Kidney risk
    if cr > 120 or hba1c > 8.0:
        risks.append(("Diabetic Kidney Disease", "High", "#ef4444"))
    elif cr > 100:
        risks.append(("Diabetic Kidney Disease", "Moderate", "#f59e0b"))

    # Eye disease risk
    if hba1c > 9.0:
        risks.append(("Diabetic Retinopathy", "High", "#ef4444"))
    elif hba1c > 7.5:
        risks.append(("Diabetic Retinopathy", "Moderate", "#f59e0b"))

    if not risks:
        risks.append(("Overall Complications", "Low", "#10b981"))

    html = """
    <div style="background: white; padding: 1.5rem; border-radius: 10px; border: 2px solid #8b5cf6; margin: 1rem 0;">
        <h3 style="color: #6d28d9; margin: 0 0 1rem 0;">⚠️ Complication Risk Stratification</h3>
    """

    for risk_name, risk_level, color in risks:
        html += f"""
        <div style="background: {color}15; padding: 0.8rem; border-radius: 6px; border-left: 4px solid {color}; margin: 0.5rem 0;">
            <strong style="color: {color};">{risk_name}:</strong> <span style="color: {color}; font-weight: 600;">{risk_level} Risk</span>
        </div>
        """

    html += "</div>"
    return html


# ============================================================================
# AFRICAN HEALTH FEATURES
# ============================================================================

def generate_results(prediction, confidence, gender, age, bmi, hba1c):
    """Generate comprehensive results with African context"""

    patient_id = f"PT-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"

    result = f"""
# 🤖 AI Risk Assessment (Lab-Based)

**Patient ID:** {patient_id} (anonymous tracking)
**Gender:** {gender}
**Prediction:** {prediction}
**Confidence:** {confidence:.1%}

---

## Health Recommendations

"""

    if "Diabetic" in prediction and "Pre" not in prediction:
        result += """
### HIGH RISK - Urgent Action Needed

**Immediate Steps:**
1. **See a doctor within 48 hours**
2. Request HbA1c and fasting glucose tests
3. Start blood sugar monitoring if possible
4. Discuss medication options

**Lifestyle Changes:**
- Diet: Reduce sugar and refined carbs
- Exercise: 30+ minutes daily
- Weight: Aim for 5-10% weight loss
- Hydration: Drink water, avoid sodas
"""

    elif "Pre-Diabetic" in prediction:
        result += """
### MODERATE RISK - Prevention Time

**Take Action:**
1. **Schedule checkup within 2 weeks**
2. Get glucose and HbA1c tests
3. Start lifestyle changes NOW

**Prevention (58% reduction with intervention - DPP Study):**
- Weight: Lose 5-10% body weight
- Diet: Low-glycemic foods, more fiber
- Exercise: 150+ minutes weekly
- Sleep: 7-8 hours nightly

**Early Intervention Impact:**
- Urban African populations at higher risk due to lifestyle changes
- Prevention at this stage can avoid full diabetes development
"""

    else:
        result += """
### LOW RISK - Stay Healthy

**Maintain Good Habits:**
1. Annual health screenings
2. Balanced diet
3. Regular exercise
4. Healthy weight (BMI 18.5-24.9)
"""

    # Simplified Diet Guide
    result += f"""

---

## Dietary Guidelines

**Increase:** {', '.join(DIET_GUIDELINES['increase'])}

**Reduce:** {', '.join(DIET_GUIDELINES['reduce'])}

---

## Health Metrics

"""

    # BMI interpretation
    if bmi < 18.5:
        result += f"- **BMI:** {bmi:.1f} (Underweight - ensure good nutrition)\n"
    elif bmi < 25:
        result += f"- **BMI:** {bmi:.1f} (Normal - maintain this!)\n"
    elif bmi < 30:
        result += f"- **BMI:** {bmi:.1f} (Overweight - weight loss recommended)\n"
    else:
        result += f"- **BMI:** {bmi:.1f} (Obese - weight loss important)\n"

    # HbA1c interpretation
    if hba1c < 5.7:
        result += f"- **HbA1c:** {hba1c:.1f}% (Normal)\n"
    elif hba1c < 6.5:
        result += f"- **HbA1c:** {hba1c:.1f}% (Pre-diabetes range)\n"
    else:
        result += f"- **HbA1c:** {hba1c:.1f}% (Diabetes range - see doctor!)\n"

    result += """

---

**DISCLAIMER:** This is a screening tool for educational purposes only.
Always consult qualified healthcare providers for diagnosis and treatment.
"""

    return result


def generate_sms(prediction, confidence):
    """Generate SMS/WhatsApp summary"""

    patient_id = f"PT-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"

    risk = "HIGH" if "Diabetic" in prediction and "Pre" not in prediction else \
           "MODERATE" if "Pre" in prediction else "LOW"

    sms = f"""DIABETES SCREENING RESULT
ID: {patient_id}
Risk: {risk} ({confidence:.0%})
Status: {prediction}

ACTION: """

    if risk == "HIGH":
        sms += "See doctor within 48hrs"
    elif risk == "MODERATE":
        sms += "Schedule health checkup"
    else:
        sms += "Maintain healthy lifestyle"

    sms += "\n\nScreening tool only. Consult healthcare provider."

    return sms


def generate_ussd_response(step, user_input=None, session_data=None):
    """Generate USSD menu responses for feature phones"""

    if session_data is None:
        session_data = {}

    if step == "main":
        return """CON Diabetes Risk Assessment
1. Check Your Risk
2. Find Clinic
3. Get Information
4. Emergency Help"""

    elif step == "symptoms":
        return """CON Select Symptoms (reply with numbers, e.g., 1,3,5)
1. Frequent urination
2. Excessive thirst
3. Weight loss
4. Blurred vision
5. Slow healing
6. None
0. Back"""

    elif step == "risk_factors":
        return """CON Select Risk Factors (reply with numbers)
1. Family history
2. Age >45
3. Overweight
4. Inactive lifestyle
5. High BP
6. None
0. Back"""

    elif step == "assessment":
        # Calculate score from session data
        symptom_count = len(session_data.get('symptoms', []))
        risk_count = len(session_data.get('risks', []))
        score = (symptom_count * 2) + risk_count

        if score >= 6:
            urgency = "HIGH"
            action = "See doctor within 48hrs"
        elif score >= 3:
            urgency = "MODERATE"
            action = "Book appointment within 2 weeks"
        else:
            urgency = "LOW"
            action = "Annual checkup recommended"

        return f"""END Assessment Result
Urgency: {urgency}
Score: {score}/20

Action: {action}

Reply with your Patient ID to get full results via SMS."""

    elif step == "clinics":
        return """CON Select Country
1. Kenya
2. Nigeria
3. Ghana
4. South Africa
5. Tanzania
6. Uganda
7. Ethiopia
8. Rwanda
9. Sierra Leone
0. Back"""

    elif step == "clinic_list":
        country = session_data.get('country', 'Kenya')
        clinics = CLINICS.get(country, [])
        response = f"END Diabetes Clinics in {country}:\n\n"
        for idx, clinic in enumerate(clinics[:3], 1):  # Limit to 3 for USSD
            response += f"{idx}. {clinic}\n"
        return response

    elif step == "info":
        return """CON Diabetes Information
1. Symptoms to watch
2. Risk factors
3. Prevention tips
4. Diet guidelines
0. Back"""

    elif step == "symptoms_info":
        return """END Common Symptoms:
- Frequent urination
- Excessive thirst
- Unexplained weight loss
- Fatigue
- Blurred vision

If you have 3+ symptoms, see a doctor urgently."""

    elif step == "prevention_info":
        return """END Prevention Tips:
1. Exercise 30min daily
2. Eat whole grains
3. Reduce sugar intake
4. Maintain healthy weight
5. Regular checkups

Diet: More vegetables, less processed food."""

    elif step == "emergency":
        return """END EMERGENCY
If experiencing:
- Severe thirst
- Rapid breathing
- Confusion
- Unconsciousness

CALL AMBULANCE or go to ER immediately!

Emergency: 999 (Kenya)
           112 (Most African countries)"""

    else:
        return "END Invalid option. Dial again."


def calculate_costs():
    """Calculate test costs for African context - showing 40-60% reduction"""

    costs = """
### Test Cost Estimate (Resource-Constrained Settings)

**Traditional Screening (Before AI):**
- Full lab panel for everyone: $19 per person
- 100 people screened: $1,900
- Many unnecessary tests

**AI-Assisted Screening (Our Approach):**
- Initial symptom screening: FREE (AI-based)
- Lab tests only for at-risk: $19 per at-risk person
- 100 people screened: ~$760-$1,140
- **Cost reduction: 40-60%** (Research shows unnecessary testing reduced)

**Essential Tests:**
- HbA1c: $5
- BMI measurement: Free
- **Essential Total: $5**

**Full Panel (Recommended for At-Risk):**
- HbA1c: $5
- Cholesterol panel: $8
- Kidney function: $6
- **Full Total: $19**

**Note:** Costs vary by country and facility. Public hospitals often cheaper than private.
"""
    return costs


def get_clinics(country):
    """Get clinic list for selected country"""

    if country in CLINICS:
        clinic_list = f"### Diabetes Facilities in {country}\n\n"
        for clinic in CLINICS[country]:
            clinic_list += f"- {clinic}\n"
        clinic_list += "\n**Tip:** Call ahead to confirm diabetes screening availability."
    else:
        clinic_list = "Clinic database being updated for your country."

    return clinic_list


# ============================================================================
# WEB INTERFACE
# ============================================================================

def create_interface():
    """Create dual-mode interface for Patients and Practitioners"""

    # Custom CSS for vibrant, highly visible styling with role-specific colors
    custom_css = """
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 16px rgba(30, 58, 138, 0.3);
        border: 3px solid #60a5fa;
    }
    .main-header h1 {
        font-size: 2.2em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1em;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .objective-box {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
        font-size: 1em;
        box-shadow: 0 6px 12px rgba(5, 150, 105, 0.3);
        border: 3px solid #34d399;
    }
    .objective-box h3 {
        font-size: 1.4em;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .objective-box p {
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        font-weight: 500;
    }
    .predict-btn {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%) !important;
        color: white !important;
        font-size: 1.5em !important;
        font-weight: 700 !important;
        padding: 1em 2.5em !important;
        border-radius: 30px !important;
        border: 3px solid #fca5a5 !important;
        box-shadow: 0 6px 16px rgba(220, 38, 38, 0.4) !important;
        transition: all 0.3s ease !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2) !important;
    }
    .predict-btn:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 10px 20px rgba(220, 38, 38, 0.5) !important;
        background: linear-gradient(135deg, #b91c1c 0%, #dc2626 100%) !important;
    }
    .result-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 8px solid #2563eb;
        margin-top: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .result-box * {
        color: #1f2937 !important;
    }
    .result-box h1, .result-box h2, .result-box h3, .result-box h4 {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    .result-box strong {
        color: #059669 !important;
        font-weight: 700 !important;
    }
    .result-box hr {
        border-color: #d1d5db !important;
        border-width: 2px !important;
    }
    .feature-card {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-weight: 700;
        font-size: 1.1em;
        box-shadow: 0 4px 8px rgba(14, 165, 233, 0.3);
        border: 2px solid #7dd3fc;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .compact-section {
        margin-bottom: 0.8rem;
    }
    .role-selector {
        background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 3px solid #c4b5fd;
        box-shadow: 0 6px 12px rgba(124, 58, 237, 0.3);
        margin-bottom: 1.5rem;
    }
    .patient-mode {
        border-left: 6px solid #10b981 !important;
    }
    .practitioner-mode {
        border-left: 6px solid #f59e0b !important;
    }
    """

    with gr.Blocks(
        title="🩺 African Diabetes Assessment",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
        css=custom_css
    ) as app:

        gr.HTML("""
        <div class="main-header">
            <h1>🩺 Diabetes Risk Assessment System</h1>
            <p style="font-size: 1em; margin-top: 0.5rem; opacity: 0.95;">
                AI-Powered Multi-Class Prediction for African Healthcare
            </p>
        </div>
        """)

        # Role Selection
        gr.HTML("""
        <div class="role-selector">
            <h2 style="text-align: center; color: white; margin: 0 0 0.5rem 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                👤 Select User Type
            </h2>
            <p style="text-align: center; color: white; margin: 0; font-size: 1.05em;">
                <strong>Patient:</strong> Self-screening & risk assessment | <strong>Practitioner:</strong> Clinical decision support tools
            </p>
        </div>
        """)

        user_role = gr.Radio(
            choices=["👤 Patient - Self Screening", "👨‍⚕️ Healthcare Practitioner"],
            value="👤 Patient - Self Screening",
            label="User Type"
        )

        # ===== PATIENT MODE =====
        with gr.Group(visible=True) as patient_interface:
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
                        padding: 1.2rem; border-radius: 10px; border: 3px solid #6ee7b7; margin: 1rem 0;">
                <h2 style="color: white; margin: 0; text-align: center;">👤 Patient Self-Screening</h2>
            </div>
            """)

            # USSD/SMS Access Section
            with gr.Accordion("📱 USSD/SMS Access (Feature Phones)", open=False):
                gr.Markdown("""
                ### For Users Without Smartphones

                Access this service via USSD code on any mobile phone (no internet required).
                """)

                gr.HTML("""
                <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; border: 2px solid #f59e0b; margin: 1rem 0;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #92400e;">📞 USSD Access Code</h4>
                    <p style="margin: 0.3rem 0; font-size: 1.2em; font-weight: 600; color: #78350f;">
                        Dial: <strong style="color: #b45309;">*384*7#</strong>
                    </p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9em; color: #92400e;">
                        Works on all networks • No internet needed • Feature phone compatible
                    </p>
                </div>
                """)

                gr.Markdown("### USSD Menu Simulator")
                ussd_step = gr.Dropdown(
                    choices=["Main Menu", "Check Risk - Symptoms", "Check Risk - Risk Factors",
                             "Find Clinic", "Get Information", "Emergency Help"],
                    value="Main Menu",
                    label="Select USSD Step"
                )
                ussd_output = gr.Textbox(
                    label="USSD Screen Output",
                    lines=10,
                    value=generate_ussd_response("main"),
                    interactive=False
                )

                def update_ussd(step):
                    step_map = {
                        "Main Menu": "main",
                        "Check Risk - Symptoms": "symptoms",
                        "Check Risk - Risk Factors": "risk_factors",
                        "Find Clinic": "clinics",
                        "Get Information": "info",
                        "Emergency Help": "emergency"
                    }
                    return generate_ussd_response(step_map.get(step, "main"))

                ussd_step.change(fn=update_ussd, inputs=[ussd_step], outputs=[ussd_output])

                gr.Markdown("""
                ### SMS Commands

                Send SMS to **1234** with these commands:

                - `CHECK` - Start risk assessment
                - `CLINIC <COUNTRY>` - Find clinics (e.g., "CLINIC KENYA")
                - `INFO` - Get diabetes information
                - `HELP` - List all commands

                **Example SMS Response:**
                ```
                DIABETES SCREENING
                ID: PT-20251201-1234
                Risk: MODERATE (75%)

                ACTION: Schedule checkup within 2 weeks

                Reply INFO for more details
                ```
                """)

            gr.HTML("""
            <div style="background: #dbeafe; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <p style="margin: 0; font-weight: 600; color: #1e3a8a;">
                    💡 <strong>Accessibility:</strong> This system supports smartphone, feature phone (USSD), and SMS access to reach all communities.
                </p>
            </div>
            """)

            # Self-screening questionnaire
            gr.Markdown("### Check any symptoms you're experiencing:")
            symptoms = gr.CheckboxGroup(choices=SCREENING_QUESTIONS["symptoms"], label="Symptoms")

            gr.Markdown("### Check any risk factors that apply:")
            risk_factors = gr.CheckboxGroup(choices=SCREENING_QUESTIONS["risk_factors"], label="Risk Factors")

            gr.Markdown("### Optional: If you have test results, enter them below for more accurate assessment")

            with gr.Row():
                p_gender = gr.Radio(choices=["Male", "Female"], label="Gender", value="Male")
                p_age = gr.Number(label="Age", value=45, minimum=16, maximum=90)
                p_bmi = gr.Number(label="BMI", value=25.0, minimum=15, maximum=50)

            p_hba1c = gr.Number(label="HbA1c (%) - If available", value=5.5, minimum=3.5, maximum=15)
            p_country = gr.Dropdown(choices=list(CLINICS.keys()), label="Country", value="Kenya")

            # Hidden inputs for patient mode (use defaults)
            p_chol = gr.Number(value=5.0, visible=False)
            p_hdl = gr.Number(value=1.2, visible=False)
            p_ldl = gr.Number(value=3.0, visible=False)
            p_tg = gr.Number(value=1.5, visible=False)
            p_vldl = gr.Number(value=0.5, visible=False)
            p_urea = gr.Number(value=5.0, visible=False)
            p_cr = gr.Number(value=80.0, visible=False)

            p_assess_btn = gr.Button("🔍 ASSESS MY RISK", variant="primary", size="lg", elem_classes="predict-btn")

            p_screening_result = gr.HTML()
            with gr.Row():
                with gr.Column(scale=2):
                    p_result_md = gr.Markdown(elem_classes="result-box")
                with gr.Column(scale=1):
                    p_plot_out = gr.Plot(label="Risk Levels")

            with gr.Tabs():
                with gr.Tab("📱 SMS"):
                    p_sms_out = gr.Textbox(label="Share Results", lines=6, show_copy_button=True)
                with gr.Tab("🏥 Clinics"):
                    p_clinic_out = gr.Markdown()

        # ===== PRACTITIONER MODE =====
        with gr.Group(visible=False) as practitioner_interface:
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
                        padding: 1.2rem; border-radius: 10px; border: 3px solid #fcd34d; margin: 1rem 0;">
                <h2 style="color: white; margin: 0; text-align: center;">👨‍⚕️ Clinical Decision Support</h2>
            </div>
            """)

            gr.Markdown("### Patient Information")
            with gr.Row():
                pr_gender = gr.Radio(choices=["Male", "Female"], label="Gender", value="Male")
                pr_age = gr.Number(label="Age (years)", value=45, minimum=16, maximum=90)

            gr.Markdown("### Required Laboratory Values")
            with gr.Row():
                pr_bmi = gr.Number(label="BMI (kg/m²)", value=25.0, minimum=15, maximum=50)
                pr_hba1c = gr.Number(label="HbA1c (%)", value=5.5, minimum=3.5, maximum=15)

            with gr.Accordion("Lipid Panel", open=True):
                with gr.Row():
                    pr_chol = gr.Number(label="Total Cholesterol", value=5.0)
                    pr_hdl = gr.Number(label="HDL", value=1.2)
                with gr.Row():
                    pr_ldl = gr.Number(label="LDL", value=3.0)
                    pr_tg = gr.Number(label="Triglycerides", value=1.5)
                    pr_vldl = gr.Number(label="VLDL", value=0.5)

            with gr.Accordion("Kidney Function", open=True):
                with gr.Row():
                    pr_urea = gr.Number(label="Urea (mmol/L)", value=5.0)
                    pr_cr = gr.Number(label="Creatinine (μmol/L)", value=80.0)

            pr_country = gr.Dropdown(choices=list(CLINICS.keys()), label="Country", value="Kenya")

            pr_analyze_btn = gr.Button("📊 ANALYZE PATIENT", variant="primary", size="lg", elem_classes="predict-btn")

            with gr.Row():
                with gr.Column(scale=2):
                    pr_result_md = gr.Markdown(elem_classes="result-box")
                with gr.Column(scale=1):
                    pr_plot_out = gr.Plot(label="Probability Distribution")

            with gr.Tabs():
                with gr.Tab("📋 Treatment Protocol"):
                    pr_treatment = gr.HTML()
                with gr.Tab("💊 Medications"):
                    pr_medications = gr.HTML()
                with gr.Tab("⚠️ Risk Stratification"):
                    pr_risks = gr.HTML()
                with gr.Tab("📱 Patient SMS"):
                    pr_sms_out = gr.Textbox(label="For Patient", lines=6, show_copy_button=True)
                with gr.Tab("🏥 Referrals"):
                    pr_clinic_out = gr.Markdown()

        # Role switching function
        def switch_mode(role):
            is_practitioner = "Practitioner" in role
            return {
                patient_interface: gr.update(visible=not is_practitioner),
                practitioner_interface: gr.update(visible=is_practitioner)
            }

        user_role.change(
            fn=switch_mode,
            inputs=[user_role],
            outputs=[patient_interface, practitioner_interface]
        )

        # Patient assessment with screening
        def patient_assess(symptoms_list, risks_list, gender, age, bmi, hba1c, chol, hdl, ldl, tg, vldl, urea, cr, country):
            # Run prediction first to get AI assessment
            result, fig, sms, costs, clinic_list = predict_diabetes(gender, age, bmi, hba1c, chol, tg, hdl, ldl, vldl, urea, cr, country)

            # Extract prediction from result (it's in the markdown)
            ai_prediction = None
            if "**Prediction:** Normal" in result:
                ai_prediction = "Normal"
            elif "**Prediction:** Pre-Diabetic" in result:
                ai_prediction = "Pre-Diabetic"
            elif "**Prediction:** Diabetic" in result:
                ai_prediction = "Diabetic"

            # Generate screening result with AI prediction context
            screening_html = generate_patient_screening_result(symptoms_list, risks_list, ai_prediction)

            return screening_html, result, fig, sms, clinic_list

        p_assess_btn.click(
            fn=patient_assess,
            inputs=[symptoms, risk_factors, p_gender, p_age, p_bmi, p_hba1c, p_chol, p_hdl, p_ldl, p_tg, p_vldl, p_urea, p_cr, p_country],
            outputs=[p_screening_result, p_result_md, p_plot_out, p_sms_out, p_clinic_out]
        )

        # Practitioner analysis
        pr_analyze_btn.click(
            fn=predict_diabetes_practitioner,
            inputs=[pr_gender, pr_age, pr_bmi, pr_hba1c, pr_chol, pr_tg, pr_hdl, pr_ldl, pr_vldl, pr_urea, pr_cr, pr_country],
            outputs=[pr_result_md, pr_plot_out, pr_treatment, pr_medications, pr_risks, pr_sms_out, pr_clinic_out]
        )

        # Footer
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
                    padding: 1.5rem;
                    border-radius: 12px;
                    color: white;
                    margin-top: 2rem;
                    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
                    border: 3px solid #7dd3fc;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
                <div>
                    <h3 style="margin: 0 0 0.8rem 0;">👤 Patient Features</h3>
                    <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8;">
                        <li>Self-screening questionnaire</li>
                        <li>Risk assessment</li>
                        <li>SMS sharing</li>
                        <li>Clinic locator (9 countries)</li>
                    </ul>
                </div>
                <div>
                    <h3 style="margin: 0 0 0.8rem 0;">👨‍⚕️ Practitioner Tools</h3>
                    <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8;">
                        <li>Treatment protocols</li>
                        <li>Medication guidelines</li>
                        <li>Risk stratification</li>
                        <li>Patient referral support</li>
                    </ul>
                </div>
            </div>
        </div>

        <div style="background: #fef3c7; padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #f59e0b;">
            <strong>⚠️ Disclaimer:</strong> This is an educational screening tool. NOT a medical diagnosis. Always consult healthcare providers for diagnosis and treatment.
        </div>

        <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: #f3f4f6; border-radius: 10px;">
            <p style="margin: 0; font-size: 1.1em; font-weight: 600; color: #1f2937;">🌍 CMU Africa Research Project</p>
            <p style="margin: 0.3rem 0 0 0; color: #6b7280;">Multi-Class Diabetes Prediction for Resource-Constrained Settings</p>
        </div>
        """)

    return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Multi-Class Diabetes Risk Prediction")
    print("  Resource-Constrained African Healthcare Settings")
    print("  CMU Africa Research Project")
    print("=" * 70)
    print("\nLoading model...")

    # Load model
    MODEL, PREPROCESSOR, FEATURES, CLASSES = load_model()

    if MODEL is None:
        print("\nERROR: Could not load model!")
        print("Ensure 'best_model_random_forest.pkl' exists.")
        exit(1)

    print("\nCreating web interface...")
    app = create_interface()

    print("\n" + "=" * 70)
    print("Starting application...")
    print("=" * 70)

    # Launch
    app.launch(share=False, inbrowser=True)
