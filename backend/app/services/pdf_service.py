
from fpdf import FPDF
import datetime
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FONTS_DIR = os.path.join(BASE_DIR, "assets", "fonts")

class PDF(FPDF):
    def header(self):
        # Logo could be added here
        if os.path.exists(os.path.join(FONTS_DIR, 'DejaVuSans-Bold.ttf')):
             self.set_font('DejaVu', 'B', 16)
        else:
             self.set_font('Arial', 'B', 16)
             
        self.cell(0, 10, 'CardioHealth Risk Report', 0, 1, 'C')
        
        if os.path.exists(os.path.join(FONTS_DIR, 'DejaVuSans-Oblique.ttf')):
             self.set_font('DejaVu', 'I', 10)
        else:
             self.set_font('Arial', 'I', 10)
             
        self.cell(0, 8, 'Personalized Cardiovascular Assessment', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        if os.path.exists(os.path.join(FONTS_DIR, 'DejaVuSans-Oblique.ttf')):
             self.set_font('DejaVu', 'I', 8)
        else:
             self.set_font('Arial', 'I', 8)
             
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'L')
        self.cell(0, 10, datetime.date.today().strftime("%B %d, %Y"), 0, 0, 'C')
        self.cell(0, 10, 'CardioHealth Risk Predictor Pro', 0, 0, 'R')

    def chapter_title(self, title):
        if os.path.exists(os.path.join(FONTS_DIR, 'DejaVuSans-Bold.ttf')):
             self.set_font('DejaVu', 'B', 14)
        else:
             self.set_font('Arial', 'B', 14)
             
        self.set_fill_color(230, 230, 250) # A light blue/lavender background
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        if os.path.exists(os.path.join(FONTS_DIR, 'DejaVuSans.ttf')):
             self.set_font('DejaVu', '', 11)
        else:
             self.set_font('Arial', '', 11)
             
        self.multi_cell(0, 6, body)
        self.ln()

    def result_summary(self, risk_level, risk_color, probability_text):
        if os.path.exists(os.path.join(FONTS_DIR, 'DejaVuSans-Bold.ttf')):
             self.set_font('DejaVu', 'B', 24)
        else:
             self.set_font('Arial', 'B', 24)
             
        if risk_color == 'red':
            self.set_text_color(220, 50, 50)
        elif risk_color == 'orange':
            self.set_text_color(255, 165, 0)
        else:
            self.set_text_color(34, 139, 34)
        
        self.cell(0, 12, f"Result: {risk_level}", 0, 1, 'C')
        
        if os.path.exists(os.path.join(FONTS_DIR, 'DejaVuSans.ttf')):
             self.set_font('DejaVu', '', 16)
        else:
             self.set_font('Arial', '', 16)
             
        self.set_text_color(0, 0, 0) # Reset color
        self.cell(0, 10, f"({probability_text})", 0, 1, 'C')
        self.ln(10)

def generate_pdf_report(user_input_report, rec_data):
    """
    Generates a downloadable PDF report from user inputs and prediction results.
    """
    pdf = PDF()
    
    # Add fonts if they exist
    dejavu_regular = os.path.join(FONTS_DIR, 'DejaVuSans.ttf')
    dejavu_bold = os.path.join(FONTS_DIR, 'DejaVuSans-Bold.ttf')
    dejavu_oblique = os.path.join(FONTS_DIR, 'DejaVuSans-Oblique.ttf')
    dejavu_bold_oblique = os.path.join(FONTS_DIR, 'DejaVuSans-BoldOblique.ttf')
    
    if os.path.exists(dejavu_regular):
        pdf.add_font('DejaVu', '', dejavu_regular, uni=True)
    if os.path.exists(dejavu_bold):
        pdf.add_font('DejaVu', 'B', dejavu_bold, uni=True)
    if os.path.exists(dejavu_oblique):
        pdf.add_font('DejaVu', 'I', dejavu_oblique, uni=True)
    if os.path.exists(dejavu_bold_oblique):
        pdf.add_font('DejaVu', 'BI', dejavu_bold_oblique, uni=True)
        
    pdf.add_page()
    
    # 1. Prediction Summary
    result_text = "Cardiovascular Disease" if "Cardiovascular Disease" in rec_data.get('risk_level', '') else "No Cardiovascular Disease"
    # Actually risk_level is mostly "High Risk", "Moderate Risk", etc.
    # But result_summary takes risk_level directly.
    
    pdf.result_summary(rec_data['risk_level'], rec_data['risk_color'], rec_data['probability'])

    # 2. Key Risk Factors
    pdf.chapter_title('Key Risk Factors Identified')
    pdf.chapter_body(rec_data.get('factors', ''))

    # 3. Personalized Recommendations
    pdf.chapter_title('Personalized Recommendations & Actions')
    pdf.chapter_body(rec_data.get('recommendations', ''))

    # 4. Screening & Monitoring Plan
    pdf.chapter_title('Screening & Monitoring Plan')
    pdf.chapter_body(rec_data.get('screening', ''))

    # 5. User Input Summary
    pdf.add_page()
    pdf.chapter_title('Patient Data Provided for this Assessment')
    
    # Create a simple table for user inputs
    if os.path.exists(dejavu_bold):
         pdf.set_font('DejaVu', 'B', 11)
    else:
         pdf.set_font('Arial', 'B', 11)

    col_width = pdf.w / 2.5
    row_height = 8

    # Reverse map the categorical values for readability
    cholesterol_rev_map = {1: "Normal", 2: "Above Normal", 3: "Well Above"}
    gluc_rev_map = {1: "Normal", 2: "Above Normal", 3: "Well Above"}
    smoke_rev_map = {0: "Non-smoker", 1: "Smoker"}
    alco_rev_map = {0: "Non-drinker", 1: "Moderate", 2: "Heavy"}
    active_rev_map = {0: "Sedentary", 1: "Moderately Active", 2: "Very Active"}
    stress_rev_map = {0: "Low", 1: "Moderate", 2: "High"}

    # Handle potentially missing keys gracefully
    def get_val(key, default='N/A'):
        return user_input_report.get(key, default)

    input_data_table = {
        "Patient Full Name": get_val('full_name'),
        "Patient Phone": get_val('phone_number'),
        "Age": f"{int(get_val('age', 0))} years",
        "Gender": get_val('gender'),
        "Height": f"{int(get_val('height', 0))} cm",
        "Weight": f"{get_val('weight', 0):.1f} kg",
        "Systolic BP": f"{int(get_val('ap_hi', 0))} mmHg",
        "Diastolic BP": f"{int(get_val('ap_lo', 0))} mmHg",
        "Cholesterol": cholesterol_rev_map.get(get_val('cholesterol'), "N/A"),
        "Glucose": gluc_rev_map.get(get_val('gluc', 1), "N/A"),
        "Smoker": smoke_rev_map.get(get_val('smoke'), "N/A"),
        "Alcohol Intake": alco_rev_map.get(get_val('alco', 0), "N/A"),
        "Activity Level": active_rev_map.get(get_val('active', 1), "N/A"),
        "Stress Level": stress_rev_map.get(get_val('stress', 1), "N/A"),
    }
    
    for key, value in input_data_table.items():
        if os.path.exists(dejavu_bold):
            pdf.set_font('DejaVu', 'B', 11)
        else:
            pdf.set_font('Arial', 'B', 11)
            
        pdf.cell(col_width, row_height, f"{key}:", border=1)
        
        if os.path.exists(dejavu_regular):
            pdf.set_font('DejaVu', '', 11)
        else:
            pdf.set_font('Arial', '', 11)
            
        pdf.cell(col_width, row_height, str(value), border=1)
        pdf.ln(row_height)
    
    pdf.ln(10)
    
    # Disclaimer
    if os.path.exists(dejavu_oblique):
        pdf.set_font('DejaVu', 'I', 9)
    else:
        pdf.set_font('Arial', 'I', 9)
        
    pdf.multi_cell(0, 5, "Disclaimer: This assessment is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns.")

    # Return the PDF data as bytes
    return bytes(pdf.output(dest='S'))
