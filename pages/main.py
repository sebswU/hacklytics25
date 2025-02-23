import streamlit as st
import pandas as pd
import numpy as np
import sklearn as sk

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, roc_curve

df = pd.read_csv('Nutritions_US.csv', encoding = "unicode-escape")
df.columns = df.columns.str.replace('[()]', '', regex=True).str.strip()
model = ''

def visualize(data: pd.DataFrame = df, indX: str = "Protein_g", depY = 'Calcium_mg'):
    """graph that depicts relationship represented by data"""
    st.scatter_chart(data, x=indX, y=depY)

@st.cache_resource
def fit(input_vals: list, output_vals: list):
    """"""
    features = [input_vals]
    targets = [output_vals]
    data=df[features + targets].dropna()

    # Split the data into training and testing sets
    X,y = data[features], data[targets]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor
    base_model = RandomForestRegressor(random_state=42, n_estimators=100)
    model = MultiOutputRegressor(base_model)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model for each target variable
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')  # Per target variable
    r2 = r2_score(y_test, y_pred, multioutput='variance_weighted')      # Overall R² score

    st.text(f"Mean Squared Error (MSE) per target: {mse}")
    st.text(f"Overall R² Score: {r2}")
    return model

@st.cache_resource
def predict(input_vals: list, targets: list, model):
    try:
        predicted_micronutrients = model.predict(input)
        st.text(f"Predicted micronutrients for input {input[0]}:")
        for micronutrient, value in zip(targets, predicted_micronutrients[0]):
            print(f"  {micronutrient}: {value:.2f}")
    except:
        st.text('this model did not perform properly')










st.title('Food Data Analysis')

chartx = st.radio("dep var", 
                  ['NDB_No', 'Shrt_Desc', 'Water_g', 'Energ_Kcal', 'Protein_g',
       'Lipid_Tot_g', 'Ash_g', 'Carbohydrt_g', 'Fiber_TD_g', 'Sugar_Tot_g',
       'Calcium_mg', 'Iron_mg', 'Magnesium_mg', 'Phosphorus_mg',
       'Potassium_mg', 'Sodium_mg', 'Zinc_mg', 'Copper_mg', 'Manganese_mg',
       'Selenium_¾g', 'Vit_C_mg', 'Thiamin_mg', 'Riboflavin_mg', 'Niacin_mg',
       'Panto_Acid_mg', 'Vit_B6_mg', 'Folate_Tot_¾g', 'Folic_Acid_¾g',
       'Food_Folate_¾g', 'Folate_DFE_¾g', 'Choline_Tot_ mg', 'Vit_B12_¾g',
       'Vit_A_IU', 'Vit_A_RAE', 'Retinol_¾g', 'Alpha_Carot_¾g',
       'Beta_Carot_¾g', 'Beta_Crypt_¾g', 'Lycopene_¾g', 'Lut+Zea_ ¾g',
       'Vit_E_mg', 'Vit_D_¾g', 'Vit_D_IU', 'Vit_K_¾g', 'FA_Sat_g', 'FA_Mono_g',
       'FA_Poly_g', 'Cholestrl_mg', 'GmWt_1', 'GmWt_Desc1', 'GmWt_2',
       'GmWt_Desc2'])

charty = chartx = st.radio("dep var", 
                  ['NDB_No', 'Shrt_Desc', 'Water_g', 'Energ_Kcal', 'Protein_g',
       'Lipid_Tot_g', 'Ash_g', 'Carbohydrt_g', 'Fiber_TD_g', 'Sugar_Tot_g',
       'Calcium_mg', 'Iron_mg', 'Magnesium_mg', 'Phosphorus_mg',
       'Potassium_mg', 'Sodium_mg', 'Zinc_mg', 'Copper_mg', 'Manganese_mg',
       'Selenium_¾g', 'Vit_C_mg', 'Thiamin_mg', 'Riboflavin_mg', 'Niacin_mg',
       'Panto_Acid_mg', 'Vit_B6_mg', 'Folate_Tot_¾g', 'Folic_Acid_¾g',
       'Food_Folate_¾g', 'Folate_DFE_¾g', 'Choline_Tot_ mg', 'Vit_B12_¾g',
       'Vit_A_IU', 'Vit_A_RAE', 'Retinol_¾g', 'Alpha_Carot_¾g',
       'Beta_Carot_¾g', 'Beta_Crypt_¾g', 'Lycopene_¾g', 'Lut+Zea_ ¾g',
       'Vit_E_mg', 'Vit_D_¾g', 'Vit_D_IU', 'Vit_K_¾g', 'FA_Sat_g', 'FA_Mono_g',
       'FA_Poly_g', 'Cholestrl_mg', 'GmWt_1', 'GmWt_Desc1', 'GmWt_2',
       'GmWt_Desc2'])
if st.button("see relationship"):
    try:
        visualize(chartx, charty)
    except:
        st.text("relationship didnt work out :pensive:")



options = st.multiselect(
    "Select Macronutrients and Micronutrients var",
    ['NDB_No', 'Shrt_Desc', 'Water_g', 'Energ_Kcal', 'Protein_g',
       'Lipid_Tot_g', 'Ash_g', 'Carbohydrt_g'],
    ['Fiber_TD_g', 'Sugar_Tot_g',
       'Calcium_mg', 'Iron_mg', 'Magnesium_mg', 'Phosphorus_mg',
       'Potassium_mg', 'Sodium_mg', 'Zinc_mg', 'Copper_mg', 'Manganese_mg',
       'Selenium_¾g', 'Vit_C_mg', 'Thiamin_mg', 'Riboflavin_mg', 'Niacin_mg',
       'Panto_Acid_mg', 'Vit_B6_mg', 'Folate_Tot_¾g', 'Folic_Acid_¾g',
       'Food_Folate_¾g', 'Folate_DFE_¾g', 'Choline_Tot_ mg', 'Vit_B12_¾g',
       'Vit_A_IU', 'Vit_A_RAE', 'Retinol_¾g', 'Alpha_Carot_¾g',
       'Beta_Carot_¾g', 'Beta_Crypt_¾g', 'Lycopene_¾g', 'Lut+Zea_ ¾g',
       'Vit_E_mg', 'Vit_D_¾g', 'Vit_D_IU', 'Vit_K_¾g', 'FA_Sat_g', 'FA_Mono_g',
       'FA_Poly_g', 'Cholestrl_mg', 'GmWt_1', 'GmWt_Desc1', 'GmWt_2',
       'GmWt_Desc2'],
)

num_values = len(options)



macro_values = []


for i in range(num_values):
    value = st.slider(f"Enter # in grams content of {options[i]}",
                      min_value=1, max_value=100, value=5, key=f"macro_{i}")
    macro_values.append(value)

if macro_values:
    input = np.array([[macro_values]])

else:
    st.write("set values what are you waiting for")

if st.button("predict"):
    try:
        model.predict(input, options[1], model)
    except:
        st.write("Can't predict anything yet")

    
