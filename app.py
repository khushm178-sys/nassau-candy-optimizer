import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Nassau Candy Optimizer', page_icon='🍬', layout='wide')
st.title('🍬 Nassau Candy — Factory Reallocation & Shipping Optimization')
st.markdown('---')

FACTORY_COORDS = {
    "Lot's O' Nuts"    : (32.881893, -111.768036),
    "Wicked Choccy's"  : (32.076176,  -81.088371),
    "Sugar Shack"      : (48.119140,  -96.181150),
    "Secret Factory"   : (41.446333,  -90.565487),
    "The Other Factory": (35.117500,  -89.971107)
}

REGION_COORDS = {
    "Interior" : (39.500000,  -98.350000),
    "Atlantic"  : (35.000000,  -78.000000),
    "Gulf"      : (29.760000,  -95.370000),
    "Pacific"   : (34.050000, -118.240000)
}

PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise"  : "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows"          : "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious"     : "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate"         : "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel"  : "Wicked Choccy's",
    "Laffy Taffy"                        : "Sugar Shack",
    "SweeTARTS"                          : "Sugar Shack",
    "Nerds"                              : "Sugar Shack",
    "Fun Dip"                            : "Sugar Shack",
    "Fizzy Lifting Drinks"               : "Sugar Shack",
    "Everlasting Gobstopper"             : "Secret Factory",
    "Hair Toffee"                        : "The Other Factory",
    "Lickable Wallpaper"                 : "Secret Factory",
    "Wonka Gum"                          : "Secret Factory",
    "Kazookles"                          : "The Other Factory"
}

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

@st.cache_data
def load_data():
    df = pd.read_csv('Nassau Candy Distributor (1).csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
    df['Lead Time']  = (df['Ship Date'] - df['Order Date']).dt.days
    df['Factory']    = df['Product Name'].map(PRODUCT_FACTORY)
    df['Profit Margin (%)'] = (df['Gross Profit'] / df['Sales'] * 100).round(2)
    df['Shipping Distance (miles)'] = df.apply(
        lambda r: round(haversine(*FACTORY_COORDS[r['Factory']],
                                  *REGION_COORDS[r['Region']]), 2)
        if pd.notna(r['Factory']) and r['Factory'] in FACTORY_COORDS else None, axis=1)
    return df

@st.cache_resource
def train_model(df):
    ml = df[['Factory','Region','Ship Mode','Division',
             'Shipping Distance (miles)','Sales','Cost',
             'Gross Profit','Units','Lead Time']].copy().dropna()
    le_dict = {}
    for col in ['Factory','Region','Ship Mode','Division']:
        le = LabelEncoder()
        ml[col] = le.fit_transform(ml[col])
        le_dict[col] = le
    X = ml.drop('Lead Time', axis=1)
    y = ml['Lead Time']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, scaler, le_dict, X.columns.tolist()

df = load_data()
model, scaler, le_dict, feature_cols = train_model(df)

import os
sim_results = pd.read_csv('simulation_results.csv') if os.path.exists('simulation_results.csv') else None
top_recs    = pd.read_csv('top_recommendations.csv') if os.path.exists('top_recommendations.csv') else None

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header('🔧 Filters')
sel_product  = st.sidebar.selectbox('Product',   ['All'] + sorted(df['Product Name'].unique().tolist()))
sel_region   = st.sidebar.selectbox('Region',    ['All'] + sorted(df['Region'].unique().tolist()))
sel_shipmode = st.sidebar.selectbox('Ship Mode', ['All'] + sorted(df['Ship Mode'].unique().tolist()))
priority     = st.sidebar.slider('Optimization Priority', 0, 100, 50, help='0=Speed  100=Profit')
st.sidebar.markdown(f'**Mode:** {"⚡ Speed" if priority < 40 else ("💰 Profit" if priority > 60 else "⚖️ Balanced")}')

# ── KPI Cards ─────────────────────────────────────────────────────────
st.subheader('📊 Platform KPIs')
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric('Total Orders',      f"{len(df):,}")
k2.metric('Avg Lead Time',     f"{df['Lead Time'].mean():.1f} days")
k3.metric('Total Sales',       f"${df['Sales'].sum():,.0f}")
k4.metric('Avg Profit Margin', f"{df['Profit Margin (%)'].mean():.1f}%")
k5.metric('Avg Distance',      f"{df['Shipping Distance (miles)'].mean():.0f} mi")
st.markdown('---')

# ── Module 1: Factory Simulator ───────────────────────────────────────
st.subheader('🏭 Module 1 — Factory Optimization Simulator')
col1, col2 = st.columns(2)
with col1:
    sim_product  = st.selectbox('Product',   sorted(df['Product Name'].unique()), key='s1')
    sim_region   = st.selectbox('Region',    sorted(df['Region'].unique()),       key='s2')
    sim_shipmode = st.selectbox('Ship Mode', sorted(df['Ship Mode'].unique()),    key='s3')
    run_sim = st.button('▶ Run Simulation')

with col2:
    if run_sim:
        division = df[df['Product Name']==sim_product]['Division'].iloc[0]
        sim_data = []
        for factory in FACTORY_COORDS:
            dist = haversine(*FACTORY_COORDS[factory], *REGION_COORDS[sim_region])
            f_enc = le_dict['Factory'].transform([factory])[0] if factory in le_dict['Factory'].classes_ else 0
            feats = [f_enc,
                     le_dict['Region'].transform([sim_region])[0],
                     le_dict['Ship Mode'].transform([sim_shipmode])[0],
                     le_dict['Division'].transform([division])[0],
                     dist,
                     df[df['Product Name']==sim_product]['Sales'].mean(),
                     df[df['Product Name']==sim_product]['Cost'].mean(),
                     df[df['Product Name']==sim_product]['Gross Profit'].mean(),
                     df[df['Product Name']==sim_product]['Units'].mean()]
            pred = model.predict(scaler.transform([feats]))[0]
            sim_data.append({'Factory': factory, 'Predicted Lead Time': round(pred,1),
                             'Distance (mi)': round(dist,0),
                             'Current': '✅' if factory==PRODUCT_FACTORY.get(sim_product,'') else ''})
        sim_df = pd.DataFrame(sim_data).sort_values('Predicted Lead Time')
        fig, ax = plt.subplots(figsize=(6,3))
        colors = ['#58C9A0' if r=='✅' else '#4A90D9' for r in sim_df['Current']]
        ax.barh(sim_df['Factory'], sim_df['Predicted Lead Time'], color=colors)
        ax.set_xlabel('Predicted Lead Time (days)')
        ax.set_title(f'{sim_product}')
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.dataframe(sim_df.reset_index(drop=True), use_container_width=True)

st.markdown('---')

# ── Module 2: What-If ─────────────────────────────────────────────────
st.subheader('🔄 Module 2 — What-If Scenario Analysis')
col3, col4 = st.columns(2)
with col3:
    wa_product  = st.selectbox('Product',              sorted(df['Product Name'].unique()), key='w1')
    wa_region   = st.selectbox('Region',               sorted(df['Region'].unique()),       key='w2')
    wa_shipmode = st.selectbox('Ship Mode',            sorted(df['Ship Mode'].unique()),    key='w3')
    wa_alt      = st.selectbox('Alternative Factory',  list(FACTORY_COORDS.keys()),         key='w4')

with col4:
    curr_factory = PRODUCT_FACTORY.get(wa_product, list(FACTORY_COORDS.keys())[0])
    division     = df[df['Product Name']==wa_product]['Division'].iloc[0]

    def pred_lt(factory):
        dist = haversine(*FACTORY_COORDS[factory], *REGION_COORDS[wa_region])
        f_enc = le_dict['Factory'].transform([factory])[0] if factory in le_dict['Factory'].classes_ else 0
        feats = [f_enc,
                 le_dict['Region'].transform([wa_region])[0],
                 le_dict['Ship Mode'].transform([wa_shipmode])[0],
                 le_dict['Division'].transform([division])[0],
                 dist,
                 df[df['Product Name']==wa_product]['Sales'].mean(),
                 df[df['Product Name']==wa_product]['Cost'].mean(),
                 df[df['Product Name']==wa_product]['Gross Profit'].mean(),
                 df[df['Product Name']==wa_product]['Units'].mean()]
        return round(model.predict(scaler.transform([feats]))[0], 1)

    curr_lt = pred_lt(curr_factory)
    alt_lt  = pred_lt(wa_alt)
    diff    = round(curr_lt - alt_lt, 1)
    st.metric('Current Factory',   curr_factory)
    st.metric('Current Lead Time', f'{curr_lt} days')
    st.metric('Alt Factory',       wa_alt)
    st.metric('Alt Lead Time',     f'{alt_lt} days')
    st.metric('Improvement',       f'{diff:+.1f} days', delta=f'{"Better ✅" if diff>0 else "Worse ❌"}')
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(['Current\n'+curr_factory[:12], 'Alt\n'+wa_alt[:12]],
           [curr_lt, alt_lt],
           color=['#E87D7D', '#58C9A0'])
    ax.set_ylabel('Lead Time (days)')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.markdown('---')

# ── Module 3: Recommendations ─────────────────────────────────────────
st.subheader('🏆 Module 3 — Recommendation Dashboard')
if top_recs is not None:
    filtered = top_recs.copy()
    if sel_product  != 'All': filtered = filtered[filtered['Product']   == sel_product]
    if sel_region   != 'All': filtered = filtered[filtered['Region']    == sel_region]
    if sel_shipmode != 'All': filtered = filtered[filtered['Ship Mode'] == sel_shipmode]
    if priority < 40:   filtered = filtered.sort_values('LT Reduction (%)', ascending=False)
    elif priority > 60: filtered = filtered.sort_values('Avg Gross Profit', ascending=False)
    else:
        filtered['Score'] = filtered['LT Reduction (%)'] * 0.5 + filtered.get('Avg Gross Profit', 0) * 0.5
        filtered = filtered.sort_values('Score', ascending=False)
    st.markdown(f'**{len(filtered)} recommendations**')
    cols = [c for c in ['Product','Current Factory','Alt Factory','Region',
                         'Ship Mode','Current Lead Time','Alt Lead Time','LT Reduction (%)']
            if c in filtered.columns]
    st.dataframe(filtered[cols].head(15).reset_index(drop=True), use_container_width=True)
    if len(filtered) > 0:
        top10 = filtered.head(10).copy()
        top10['Label'] = top10['Product'].str[:15] + ' → ' + top10['Alt Factory']
        fig, ax = plt.subplots(figsize=(9,4))
        sns.barplot(x=top10['LT Reduction (%)'], y=top10['Label'], palette='Greens_d', ax=ax)
        ax.set_xlabel('Lead Time Reduction (%)')
        ax.set_title('Top Reassignment Recommendations')
        plt.tight_layout()
        st.pyplot(fig); plt.close()
else:
    st.warning('simulation_results.csv not found — run Phase 4 in notebook first!')

st.markdown('---')

# ── Module 4: Risk Panel ──────────────────────────────────────────────
st.subheader('⚠️ Module 4 — Risk & Impact Panel')
col5, col6 = st.columns(2)
with col5:
    st.markdown('**Sales & Profit by Factory**')
    fp = df.groupby('Factory')[['Sales','Gross Profit']].sum()
    fig, ax = plt.subplots(figsize=(6,3))
    fp.plot(kind='bar', ax=ax, color=['#4A90D9','#58C9A0'], edgecolor='white')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col6:
    st.markdown('**Lead Time Heatmap — Factory × Region**')
    pivot = df.groupby(['Factory','Region'])['Lead Time'].mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(6,3))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

if sim_results is not None:
    high_risk = sim_results[sim_results['Alt Distance'] > 1500].sort_values('LT Reduction (%)', ascending=False).head(10)
    st.markdown('**High Risk Reassignments (Distance > 1500 miles)**')
    if len(high_risk) > 0:
        cols = [c for c in ['Product','Current Factory','Alt Factory','Region','Alt Distance','LT Reduction (%)'] if c in high_risk.columns]
        st.dataframe(high_risk[cols].reset_index(drop=True), use_container_width=True)
    else:
        st.success('No high risk reassignments found!')

st.markdown('---')
st.subheader('📋 Raw Data Explorer')
fdf = df.copy()
if sel_product  != 'All': fdf = fdf[fdf['Product Name'] == sel_product]
if sel_region   != 'All': fdf = fdf[fdf['Region']       == sel_region]
if sel_shipmode != 'All': fdf = fdf[fdf['Ship Mode']    == sel_shipmode]
st.dataframe(fdf[['Product Name','Factory','Region','Ship Mode','Lead Time',
                   'Shipping Distance (miles)','Sales','Gross Profit',
                   'Profit Margin (%)']].reset_index(drop=True), use_container_width=True)