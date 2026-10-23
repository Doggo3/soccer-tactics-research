from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from xgboost import XGBClassifier
P=Path(__file__).parent
st.set_page_config(page_title='Soccer Attacking Phase Classification',page_icon='⚽',layout='wide')
st.markdown("""<style>
.stApp {background:#fff;color:#17251c;}
.block-container{max-width:1150px;padding-top:2rem;}
h1{font-size:2.5rem!important;} h2,h3{color:#17251c!important;}
p,li,[data-testid="stMarkdownContainer"] p{font-size:18px!important;line-height:1.65;}
[data-testid="stCaptionContainer"] p{font-size:15px!important;color:#45574b!important;}
[data-testid="stMetric"]{background:#eef4ef;border:1px solid #c7d5ca;border-radius:8px;padding:18px;}
[data-baseweb="tab"]{font-size:18px!important;min-height:56px;padding:12px 22px!important;}
[data-testid="stForm"]{border:2px solid #b5cabc;padding:20px;border-radius:8px;}
button[kind="primaryFormSubmit"]{min-height:48px;font-size:18px!important;}
</style>""",unsafe_allow_html=True)
@st.cache_resource
def load():
 m=XGBClassifier();m.load_model(P/'model.ubj')
 return m,json.loads((P/'metadata.json').read_text()),pd.read_csv(P/'examples.csv')
m,meta,examples=load();features=meta['features'];classes=meta['classes']
st.caption('Jack Chiang · Soccer tactics research')
st.title('Classifying Soccer Attacking Phases')
st.write('I tested whether player and ball tracking data could identify five types of attack. This app lets you try the model on real phases and see how changing the inputs affects its prediction.')
a,b,c,d=st.columns(4)
a.metric('Attacking phases','2,955');b.metric('Matches','10');c.metric('App model accuracy',f"{meta['accuracy']:.1%}");d.metric('App model macro F1',f"{meta['macro_f1']:.3f}")
st.caption('The app uses a retrained XGBoost model. Results reported in the paper are shown in the Research tab.')
overview,predict,batch,research=st.tabs(['Start here','Try the model','Upload a CSV','Research'])
with overview:
 st.subheader('What does this project do?')
 st.write('A soccer attack can develop in several ways. I trained machine-learning models to recognize five types of attacking phase from measurements of the players and ball.')
 st.write('An attacking phase is a short section of play. The model receives 18 measurements about that section, such as where the ball starts, how far it moves, and how spread out the teams are. It returns one of the five labels below.')
 st.table(pd.DataFrame({'Type of attack':['Build-up','Creation','Direct attack','Fast transition','Finishing'],'Meaning':['Building the attack from farther away from the opponent’s goal.','Working to create a scoring opportunity.','Moving the ball forward quickly.','Attacking soon after winning possession.','The final part of the attack near the opponent’s goal.']}))
 st.subheader('What I found')
 st.write('In my paper, Enhanced XGBoost classified 91.9% of test phases correctly—about 92 out of 100. Enhanced Random Forest scored best when all five tactics were weighted equally. Fast transitions were hardest to identify, partly because there were only 132 examples in the dataset.')
 st.write('The study used 2,955 phases from 10 Australian A-League matches. Eight matches were used to train the models, and two different matches were used to test them.')
 st.info('To try it: open Try the model, pick an example, and look at the prediction. You can change the measurements and press Predict tactic to see what happens. No upload is needed.')
 st.caption('The app uses numerical tracking features, not video. Its labels come from SkillCorner’s existing system. It identifies the type of attack; it does not choose a tactic for a coach.')
with predict:
 st.subheader('Try a phase from a test match')
 st.write('**1. Choose a tactic and sample below.** The prediction appears underneath.\n\n**2. Open the inputs to change the phase.** Press **Predict tactic** to see the updated result.')
 label=st.selectbox('Choose a tactic to explore',classes,format_func=lambda v:v.replace('_',' ').capitalize())
 descriptions={'build_up':'A controlled attack, usually starting farther from the opposing goal.','creation':'Developing an attacking opportunity.','direct_attack':'Moving the ball forward quickly.','fast_transition':'Attacking soon after winning possession.','finishing':'The final part of an attack near the opposing goal.'}
 st.caption(descriptions[label])
 subset=examples[examples.tactic_label==label]
 idx=st.selectbox('Sample phase',subset.index.tolist(),format_func=lambda i:f"Match {int(examples.loc[i,'match_id'])} · phase {int(examples.loc[i,'phase_index'])} · minute {examples.loc[i,'match_minute']:.1f}")
 row=examples.loc[[idx],features].copy()
 with st.form('phase'):
  with st.expander('Change the inputs — click to open'):
   st.caption('Distances are in meters; shares range from 0 to 1. Progress and progress speed retain the signed convention in the supplied dataset.')
   columns=st.columns(3)
   vals={}
   for i,f in enumerate(features):
    vals[f]=columns[i%3].number_input(f.replace('_',' ').capitalize(),value=float(row.iloc[0][f]),format='%.4f',key=f'{idx}_{f}')
  clicked=st.form_submit_button('Predict tactic',type='primary')
 x=pd.DataFrame([vals],columns=features)
 if not np.isfinite(x.to_numpy()).all(): st.error('All inputs must be finite numbers.')
 else:
  probs=m.predict_proba(x)[0]; winner=classes[int(probs.argmax())]
  left,right=st.columns([1,2])
  left.caption('MODEL PREDICTION');left.subheader(winner.replace('_',' ').title());left.metric('Model probability',f'{probs.max():.1%}')
  left.caption(f"Reference label for the original sample: {label}. Changing inputs creates a hypothetical phase.")
  right.plotly_chart(px.bar(pd.DataFrame({'Tactic':[v.replace('_',' ').capitalize() for v in classes],'Probability':probs}),x='Probability',y='Tactic',orientation='h',range_x=[0,1],color_discrete_sequence=['#285943']).update_layout(template='plotly_white',font=dict(size=16,color='#17251c'),paper_bgcolor='white',plot_bgcolor='white',height=370,margin=dict(l=10,r=20,t=10,b=30)).update_xaxes(tickformat='.0%'),width='stretch')
  st.caption('These scores show how the model ranks the five labels. They do not tell you which tactic a team should use.')
with batch:
 st.subheader('Classify your own phase features')
 st.write('Upload a CSV containing the same 18 feature columns. Raw video and tracking coordinates must first be processed into phase features.')
 st.download_button('Download example CSV',examples[features].head(10).to_csv(index=False),'example_phases.csv','text/csv')
 upload=st.file_uploader('Phase features CSV',type=['csv'])
 if upload:
  try:
   data=pd.read_csv(upload)
   missing=[f for f in features if f not in data]
   if missing: st.error('Missing columns: '+', '.join(missing))
   elif not 0<len(data)<=10000: st.error('Upload between 1 and 10,000 rows.')
   else:
    x=data[features].apply(pd.to_numeric,errors='raise')
    if not np.isfinite(x.to_numpy()).all(): st.error('Remove missing or infinite feature values.')
    else:
     probs=m.predict_proba(x);out=data.copy();out['predicted_tactic']=[classes[i] for i in probs.argmax(axis=1)];out['model_probability']=probs.max(axis=1)
     st.dataframe(out,width='stretch');st.download_button('Download predictions',out.to_csv(index=False),'tactic_predictions.csv','text/csv')
  except (ValueError,TypeError,pd.errors.ParserError,UnicodeDecodeError): st.error('Could not read the CSV. Check that feature values are numeric and the file is UTF-8 encoded.')
with research:
 st.subheader('My research')
 st.write('Machine Learning-Based Classification of Soccer Attacking Phases Using Spatiotemporal Tracking Features')
 st.write('This project uses SkillCorner open tracking data from 10 Australian A-League matches to distinguish build-up, creation, direct attack, fast transition, and finishing.')
 st.write('The labels came from SkillCorner’s AI-based Phases of Play system. My model attempts to reproduce those labels using a smaller set of tracking features; they are not independent labels created by coaches for this project.')
 st.subheader('Results reported in the paper')
 results=pd.DataFrame({'Model':['Original Random Forest','Enhanced Random Forest','Original XGBoost','Enhanced XGBoost','SVC','MLP','KNN'],'Accuracy':[.912,.912,.914,.919,.879,.858,.756],'Macro F1':[.863,.873,.870,.864,.825,.790,.649]})
 st.dataframe(results.style.format({'Accuracy':'{:.1%}','Macro F1':'{:.3f}'}),hide_index=True,width='stretch')
 st.write('Enhanced XGBoost had the highest accuracy (91.9%). Enhanced Random Forest had the highest macro F1 (0.873), which gives each tactic equal weight. Fast transition was the hardest category: the paper reports 13 correct predictions out of 25 test examples.')
 st.subheader('Model running in this app')
 st.write(f"The deployed XGBoost model was retrained from the supplied notebook configuration on {meta['train_phases']:,} phases. Matches 2015213 and 1899585 were held out entirely for evaluation ({meta['test_phases']:,} phases). The scores above were recomputed for this deployed model.")
 st.plotly_chart(px.imshow(meta['confusion'],x=classes,y=classes,text_auto=True,labels={'x':'Predicted tactic','y':'Reference tactic','color':'Phases'},color_continuous_scale='Greens',title='Held-out confusion matrix'),width='stretch')
 st.write('The 18 features describe ball position and progress, attacking and defending shape, nearby defenders, phase timing, and progress speed. Match and phase identifiers are excluded from training inputs.')
 st.subheader('Limitations and next steps')
 st.write('The study covers 10 matches from one league. Averaging features across each phase loses the order of player and ball movements. More matches, other leagues, and sequence-based models would help test how well the approach generalizes.')
 st.caption('Based on the local paper draft TS8_ Final Research Paper Task Sheet-4.pdf, sections 2–7.')
 st.markdown('[SkillCorner open data](https://github.com/SkillCorner/opendata)')
